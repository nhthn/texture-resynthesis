import pathlib

import torch
import torch.nn.functional
import torch.optim
import torchaudio


class Spectrogram(torch.nn.Module):

    def __init__(self, audio, sample_rate):
        super().__init__()
        hop_length = 64
        self.spectrogram_kwargs = {
            "n_fft": 2048,
            "hop_length": hop_length,
            "power": 1.0,
        }
        self.spectrogram = torchaudio.transforms.Spectrogram(**self.spectrogram_kwargs)
        self.target_spectrogram = self.spectrogram.forward(audio)

        self.spectrogram_2 = torchaudio.transforms.Spectrogram(
            n_fft=int(2.0 * sample_rate / hop_length),
            power=1.0,
        )

        target_features_unnormalized = self.get_features(self.target_spectrogram)
        self.feature_weights = {
            "spectrogram_energy": 8.0,
            "spectral_flux": 5.0,
            "modulation_spectrogram_energy": 10.0,
            "covariance": 10.0,
            "spectral_flatness": 10.0,
        }
        self.normalization_factors = {
            key: self.feature_weights.get(key, 1.0) / torch.sqrt(torch.mean(torch.square(value)))
            for key, value in target_features_unnormalized.items()
        }
        self.target_features = self.normalize_features(target_features_unnormalized)
        self.estimated_spectrogram = torch.nn.Parameter(
            torch.rand(self.target_spectrogram.shape)
        )

    def get_stats(self, spectrogram, prefix):
        energy = torch.mean(spectrogram, axis=-1)
        variance = torch.var(spectrogram, axis=-1)
        skewness = torch.sum((spectrogram - energy[..., None]) ** 3, axis=-1)
        kurtosis = torch.sum((spectrogram - energy[..., None]) ** 4, axis=-1)
        return {
            f"{prefix}_energy": energy,
            f"{prefix}_variance": variance,
            f"{prefix}_skewness": skewness,
            f"{prefix}_kurtosis": kurtosis,
        }

    def get_features(self, spectrogram):
        result = {}
        # Dimensions of s: (frequency, time)
        # Remove DC and Nyquist
        s = spectrogram[1:-1, :]
        result.update(self.get_stats(s, prefix="spectrogram"))

        energy_by_frame = torch.mean(spectrogram, axis=0)
        result["spectral_variance"] = torch.mean(torch.var(spectrogram, axis=0))
        result["spectral_skewness"] = torch.mean((spectrogram - energy_by_frame[None, ...]) ** 3)
        result["spectral_kurtosis"] = torch.mean((spectrogram - energy_by_frame[None, ...]) ** 4)
        result["spectral_flux"] = torch.mean(torch.abs(torch.diff(spectrogram, axis=1)), axis=1)

        result.update(self.get_stats(energy_by_frame, prefix="energy"))

        covariance = torch.cov(s)
        result["covariance"] = covariance

        # Dimensions of s2: (audio frequency, modulation frequency, time)
        s2 = self.spectrogram_2.forward(s)
        # Ignore dc, as it is already captured by energy.
        s2 = s2[:, 1:, :]
        # 1/f weighting for modulation frequency.
        s2 *= (1 / (1 + torch.arange(s2.shape[1])))[None, :, None]
        result.update(self.get_stats(s2, prefix="modulation_spectrogram"))

        return result

    def normalize_features(self, features):
        return torch.concatenate([
            torch.flatten(value * self.normalization_factors[key])
            for key, value in features.items()
        ])

    def forward(self):
        return self.normalize_features(
            self.get_features(self.estimated_spectrogram)
        )


def resynthesize(audio, sample_rate):
    model = Spectrogram(audio, sample_rate)

    learning_rate = 1.0
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    last_loss = None
    spectrogram = None

    try:
        for i in range(100):
            prediction = model.forward()
            loss = torch.nn.functional.mse_loss(prediction, model.target_features)
            loss.backward()
            if last_loss is None:
                step_type = "initial"
            elif loss < last_loss:
                step_type = "better"
                spectrogram = model.estimated_spectrogram.detach()
                learning_rate *= 1.1
            else:
                step_type = "worse"
                learning_rate *= 0.5
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            last_loss = loss
            print(loss)
            print(step_type)
            optimizer.step()
            optimizer.zero_grad()
    except KeyboardInterrupt:
        pass

    # Zero out DC and Nyquist
    spectrogram[0, :] = 0
    spectrogram[-1, :] = 0

    print("Running phase reconstruction...")
    griffin_lim = torchaudio.transforms.GriffinLim(**model.spectrogram_kwargs)
    audio_in = griffin_lim.forward(spectrogram)
    return audio_out


ROOT = pathlib.Path(__file__).resolve().parent
EXAMPLE_FILES = ROOT / "example_files"
IN_FILES = ROOT / "in_files"
OUT_FILES = ROOT / "out_files"


def main():
    OUT_FILES.mkdir(exist_ok=True)
    IN_FILES.mkdir(exist_ok=True)
    for file_name in EXAMPLE_FILES.glob("*.wav"):
        in_file_name = IN_FILES / (file_name.stem + ".wav")
        out_file_name = OUT_FILES / (file_name.stem + ".resynthesized.wav")
        audio_in, sample_rate = torchaudio.load(str(file_name))
        audio_in = audio_in[0, :int(sample_rate * 5.0)]
        audio_out = resynthesize(audio_in, sample_rate)
        torchaudio.save(str(in_file_name), audio_in[None, :], sample_rate)
        torchaudio.save(str(out_file_name), audio_out[None, :], sample_rate)


if __name__ == "__main__":
    main()