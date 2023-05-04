import argparse
import json
import logging
import pathlib
import time

import numpy as np
import scipy.interpolate
import torch
import torch.nn.functional
import torch.optim
import torchaudio

logger = logging.getLogger("texture_resynthesis")

ISO_226_LU = [
    (20.0, -31.6),
    (25.0, -27.2),
    (31.5, -23.0),
    (40.0, -19.1),
    (50.0, -15.9),
    (63.0, -13.0),
    (80.0, -10.3),
    (100.0, -8.1),
    (125.0, -6.2),
    (160.0, -4.5),
    (200.0, -3.1),
    (250.0, -2.0),
    (315.0, -1.1),
    (400.0, -0.4),
    (500.0, 0.0),
    (630.0, 0.3),
    (800.0, 0.3),
    (1000.0, 0.0),
    (1250.0, -2.7),
    (1600.0, -4.1),
    (2000.0, -1.0),
    (2500.0, 1.7),
    (3150.0, 2.5),
    (4000.0, 1.2),
    (5000.0, -2.1),
    (6300.0, -7.1),
    (8000.0, -11.2),
    (10000.0, -10.7),
    (12500.0, -3.1),
]


def db_to_linear(db):
    return 10 ** (db / 20)


def linear_to_db(linear):
    return 20 * np.log10(linear)


def frequency_to_weight(freqs):
    table = np.array(ISO_226_LU)
    reference_freqs = table[:, 0]
    gain_db = table[:, 1]
    log_reference_freqs = np.log(reference_freqs)
    interpolator = scipy.interpolate.interp1d(
        log_reference_freqs, gain_db, fill_value="extrapolate"
    )
    return db_to_linear(interpolator(np.log(freqs)))


def get_spectral_flatness(spectrum, axis=0):
    log_spectrogram = torch.log(1 + torch.abs(spectrum))
    return (
        torch.exp(torch.mean(spectrum, axis=axis))
        / (1 + torch.mean(torch.abs(spectrum), axis=axis))
    )


class ResynthesisFeatures(torch.nn.Module):

    def __init__(self, audio, sample_rate):
        super().__init__()
        self.sample_rate = sample_rate
        hop_length = 64
        self.spectrogram_kwargs = {
            "n_fft": 2048,
            "hop_length": hop_length,
            "power": 1.0,
        }
        self.spectrogram = torchaudio.transforms.Spectrogram(**self.spectrogram_kwargs)
        self.target_spectrogram = self.spectrogram.forward(audio)

        spectrogram_2_nfft = int(2.0 * self.sample_rate / hop_length)
        self.spectrogram_2 = torchaudio.transforms.Spectrogram(
            n_fft=spectrogram_2_nfft,
            hop_length=spectrogram_2_nfft // 4,
            power=1.0,
        )

        target_features_unnormalized = self.get_unnormalized_features(self.target_spectrogram)
        self.feature_weights = {
            "spectrogram_energy": 8.0,
            "spectral_flux": 5.0,
            "modulation_spectrogram_energy": 10.0,
            "covariance": 10.0,
            "spectral_flatness_energy": 50.0,
            "spectral_flatness_variance": 50.0,
        }
        self.normalization_factors = {
            key: self.feature_weights.get(key, 1.0) / torch.sqrt(torch.mean(torch.square(value)))
            for key, value in target_features_unnormalized.items()
        }
        self.target_features = self.normalize_features(target_features_unnormalized)
        self.estimated_spectrogram = torch.nn.Parameter(
            torch.rand(self.target_spectrogram.shape) * torch.mean(self.target_spectrogram)
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

    def get_unnormalized_features(self, spectrogram):
        result = {}

        bin_freqs = np.linspace(
            0, self.sample_rate / 2, spectrogram.shape[0]
        )

        # Dimensions of s: (frequency, time)
        # Remove DC and Nyquist
        frequency_slice = slice(1, -1)
        s = spectrogram[frequency_slice, :]
        bin_freqs = bin_freqs[frequency_slice]
        s = s * torch.from_numpy(frequency_to_weight(bin_freqs[:, None]))
        result.update(self.get_stats(s, prefix="spectrogram"))

        energy_by_frame = torch.mean(spectrogram, axis=0)
        result["spectral_variance"] = torch.mean(torch.var(spectrogram, axis=0))
        result["spectral_skewness"] = torch.mean((spectrogram - energy_by_frame[None, ...]) ** 3)
        result["spectral_kurtosis"] = torch.mean((spectrogram - energy_by_frame[None, ...]) ** 4)

        spectral_flatness = get_spectral_flatness(s)
        result.update(self.get_stats(spectral_flatness, prefix="spectral_flatness"))

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

    def get_features(self, spectrogram):
        return self.normalize_features(self.get_unnormalized_features(spectrogram))

    def forward(self):
        return self.get_features(self.estimated_spectrogram)


def resynthesize(
    audio,
    sample_rate,
):
    start = time.time()

    model = ResynthesisFeatures(audio, sample_rate)

    optimizer = torch.optim.Rprop(model.parameters(), lr=1.0)

    last_loss = None
    spectrogram = None

    loss_function = torch.nn.functional.mse_loss
    reference_loss = loss_function(
        model.target_features,
        torch.zeros_like(model.target_features)
    )

    def loss_to_error(loss):
        return np.sqrt(float(loss / reference_loss))

    noise_floor_table_db = np.linspace(-60.0, 0.0, 10)
    noise_floor_table_linear = db_to_linear(noise_floor_table_db)
    noise_floor_table_error = []
    reference_rms = torch.sqrt(torch.mean(torch.square(audio)))

    for noise_floor in noise_floor_table_linear:
        noise = torch.normal(0, noise_floor, audio.shape) * reference_rms
        noisy_audio = audio + noise
        noisy_features = model.get_features(model.spectrogram(noisy_audio))
        noise_floor_table_error.append(
            loss_to_error(loss_function(noisy_features, model.target_features))
        )

    noise_floor_interpolator = scipy.interpolate.interp1d(
        noise_floor_table_error,
        noise_floor_table_linear,
        fill_value="extrapolate",
    )

    error_history = []
    noise_floor_history = []
    try:
        for iteration_number in range(1, 100 + 1):
            logger.info(f"--- Iteration #{iteration_number} ---")
            prediction = model.forward()
            loss = loss_function(prediction, model.target_features)
            loss.backward()
            if last_loss is None:
                step_type = "initial"
            elif loss < last_loss:
                step_type = "better"
                spectrogram = model.estimated_spectrogram.detach()
            else:
                step_type = "worse"

            error = loss_to_error(loss)
            noise_floor_linear = noise_floor_interpolator(error)
            noise_floor_db = linear_to_db(noise_floor_linear)

            error_history.append(error)
            noise_floor_history.append(noise_floor_db)
            logger.info(
                f"Error = {error * 100:.2f}% ({step_type}), "
                f"estimated noise floor = {noise_floor_db:+.2f} dB"
            )
            last_loss = loss
            optimizer.step()
            optimizer.zero_grad()
        else:
            logger.info(f"Max iterations reached.")
    except KeyboardInterrupt:
        logger.warning("Interrupted by user.")

    # Zero out DC and Nyquist
    spectrogram[0, :] = 0
    spectrogram[-1, :] = 0

    logger.debug("Running phase reconstruction.")
    griffin_lim = torchaudio.transforms.GriffinLim(**model.spectrogram_kwargs)
    audio_out = griffin_lim.forward(spectrogram)

    time_elapsed = time.time() - start
    minutes, seconds = divmod(int(time_elapsed), 60)
    time_elapsed_string = f"{minutes}m {seconds}s"

    info = {
        "error_history": error_history,
        "noise_floor_history": noise_floor_history,
        "time_elapsed": time_elapsed,
        "time_elapsed_string": time_elapsed_string,
    }
    return audio_out, info


ROOT = pathlib.Path(__file__).resolve().parent
EXAMPLE_FILES = ROOT / "example_files"
IN_FILES = ROOT / "in_files"
OUT_FILES = ROOT / "out_files"


def run_batch():
    OUT_FILES.mkdir(exist_ok=True)
    IN_FILES.mkdir(exist_ok=True)
    for file_name in EXAMPLE_FILES.glob("*.wav"):
        in_file_name = IN_FILES / (file_name.stem + ".wav")
        out_file_name = OUT_FILES / (file_name.stem + ".resynthesized.wav")
        audio_in, sample_rate = torchaudio.load(str(file_name))
        audio_in = audio_in[0, :int(sample_rate * 5.0)]
        torchaudio.save(str(in_file_name), audio_in[None, :], sample_rate)
        logging.info(f"Input file saved to {str(in_file_name)}.")
        audio_out, info = resynthesize(audio_in, sample_rate)
        logging.info(f"Resynthesis took {info['time_elapsed_string']}.")
        torchaudio.save(str(out_file_name), audio_out[None, :], sample_rate)
        logging.info(f"Output file saved to {str(out_file_name)}.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("in_file", type=str)
    args = parser.parse_args()

    in_file_path = pathlib.Path(args.in_file)
    truncated_file_path = in_file_path.parent / (in_file_path.stem + ".truncated.wav")
    out_file_path = in_file_path.parent / (in_file_path.stem + ".resynthesized.wav")
    info_file_path = in_file_path.parent / (in_file_path.stem + ".info.json")

    audio_in, sample_rate = torchaudio.load(str(in_file_path))
    audio_in = audio_in[0, :int(sample_rate * 5.0)]
    torchaudio.save(str(truncated_file_path), audio_in[None, :], sample_rate)
    logging.info(f"Truncated input file saved to {str(truncated_file_path)}.")
    audio_out, info = resynthesize(audio_in, sample_rate)
    logging.info(f"Resynthesis took {info['time_elapsed_string']}.")
    torchaudio.save(str(out_file_path), audio_out[None, :], sample_rate)
    logging.info(f"Output file saved to {str(out_file_path)}.")
    with open(str(info_file_path), "w") as file:
        json.dump(info, file, indent=4)
    logging.info(f"Info file saved to {str(info_file_path)}.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main()