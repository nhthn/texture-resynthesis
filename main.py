import argparse
import json
import logging
import pathlib
import time

import numpy as np
import scipy.interpolate
import scipy.stats
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


def safe_db_to_linear(db):
    """An alternative to traditional dB-to-linear conversion that ensures that large dB
    values do not produce huge partial derivatives of the loss function, which can cause
    major issues in the optimizer.

    I initially tried clamping the dB value to a maximum but this makes the partial
    derivative undefined and effectively freezes the bin value, resulting in very loud
    isolated "chirps." Using a linear slope above the threshold is a reasonable middle
    ground.
    """
    return torch.minimum(db_to_linear(db), torch.abs(db) + 1)


def linear_to_db(linear):
    if isinstance(linear, torch.Tensor):
        return 20 * torch.log10(linear)
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


def print_log_spectrogram_summary_stats(prefix, log_spectrogram):
    s = log_spectrogram.detach()
    logger.info(
        f"{prefix}: "
        f"mean = {torch.mean(s):.2f} dB, "
        f"median = {torch.median(s):.2f} dB, "
        f"stdev = {torch.std(s):.2f} dB"
    )


class ResynthesisFeatures(torch.nn.Module):

    def __init__(self, audio, sample_rate):
        super().__init__()
        self.sample_rate = sample_rate
        self.hop_length = 64
        self.spectrogram_kwargs = {
            "n_fft": 2048,
            "hop_length": self.hop_length,
            "power": 1.0,
        }
        self.spectrogram = torchaudio.transforms.Spectrogram(**self.spectrogram_kwargs)
        self.target_log_spectrogram = self.compute_log_spectrogram(audio)
        if torch.any(torch.isnan(self.target_log_spectrogram)):
            raise ValueError("NaN detected in log spectrogram")

        self.frame_rate = self.hop_length / self.sample_rate
        spectrogram_2_nfft = int(2.0 / self.frame_rate)
        self.spectrogram_2 = torchaudio.transforms.Spectrogram(
            n_fft=spectrogram_2_nfft,
            hop_length=spectrogram_2_nfft // 4,
            power=1.0,
        )

        target_features_unnormalized = self.get_unnormalized_features(self.target_log_spectrogram)
        self.feature_weights = {
            "spectrogram_energy": 8.0,
            "spectral_flux": 5.0,
            "modulation_spectrogram_energy": 10.0,
            "covariance": 10.0,
            "spectral_flatness_energy": 50.0,
            "spectral_flatness_variance": 50.0,
            "transient": 20.0,
        }
        self.normalization_factors = {
            key: self.feature_weights.get(key, 1.0) / torch.sqrt(torch.mean(torch.square(value)))
            for key, value in target_features_unnormalized.items()
        }
        self.target_features = self.normalize_features(target_features_unnormalized)
        self.estimated_log_spectrogram = torch.nn.Parameter(
            torch.randn_like(self.target_log_spectrogram)
            * torch.std(self.target_log_spectrogram)
            + torch.median(self.target_log_spectrogram)
        )

    def compute_log_spectrogram(self, audio):
        return linear_to_db(self.spectrogram(audio))

    def get_stats(self, tensor, prefix, weights=1.0):
        """Compute a standard set of statistics over the final axis of a tensor.
        """
        energy = torch.mean(tensor, axis=-1) * weights
        variance = torch.var(tensor, axis=-1) * weights
        skewness = torch.sum((tensor - energy[..., None]) ** 3, axis=-1) * weights
        kurtosis = torch.sum((tensor - energy[..., None]) ** 4, axis=-1) * weights
        return {
            f"{prefix}_energy": energy,
            f"{prefix}_variance": variance,
            f"{prefix}_skewness": skewness,
            f"{prefix}_kurtosis": kurtosis,
        }

    def get_unnormalized_features(self, log_spectrogram):
        result = {}

        bin_indices = np.arange(log_spectrogram.shape[0])
        bin_freqs = np.linspace(
            0, self.sample_rate / 2, log_spectrogram.shape[0]
        )

        # Dimensions of s: (frequency, time)
        # Remove DC and Nyquist
        frequency_slice = slice(1, -1)
        s_db = log_spectrogram[frequency_slice, :]
        bin_indices = bin_indices[frequency_slice]
        bin_freqs = bin_freqs[frequency_slice]
        weights = torch.from_numpy(frequency_to_weight(bin_freqs))

        result.update(self.get_stats(s_db, prefix="spectrogram"))

        s = safe_db_to_linear(s_db)
        # There is no need to weight s_db with inverse equal-loudness curves as multiplication
        # by the curve results in a translation in the dB domain, and all features except
        # energy are translation-invariant.
        s = s * weights[:, None]

        energy_by_frame = torch.mean(s, axis=0)
        result["spectral_variance"] = torch.mean(torch.var(s, axis=0))
        result["spectral_skewness"] = torch.mean((s - energy_by_frame[None, ...]) ** 3)
        result["spectral_kurtosis"] = torch.mean((s - energy_by_frame[None, ...]) ** 4)

        spectral_flatness = get_spectral_flatness(s)
        result.update(self.get_stats(spectral_flatness, prefix="spectral_flatness"))

        result["spectral_flux"] = torch.mean(torch.abs(torch.diff(s, axis=1)), axis=1)

        result.update(self.get_stats(energy_by_frame, prefix="energy"))

        covariance = torch.cov(s)
        result["covariance"] = covariance

        s_squared = s * s
        transient = torch.sqrt(torch.mean(s_squared[:, :-1] / (s_squared[:, 1:] + 1e-3), axis=-1))
        result["transient"] = transient

        # Dimensions of s2: (audio frequency, modulation frequency, time)
        s2 = self.spectrogram_2.forward(s)
        mod_freqs = torch.arange(s2.shape[1])
        # Ignore dc, as it is already captured by energy.
        s2 = s2[:, 1:, :]
        mod_freqs = mod_freqs[1:]
        # Power law weighting for modulation frequency.
        mod_weights = mod_freqs ** -0.5
        mod_weights = mod_weights[None, :]

        result.update(self.get_stats(s2, prefix="modulation_spectrogram", weights=mod_weights))

        return result

    def normalize_features(self, features):
        return torch.concatenate([
            torch.flatten(value * self.normalization_factors[key])
            for key, value in features.items()
        ])

    def get_features(self, spectrogram):
        return self.normalize_features(self.get_unnormalized_features(spectrogram))

    def forward(self):
        return self.get_features(self.estimated_log_spectrogram)

    def get_log_spectrogram(self):
        return self.estimated_log_spectrogram.detach()


def resynthesize(
    audio,
    sample_rate,
    max_iterations=100,
    target_snr_db=60,
):
    start = time.time()

    model = ResynthesisFeatures(audio, sample_rate)
    print_log_spectrogram_summary_stats(
        "Target spectrogram", model.target_log_spectrogram
    )

    optimizer = torch.optim.Rprop(model.parameters(), lr=1.0)

    last_loss = None
    log_spectrogram = None

    loss_function = torch.nn.functional.mse_loss
    reference_loss = loss_function(
        model.target_features,
        torch.zeros_like(model.target_features)
    )

    def loss_to_error(loss):
        return np.sqrt(float(loss / reference_loss))

    logger.debug("Building SNR table...")

    snr_table_linear = db_to_linear(np.linspace(10.0, 60.0, 10))
    inv_snr_table_linear = 1 / snr_table_linear
    reference_rms = torch.sqrt(torch.mean(torch.square(audio)))

    snr_table_error = []
    for snr in snr_table_linear:
        noise = torch.normal(0.0, 1.0, audio.shape) * reference_rms / snr
        noisy_audio = audio + noise
        noisy_features = model.get_features(model.compute_log_spectrogram(noisy_audio))
        snr_table_error.append(
            loss_to_error(loss_function(noisy_features, model.target_features))
        )

    regression_result = scipy.stats.linregress(snr_table_error, inv_snr_table_linear)
    r_squared = regression_result.rvalue * regression_result.rvalue
    logger.debug(
        f"Built SNR table. "
        f"Slope = {regression_result.slope:.2f} \u00B1 {regression_result.stderr:.2f}, "
        f"intercept = {regression_result.intercept:.2f} \u00B1 {regression_result.intercept_stderr:.2f}, "
        f"R^2 = {r_squared:.2f}"
    )
    r_squared_thresold = 0.9
    if r_squared < r_squared_thresold:
        logger.warning(f"R^2 of SNR table < {r_squared_thresold}. SNR estimator may be inaccurate.")
    snr_interpolator = scipy.interpolate.interp1d(snr_table_error, inv_snr_table_linear, fill_value="extrapolate")

    error_history = []
    snr_history = []
    try:
        for iteration_number in range(1, max_iterations + 1):
            logger.info(f"--- Iteration #{iteration_number} ---")
            prediction = model.forward()
            loss = loss_function(prediction, model.target_features)
            loss.backward()
            if last_loss is None:
                step_type = "initial"
            elif loss < last_loss:
                step_type = "better"
                log_spectrogram = model.get_log_spectrogram()
            else:
                step_type = "worse"

            error = loss_to_error(loss)
            inv_snr_linear = snr_interpolator(error)
            snr_db = linear_to_db(1 / inv_snr_linear)

            error_history.append(error)
            snr_history.append(snr_db)
            logger.info(
                f"Error = {error * 100:.2f}% ({step_type}), "
                f"estimated SNR = {snr_db:.2f} dB"
            )
            if error > 1e10 or np.isnan(error):
                raise ValueError("Very high relative error, something is wrong")
            if snr_db > target_snr_db:
                logger.info("Target SNR reached.")
                break
            last_loss = loss
            optimizer.step()
            optimizer.zero_grad()
        else:
            logger.info(f"Max iterations reached.")
    except KeyboardInterrupt:
        logger.warning("Interrupted by user.")

    spectrogram = db_to_linear(log_spectrogram)
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
        "snr_history": snr_history,
        "time_elapsed": time_elapsed,
        "time_elapsed_string": time_elapsed_string,
    }
    return audio_out, info


ROOT = pathlib.Path(__file__).resolve().parent
EXAMPLE_FILES = ROOT / "example_files"
IN_FILES = ROOT / "in_files"
OUT_FILES = ROOT / "out_files"


def process_one_file(in_file_path, tag=None, length_in_seconds=5.0):
    stem = in_file_path.stem
    if tag is not None:
        stem = stem + "." + tag
    truncated_file_path = in_file_path.parent / (stem + ".truncated.wav")
    out_file_path = in_file_path.parent / (stem + ".resynthesized.wav")
    info_file_path = in_file_path.parent / (stem + ".info.json")

    audio_in, sample_rate = torchaudio.load(str(in_file_path))
    audio_in = audio_in[0, :int(sample_rate * length_in_seconds)]
    torchaudio.save(str(truncated_file_path), audio_in[None, :], sample_rate)
    logger.info(f"Truncated input file saved to {str(truncated_file_path)}.")

    audio_out, info = resynthesize(audio_in, sample_rate)
    logger.info(f"Resynthesis took {info['time_elapsed_string']}.")

    torchaudio.save(str(out_file_path), audio_out[None, :], sample_rate)
    logger.info(f"Output file saved to {str(out_file_path)}.")

    with open(str(info_file_path), "w") as file:
        json.dump(info, file, indent=4)
    logger.info(f"Info file saved to {str(info_file_path)}.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("in_files", type=str, nargs="+")
    parser.add_argument("-t", "--tag", type=str)
    args = parser.parse_args()

    for file_name in args.in_files:
        in_file_path = pathlib.Path(file_name)
        process_one_file(in_file_path, tag=args.tag)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main()