import argparse
import json
import logging
import pathlib
import time

import numpy as np
import scipy.interpolate
import scipy.stats
import torch
import torch.autograd
import torch.linalg
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


def frequency_to_weight(freqs):
    table = np.array(ISO_226_LU)
    reference_freqs = table[:, 0]
    gain_db = table[:, 1]
    log_reference_freqs = np.log(reference_freqs)
    interpolator = scipy.interpolate.interp1d(
        log_reference_freqs,
        gain_db,
        bounds_error=False,
        fill_value=(log_reference_freqs[0], log_reference_freqs[-1]),
    )
    return db_to_linear(interpolator(np.log(freqs)))


def signed_pow(base, exponent):
    return torch.sign(base) * torch.abs(base) ** exponent


def db_to_linear(db):
    return 10 ** (db / 20)


def linear_to_db(linear):
    if isinstance(linear, torch.Tensor):
        return 20 * torch.log10(linear)
    return 20 * np.log10(linear)


P_SCALE = db_to_linear(60.0)


def linear_to_p(linear):
    """Convert linear amplitude units to "P-units," a made-up psychoacoustic unit of volume
    optimized for gradient descent on the differences between spectrogram amplitudes.

    The linear scale is normalized so that 1.0 is a comfortable listening level of about 60 dB.

    Although P-units are currently very similar to dB, I have decided not to explicitly call this a
    log-spectrogram or dB spectrogram because the units might change later.
    """
    return 20 * torch.log10(1 + torch.abs(linear) * P_SCALE) * torch.sign(linear)


def p_to_linear(p):
    """Convert P-units (see docs for linear_to_p) back to linear units."""
    return ((10 ** torch.abs(p / 20)) - 1) * torch.sign(p) / P_SCALE


def get_spectral_flatness(p_spectrum, axis=0):
    return (
        torch.mean(p_spectrum, dim=axis)
        / (1 + torch.log(torch.mean(torch.exp(p_spectrum), dim=axis)))
    )


def print_p_spectrogram_summary_stats(prefix, p_spectrogram):
    s = p_spectrogram.detach()
    logger.info(
        f"{prefix}: "
        f"mean = {torch.mean(s):.2f} p, "
        f"median = {torch.median(s):.2f} p, "
        f"stdev = {torch.std(s):.2f} p"
    )


def get_cwt(array, sample_rate, scales):
    """Compute the Continuous Wavelet Transform at a number of scales. The last dimension of the
    input array is assumed to be time.
    """
    bins = []
    for scale in scales:
        window_size = int(scale * sample_rate)
        hop_size = window_size // 4
        padded = torch.nn.functional.pad(
            array, (window_size // 2, window_size // 2), mode="reflect"
        )
        window = (
            torch.hann_window(window_size)
            * torch.sin(2 * torch.pi * torch.arange(window_size) / window_size)
        )
        window = window / torch.sqrt(torch.sum(torch.square(window)))
        # Dimensions: (..., frame, time)
        windows = padded.unfold(-1, window_size, hop_size)
        while window.ndim < windows.ndim:
            window = torch.unsqueeze(window, dim=0)
        windows = windows * window
        bin_ = torch.abs(torch.sum(windows, dim=-1))
        bins.append(bin_)
    return bins


def get_spectrogram_reference_level(spectrogram, sample_rate):
    t = torch.arange(spectrogram.hop_length * 50)
    signal = torch.sin(441.0 * t * 2 * torch.pi)
    return torch.max(torch.sum(spectrogram.forward(signal), dim=0))


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
        target_linear_spectrogram = self.spectrogram.forward(audio)
        self.scaling_factor = torch.mean(
            torch.linalg.vector_norm(target_linear_spectrogram, ord=4, dim=0)
        )
        self.target_p_spectrogram = linear_to_p(target_linear_spectrogram / self.scaling_factor)
        if torch.any(torch.isnan(self.target_p_spectrogram)):
            raise ValueError("NaN detected in p-spectrogram")

        self.frame_rate = self.sample_rate / self.hop_length

        target_features_unnormalized = self.get_unnormalized_features(self.target_p_spectrogram)
        self.feature_weights = {
            "spectrogram_energy": 30.0,
            "spectral_flux_energy": 10.0,
            "spectral_flatness_energy": 10.0,
            "covariance": 5.0,
            "cwt_energy": 5.0,
        }
        self.feature_rms = {
            key: torch.sqrt(torch.mean(torch.square(value)))
            for key, value in target_features_unnormalized.items()
        }
        self.normalization_factors = {
            key: self.feature_weights.get(key, 0.0) / self.feature_rms[key]
            for key, value in target_features_unnormalized.items()
        }
        self.target_features = self.normalize_features(target_features_unnormalized)

        initial_guess = (
            torch.clip(torch.randn_like(self.target_p_spectrogram), -3, 3)
            * torch.std(self.target_p_spectrogram)
            + torch.median(self.target_p_spectrogram)
        )
        initial_guess = torch.clamp(initial_guess, min=0)

        self.estimated_p_spectrogram = torch.nn.Parameter(initial_guess)

    def compute_p_spectrogram(self, audio):
        return linear_to_p(
            self.spectrogram.forward(audio)
            / self.scaling_factor
        )

    def debug_features(self):
        for key in self.target_features.keys():
            weight = self.feature_weights.get(key, 0.0)
            if weight != 0:
                logger.debug(f"{key}: target RMS = {self.feature_rms[key]}")

    def get_stats(self, tensor: torch.Tensor, prefix: str, weights: float | torch.Tensor = 1.0):
        """Compute a standard set of statistics over the final axis of a tensor: energy (mean),
        variance, skewness, kurtosis.
        """
        energy = torch.mean(tensor, dim=-1) * weights
        variance = torch.var(tensor, dim=-1) ** (1 / 2) * weights
        skewness = signed_pow(torch.sum((tensor - energy[..., None]) ** 3, dim=-1), 1 / 3) * weights
        kurtosis = torch.sum((tensor - energy[..., None]) ** 4, dim=-1) ** (1 / 4) * weights
        return {
            f"{prefix}_energy": energy,
            f"{prefix}_variance": variance,
            f"{prefix}_skewness": skewness,
            f"{prefix}_kurtosis": kurtosis,
        }

    def get_unnormalized_features(self, p_spectrogram):
        result = {}

        bin_indices = np.arange(p_spectrogram.shape[0])
        bin_freqs = np.linspace(
            0, self.sample_rate / 2, p_spectrogram.shape[0]
        )

        # Dimensions of s: (frequency, time)
        # Remove DC and Nyquist
        frequency_slice = slice(1, -1)
        s_p = p_spectrogram[frequency_slice, :]
        bin_indices = bin_indices[frequency_slice]
        bin_freqs = bin_freqs[frequency_slice]
        weights = torch.from_numpy(frequency_to_weight(bin_freqs))

        # s_p_weighted approximates a spectrum weighted for equal loudness.
        # This is used to evaluate energy.
        # Note that we can add because p-units are approximately logarithmic.
        s_p_weighted = s_p + linear_to_p(weights)[:, None]

        # s_p_emphasized is more subjective and attempts to scale each frequency bin by how much
        # the difference contributes to the loss function.
        emphasis = torch.sqrt(weights)
        s_p_emphasized = s_p * emphasis[:, None]

        result.update(self.get_stats(s_p_weighted, prefix="spectrogram", weights=emphasis))

        # energy_by_frame = torch.mean(s_p_weighted, dim=0)
        # result["spectral_variance"] = torch.mean(torch.var(s_p_weighted, dim=0)) ** (1 / 2)
        # result["spectral_skewness"] = signed_pow(
        #     torch.mean((s_p_weighted - energy_by_frame[None, ...]) ** 3),
        #     1 / 3
        # )
        # result["spectral_kurtosis"] = torch.mean(
        #     torch.mean((s_p_weighted - energy_by_frame[None, ...]) ** 4)
        #  ) ** (1 / 4)

        spectral_flatness = get_spectral_flatness(s_p_weighted)
        result.update(self.get_stats(spectral_flatness, prefix="spectral_flatness"))

        spectral_flux = torch.abs(torch.diff(s_p_emphasized, dim=1))
        result.update(self.get_stats(spectral_flux, prefix="spectral_flux"))

        # result.update(self.get_stats(energy_by_frame, prefix="energy"))

        # Use squaring to make louder partials more important for covariance than quieter partials.
        covariance = torch.cov(s_p * s_p)
        result["covariance"] = covariance

        s_p_emphasized_2 = s_p_emphasized * s_p_emphasized
        transient = torch.sqrt(
            torch.mean(s_p_emphasized_2[:, :-1] / (s_p_emphasized_2[:, 1:] + 1e-3), dim=-1)
        )
        # result["transient"] = transient

        scales = np.geomspace(1.0, 1 / (self.frame_rate / 4), 10, endpoint=True)
        # CWT is an ordinary Python list of (frequency, time) indexed by scale
        # It's a list because each scale has a different hop size and each time axis has a different
        # size.
        cwt = get_cwt(s_p, self.frame_rate, scales)

        cwt_stats: dict = {}
        for scale, cwt_bin in zip(scales, cwt):
            # A dict where each value has dimension (frequency,)
            cwt_stats_for_scale = self.get_stats(cwt_bin, prefix="cwt", weights=emphasis)
            # weight = 1.0
            weight = 1.0
            for key, value in cwt_stats_for_scale.items():
                cwt_stats.setdefault(key, [])
                weighted_value = value * weight
                cwt_stats[key].append(weighted_value)
        # cwt_stats is a dict where each key is a different statistical feature and each value
        # has dimensions (scale, frequency).
        for key, value in cwt_stats.items():
            cwt_stats[key] = torch.stack(value)
        result.update(cwt_stats)

        for key, value in result.items():
            if torch.any(torch.isnan(value)):
                raise ValueError(f"nan found in feature '{key}'")

        return result

    def normalize_features(self, features):
        return {
            key: value * self.normalization_factors[key]
            for key, value in features.items()
        }

    def get_features(self, spectrogram):
        return self.normalize_features(self.get_unnormalized_features(spectrogram))

    def forward(self):
        return self.get_features(self.estimated_p_spectrogram)

    def get_p_spectrogram(self):
        # Clip off the first few opening and ending frames as edge artifacts can happen.
        return self.estimated_p_spectrogram[:, 3:-3].detach()


def resynthesize(
    audio,
    sample_rate,
    max_iterations=100,
    target_snr_db=60,
):
    start = time.time()

    model = ResynthesisFeatures(audio, sample_rate)
    print_p_spectrogram_summary_stats(
        "Target spectrogram", model.target_p_spectrogram
    )

    model.debug_features()

    optimizer = torch.optim.Rprop(model.parameters(), lr=1.0)

    def norm(features):
        result = torch.tensor(0.0)
        for value in features.values():
            result += torch.linalg.vector_norm(value)
        return result

    reference_error = norm(model.target_features)

    def error_function(actual, expected):
        result = torch.tensor(0.0)
        for key in expected.keys():
            result += torch.linalg.vector_norm(
                expected[key] - actual[key]
            )
        return result / reference_error

    logger.debug("Building SNR table...")

    snr_table_linear = db_to_linear(np.linspace(10.0, 60.0, 10))
    inv_snr_table_linear = 1 / snr_table_linear
    reference_rms = torch.sqrt(torch.mean(torch.square(audio)))

    snr_table_error = []
    for snr in snr_table_linear:
        noise = torch.normal(0.0, 1.0, audio.shape) * reference_rms / snr
        noisy_audio = audio + noise
        noisy_features = model.get_features(model.compute_p_spectrogram(noisy_audio))
        snr_table_error.append(
            error_function(noisy_features, model.target_features)
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

    last_error = None
    best_error = np.inf
    p_spectrogram = None
    error_history = []
    snr_history = []
    try:
        for iteration_number in range(1, max_iterations + 1):
            logger.info(f"--- Iteration #{iteration_number} ---")
            prediction = model.forward()
            error = error_function(prediction, model.target_features)
            error.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
            if last_error is None:
                step_type = "initial"
            elif error < last_error:
                step_type = "better"
            else:
                step_type = "worse"
            if error < best_error:
                p_spectrogram = model.get_p_spectrogram()
                best_error = error

            error_float = float(error.detach())
            inv_snr_linear = snr_interpolator(error_float)
            snr_db = linear_to_db(1 / inv_snr_linear)

            error_history.append(error_float)
            snr_history.append(snr_db)
            logger.info(
                f"Error = {error_float * 100:.2f}% ({step_type}), "
                f"estimated SNR = {snr_db:.2f} dB, "
                f"gradient norm = {grad_norm:.2e}"
            )
            if error_float > 1e10 or np.isnan(error_float):
                raise ValueError("Very high relative error, something is wrong")
            if snr_db > target_snr_db:
                logger.info("Target SNR reached.")
                break
            last_error = error_float
            optimizer.step()
            optimizer.zero_grad()
        else:
            logger.info(f"Max iterations reached.")
    except KeyboardInterrupt:
        logger.warning("Interrupted by user.")

    spectrogram = p_to_linear(p_spectrogram) * model.scaling_factor
    # Zero out DC and Nyquist
    spectrogram[0, :] = 0
    spectrogram[-1, :] = 0

    logger.debug("Running phase reconstruction.")
    griffin_lim = torchaudio.transforms.GriffinLim(**model.spectrogram_kwargs)
    audio_out = griffin_lim.forward(spectrogram)

    peak = torch.max(torch.abs(audio_out))
    if peak > 1.0:
        gain_db = linear_to_db(1 / peak)
        logger.warning(f"Peak exceeds 0 dBFS, applying {gain_db:.2f} dB gain")
        audio_out /= peak

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