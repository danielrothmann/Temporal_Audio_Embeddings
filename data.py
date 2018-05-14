import numpy as np
from scipy.signal import stft
from gammatone import filters as gt_filters
from gammatone import gtgram
import pywt

def perform_fft(audio_samples, segment_length=1024):
    bins, segments, spectrum = stft(x=audio_samples, nperseg=segment_length)
    num_bins = bins.__len__()
    num_segments = segments.__len__()
    frequencies = np.zeros([num_segments, num_bins])
    phases = np.zeros([num_segments, num_bins])

    for segment in range(num_segments):
        for bin in range(num_bins):
            freq, phase = cart2pol(np.real(spectrum[bin][segment]), np.imag(spectrum[bin][segment]))
            frequencies[segment][bin] = freq
            phases[segment][bin] = phase

    return frequencies, phases


def make_gammatone_filters(num_bins = 1024, cutoff_low = 30, sample_rate = 44100):
    center_freqs = gt_filters.centre_freqs(sample_rate, num_bins, cutoff_low)
    gammatone_filters = gt_filters.make_erb_filters(sample_rate, center_freqs)

    return gammatone_filters


def perform_gammatone(audio_samples, filter_coeffs):
    return gt_filters.erb_filterbank(audio_samples, filter_coeffs)


def perform_gammatone_spectrogram(audio_samples, sample_rate = 44100, window_time = 0.05, hop_time = 0.025, channels = 256, cutoff_low = 20):
    return gtgram.gtgram(audio_samples, sample_rate, window_time, hop_time, channels, cutoff_low)


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi


def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y


def lin2ulaw(x, u=255):
    x = np.sign(x) * (np.log(1 + u * np.abs(x)) / np.log(1 + u))
    return x


def ulaw2lin(x, u=255.):
    max_value = np.iinfo('uint8').max
    min_value = np.iinfo('uint8').min
    x = x.astype('float64', casting='safe')
    x -= min_value
    x /= ((max_value - min_value) / 2.)
    x -= 1.
    x = np.sign(x) * (1 / u) * (((1 + u) ** np.abs(x)) - 1)
    # x = float_to_uint8(x)
    return x


def float_to_uint8(x):
    x += 1.
    x /= 2.
    uint8_max_value = np.iinfo('uint8').max
    x *= uint8_max_value
    x = x.astype('uint8')
    return x


def one_hot(x):
    return np.eye(256, dtype='uint8')[x.astype('uint8')]
