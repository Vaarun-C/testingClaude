from scipy.signal import butter, filtfilt
import numpy as np

def high_pass(signal:np.ndarray, cutoff_freq:int, sample_rate:int, order:int=5):
    """
    Applies a high-pass filter to the input signal.

    Parameters:
    - signal (np.ndarray): The input signal to be filtered.
    - cutoff_freq (int): The cutoff frequency for the high-pass filter in Hz.
    - sample_rate (int): The sample rate of the signal in Hz.
    - order (int, optional): The order of the butterworth filter. Default is 5.

    Returns:
    - np.ndarray: The filtered signal.
    """
    nyquist = 0.5 * sample_rate  # Nyquist frequency
    normal_cutoff = cutoff_freq / nyquist  # Normalized cutoff frequency
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal


def low_pass_filter(data, cutoff_freq, sample_rate, order=5):
    """
    Applies a low-pass filter to the input signal.

    Parameters:
    - signal (np.ndarray): The input signal to be filtered.
    - cutoff_freq (int): The cutoff frequency for the low-pass filter in Hz.
    - sample_rate (int): The sample rate of the signal in Hz.
    - order (int, optional): The order of the butterworth filter. Default is 5.

    Returns:
    - np.ndarray: The filtered signal.
    """
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data


def bandpass_filter(signal: np.ndarray, low_cutoff: int, high_cutoff: int, sample_rate: int, order: int = 5) -> np.ndarray:
    """
    Applies a bandpass Butterworth filter to the input signal.

    Parameters:
    - signal (np.ndarray): The input signal to be filtered.
    - low_cutoff (int): The lower cutoff frequency for the bandpass filter in Hz.
    - high_cutoff (int): The higher cutoff frequency for the bandpass filter in Hz.
    - sample_rate (int): The sample rate of the signal in Hz.
    - order (int, optional): The order of the filter. Default is 5.

    Returns:
    - np.ndarray: The filtered signal with frequencies between low_cutoff and high_cutoff passed through.
    """
    nyquist = 0.5 * sample_rate  # Nyquist frequency
    low_normal_cutoff = low_cutoff / nyquist  # Normalized lower cutoff frequency
    high_normal_cutoff = high_cutoff / nyquist  # Normalized higher cutoff frequency

    # Design the bandpass Butterworth filter
    b, a = butter(order, [low_normal_cutoff, high_normal_cutoff], btype='band', analog=False)

    # Apply the filter to the signal
    filtered_signal = filtfilt(b, a, signal)
    
    return filtered_signal


def band_stop_filter(signal: np.ndarray, low_cutoff: int, high_cutoff: int, sample_rate: int, order: int = 5) -> np.ndarray:
    """
    Applies a band-stop (notch) Butterworth filter to the input signal.

    Parameters:
    - signal (np.ndarray): The input signal to be filtered.
    - low_cutoff (int): The lower cutoff frequency for the band-stop filter in Hz.
    - high_cutoff (int): The higher cutoff frequency for the band-stop filter in Hz.
    - sample_rate (int): The sample rate of the signal in Hz.
    - order (int, optional): The order of the filter. Default is 5.

    Returns:
    - np.ndarray: The filtered signal with frequencies between low_cutoff and high_cutoff attenuated.
    """
    nyquist = 0.5 * sample_rate  # Nyquist frequency
    low_normal_cutoff = low_cutoff / nyquist  # Normalized lower cutoff frequency
    high_normal_cutoff = high_cutoff / nyquist  # Normalized higher cutoff frequency

    # Design the band-stop Butterworth filter
    b, a = butter(order, [low_normal_cutoff, high_normal_cutoff], btype='bandstop', analog=False)

    # Apply the filter to the signal
    filtered_signal = filtfilt(b, a, signal)
    
    return filtered_signal
