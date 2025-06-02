from scipy.signal import resample_poly, butter, filtfilt
import torchaudio.transforms as T
from typing import Literal
import numpy as np
import librosa
import torch

import random

def butterworth_filter(signal: np.ndarray, sample_rate: int, low_cutoff: int, high_cutoff: int, band_type: Literal['lowpass', 'highpass', 'bandpass', 'bandstop'], order: int = 5) -> np.ndarray:
    """
    Applies a bandpass Butterworth filter to the input signal.

    Parameters:
    - signal (np.ndarray): The input signal to be filtered.
    - sample_rate (int): The sample rate of the signal in Hz.
    - low_cutoff (int): The lower cutoff frequency for the bandpass filter in Hz.
    - high_cutoff (int): The higher cutoff frequency for the bandpass filter in Hz.
    - order (int, optional): The order of the filter. Default is 5.

    Returns:
    - np.ndarray: The filtered signal with frequencies between low_cutoff and high_cutoff passed through.
    """
    # print("Performing bandpass filter...")
    nyquist = 0.5 * sample_rate  # Nyquist frequency
    low_normal_cutoff = low_cutoff / nyquist  # Normalized lower cutoff frequency
    high_normal_cutoff = high_cutoff / nyquist  # Normalized higher cutoff frequency

    # Design the bandpass Butterworth filter
    b, a = butter(order, [low_normal_cutoff, high_normal_cutoff], btype=band_type, analog=False)

    # Apply the filter to the signal
    filtered_signal = filtfilt(b, a, signal)

    return filtered_signal

def resample_audio(data: np.ndarray, orig_sr: int = 33203, target_sr: int = 16000) -> np.ndarray:
    """
    Resamples the input audio signal to a target sample rate using polyphase filtering.

    Parameters:
    - data (np.ndarray): The input audio signal as a 1D NumPy array.
    - orig_sr (int, optional): The original sample rate of the audio signal in Hz. Default is 33203.
    - target_sr (int, optional): The target sample rate to resample to in Hz. Default is 16000.

    Returns:
    - np.ndarray: The resampled audio signal at the target sample rate.
    """
    # # # # # # # # # print("Performing resampling...")
    # Compute the greatest common divisor (GCD) to minimize the up/down sampling ratio
    gcd = np.gcd(orig_sr, target_sr)
    
    # Calculate upsampling and downsampling factors
    upsample_factor = target_sr // gcd  # Upsample factor
    downsample_factor = orig_sr // gcd  # Downsample factor
    
    # Resample using polyphase filtering
    resampled_data = resample_poly(data, upsample_factor, downsample_factor)

    return resampled_data

def calc_fft(audio: np.ndarray, sr: int) -> tuple:
    """
    Computes the FFT of the input audio signal and finds the peak frequency.

    Parameters:
    - audio (np.ndarray): The input audio signal as a 1D NumPy array.
    - sr (int): The sample rate of the audio signal in Hz.

    Returns:
    - tuple: A tuple containing:
        - fft_magnitude (np.ndarray): The magnitude of the FFT of the input audio signal.
        - freq (np.ndarray): The frequency bins corresponding to the FFT.
        - peak_freq (float): The frequency of the peak in the FFT (highest magnitude).
    """
    # # # # # # # # print("Performing FFT...")
    len_audio = len(audio)  # Length of the audio signal
    
    # Calculate frequency bins directly for the real FFT (positive frequencies)
    freq = np.fft.rfftfreq(len_audio, d=1/sr)
    
    # Compute the FFT and normalize it at the same time
    fft_magnitude = np.abs(np.fft.rfft(audio)) / len_audio  # Compute magnitude of FFT (normalized)
    
    # Find the index of the peak in the FFT (highest magnitude)
    peak_idx = np.argmax(fft_magnitude)
    
    # Peak frequency
    peak_freq = freq[peak_idx]
    
    return fft_magnitude, freq, peak_freq

def spectral_subtraction(y: np.ndarray,  noise_frames: int = 10) -> np.ndarray:
    """
    Perform spectral subtraction to reduce noise from the input audio signal.

    This method assumes that the first few frames of the signal contain noise-only, 
    and uses the Short-Time Fourier Transform (STFT) to estimate and subtract the noise spectrum.

    Parameters:
    - y (np.ndarray): The input audio signal as a 1D NumPy array.
    - noise_frames (int, optional): The number of initial frames to use for noise estimation. 
                                    Default is 10, meaning the first 10 frames are assumed to be noise.

    Returns:
    - np.ndarray: The denoised audio signal after performing spectral subtraction.
    """
    # # # # # # # print("Performing spectral subtraction...")
    # Compute the Short-Time Fourier Transform (STFT) of the signal
    stft_matrix = librosa.stft(y)
    
    # Estimate the noise spectrum from the first `noise_frames` frames (assuming they are noise)
    noise_estimation = np.mean(np.abs(stft_matrix[:, :noise_frames]), axis=1, keepdims=True)
    
    # Perform spectral subtraction: subtract the noise estimate from the magnitude of the STFT
    stft_denoised = np.maximum(np.abs(stft_matrix) - noise_estimation, 0) * np.exp(1j * np.angle(stft_matrix))
    
    # Reconstruct the time-domain signal by performing the inverse STFT
    y_denoised = librosa.istft(stft_denoised)
    
    return y_denoised

def calc_stft(audio: np.ndarray, n_fft: int, hop_length: int, win_length: int) -> np.ndarray:
    """
    Compute the Short-Time Fourier Transform (STFT) of an audio signal and return the 
    log-scaled and normalized magnitude spectrogram.

    Optimized for performance by reducing unnecessary memory allocations and processing steps.

    Parameters:
    - audio (np.ndarray): The input audio signal as a 1D NumPy array.
    - sr (int, optional): The sample rate of the audio signal in Hz. Default is 16000.
    - n_fft (int, optional): The number of FFT components (window size). Default is 512.
    - hop_length (int, optional): The number of samples between consecutive frames. Default is 256.

    Returns:
    - np.ndarray: The log-scaled and normalized magnitude spectrogram (STFT).
    """
    # # # # # # print("Performing STFT...")
    # Compute the STFT (Short-Time Fourier Transform) on the audio signal
    stft_matrix = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window='hann', center=True, dtype=np.complex64)

    # Calculate the magnitude and normalize it directly
    magnitude = np.abs(stft_matrix) + 1e-10  # Add a small constant to avoid log(0)
    magnitude_normalized = (magnitude - np.min(magnitude)) / (np.max(magnitude) - np.min(magnitude))

    # Convert magnitude to decibels (dB scale) directly to save on memory usage
    magnitude_db = librosa.amplitude_to_db(magnitude_normalized, ref=np.max)

    return magnitude_db

def calc_stft_normalized(audio: np.ndarray, n_fft: int, hop_length: int, win_length: int) -> np.ndarray:
    """
    Compute the Short-Time Fourier Transform (STFT) of an audio signal and return the 
    log-scaled and normalized magnitude spectrogram.

    Optimized for performance by reducing unnecessary memory allocations and processing steps.

    Parameters:
    - audio (np.ndarray): The input audio signal as a 1D NumPy array.
    - sr (int, optional): The sample rate of the audio signal in Hz. Default is 16000.
    - n_fft (int, optional): The number of FFT components (window size). Default is 512.
    - hop_length (int, optional): The number of samples between consecutive frames. Default is 256.

    Returns:
    - np.ndarray: The log-scaled and normalized magnitude spectrogram (STFT).
    """
    # # # # # print("Performing STFT normalized...")
    # Compute the STFT (Short-Time Fourier Transform) on the audio signal
    stft_matrix = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window='hann', center=True, dtype=np.complex64)

    # Calculate the magnitude and add a small constant to avoid log(0)
    magnitude = np.abs(stft_matrix) + 1e-10

    # Convert magnitude to decibels (dB scale)
    magnitude_db = librosa.amplitude_to_db(magnitude, ref=np.max)

    # Normalize dB-scaled values to [0, 1]
    magnitude_db_normalized = (magnitude_db - np.min(magnitude_db)) / (np.max(magnitude_db) - np.min(magnitude_db))

    return magnitude_db_normalized

def calc_stft_no_normalization(audio: np.ndarray, n_fft: int, hop_length: int, win_length: int) -> np.ndarray:
    """
    Compute the Short-Time Fourier Transform (STFT) of an audio signal and return the 
    log-scaled magnitude spectrogram without per-signal normalization.

    Parameters:
    - audio (np.ndarray): The input audio signal as a 1D NumPy array.
    - n_fft (int): The number of FFT components (window size).
    - hop_length (int): The number of samples between consecutive frames.
    - win_length (int): The window length for each frame.

    Returns:
    - np.ndarray: The log-scaled magnitude spectrogram (STFT) without normalization.
    """
    # Compute the STFT (Short-Time Fourier Transform) on the audio signal
    stft_matrix = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length,
                               win_length=win_length, window='hann', center=True, dtype=np.complex64)

    # Calculate the magnitude and add a small constant to avoid log(0)
    magnitude = np.abs(stft_matrix) + 1e-10

    # Convert the magnitude to decibels using a fixed reference (e.g., 1.0)
    magnitude_db = librosa.amplitude_to_db(magnitude, ref=1.0)

    return magnitude_db


def amplify_signal(signal: np.ndarray, gain: float, max_value: float = 5.5, min_value: float = -0.5) -> np.ndarray:
    """
    Amplify or attenuate the amplitude of a signal by a given gain factor.
    
    Parameters:
    - signal (np.ndarray): The input signal as a 1D NumPy array (e.g., audio waveform).
    - gain (float): The gain factor (amplification or attenuation). Values greater than 1 amplify, 
                    values between 0 and 1 attenuate, and negative values invert the signal.
    - max_value (float, optional): The maximum allowed value for the signal. Default is 5.5.
    - min_value (float, optional): The minimum allowed value for the signal. Default is -0.5.
    
    Returns:
    - np.ndarray: The amplified or attenuated signal, clipped to the specified range.
    """
    # # # # print("Performing Amplification...")
    # Apply the gain factor to the signal
    amplified_signal = signal * gain
    
    # Clip the amplified signal to the specified range (prevent overflow)
    amplified_signal = np.clip(amplified_signal, min_value, max_value)
    
    return amplified_signal

def pad_signal(signal: np.ndarray, pad_size: int = 200) -> np.ndarray:
    """
    add the padding from the beginning and end of the signal after processing.

    Parameters:
    - signal (np.ndarray): The signal to pad and process (e.g., after filtering).
    - pad_size (int): The amount of padding added to both sides of the signal.

    Returns:
    - np.ndarray: The signal with padding removed, returning it to its original length.
    """
    # # # print("Performing padding...")
    # Pad with zeros at the beginning and end of the signal
    current_length = len(signal)
    if current_length > pad_size:
        # Trim
        return signal[:pad_size]
    elif current_length < pad_size:
        # Pad
        padding_needed = pad_size - current_length
        left_pad = padding_needed // 2
        right_pad = padding_needed - left_pad
        return np.pad(signal, (left_pad, right_pad), mode='constant')
    return signal

def pad_signal_np(signal, pad_size=200):
    """
    add the padding from the beginning and end of the signal after processing.

    Parameters:
    - signal (np.ndarray): The signal to pad and process (e.g., after filtering).
    - pad_size (int): The amount of padding added to both sides of the signal.

    Returns:
    - np.ndarray: The signal with padding removed, returning it to its original length.
    """
    # Pad with zeros at the beginning and end of the signal
    return np.pad(signal, (pad_size, pad_size), mode='constant')

def unpad_signal(padded_signal: np.ndarray, pad_size: int) -> np.ndarray:
    """
    Remove the padding from the beginning and end of the signal after processing.

    Parameters:
    - padded_signal (np.ndarray): The signal after padding and processing (e.g., after filtering).
    - pad_size (int): The amount of padding added to both sides of the signal.

    Returns:
    - np.ndarray: The signal with padding removed, returning it to its original length.
    """
    # # print("Performing unpadding...")
    return padded_signal[pad_size:-pad_size]

def normalize_signal(signal: np.ndarray, range_min: int = -1, range_max: int = 1) -> np.ndarray:
    """
    Normalize the input signal to the specified range.
    
    Parameters:
        signal (numpy array): Input signal to normalize.
        range_min (float): Minimum value of the normalized range.
        range_max (float): Maximum value of the normalized range.
        
    Returns:
        numpy array: Normalized signal.
    """
    # print("Performing normalization...")
    # Calculate the min and max of the original signal
    min_val = np.min(signal)
    max_val = np.max(signal)

    if min_val == max_val:
        print("Not normalized", end='/r')
        return signal
    
    # Normalize the signal to [0, 1]
    normalized_signal = (signal - min_val) / (max_val - min_val)
    
    # Scale to the specified range
    normalized_signal = normalized_signal * (range_max - range_min) + range_min
    
    return normalized_signal

def offset_signal(signal: np.ndarray, offset_value: float) -> np.ndarray:
    """
    Apply an offset to the input signal.
    
    Parameters:
        signal (numpy array): Input signal to offset.
        offset_value (float): The value to add to each element in the signal.
        
    Returns:
        numpy array: Offset signal.
    """
    return signal + offset_value

def time_shift(audio, sr):
    if random.random() > 0.5:
        return audio
    # Randomly shift the audio signal
    shift = int(np.random.uniform(-0.2 * sr, 0.2 * sr))
    return np.roll(audio, shift)

def frequency_mask(audio):
    if random.random() > 0.5:
        return audio
        
    # Apply frequency masking
    freq_mask = T.FrequencyMasking(freq_mask_param=30)
    audio_tensor = torch.from_numpy(audio)  # Convert NumPy array to PyTorch tensor
    masked_audio = freq_mask(audio_tensor)  # Apply frequency masking
    return masked_audio.numpy()  # Convert back to NumPy array

def split_into_frames(audio_segment, sample_rate=33203):
    frame_length = int(0.020 * sample_rate)  # 20ms
    hop_length = int(0.010 * sample_rate)    # 10ms overlap
    num_frames = 1 + (len(audio_segment) - frame_length) // hop_length
    frames = np.zeros((num_frames, frame_length))
    
    hamming_window = np.hamming(frame_length)
    
    for i in range(num_frames):
        start = i * hop_length
        end = start + frame_length
        
        # Extract frame and apply Hamming window
        frame = audio_segment[start:end]
        frames[i] = frame * hamming_window
        
    return frames