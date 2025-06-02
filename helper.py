from constants import SAMPLING_RATE, MAX_CHOICES_FOR_NOISE_COORDS
from data_utils import dsp

from SC_Wind_Noise_Generator.sc_wind_noise_generator import WindNoiseGenerator
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
from scipy.signal import butter, sosfilt
import numpy as np

import random
import math

GUSTINESS = 1 # Number of speed points. One yields constant wind. High values yield gusty wind (more than 10 can sound unnatural).
WIND_PROFILE_OUTDOOR = np.array([3.45, 6.74, 5.65, 6.34, 4.00,
                                 5.88, 3.26, 3.19, 4.78, 4.16,
                                 4.67, 4.69, 4.61, 6.53, 6.05])
SEED = 1

generator = WindNoiseGenerator(
                fs=SAMPLING_RATE,
                duration=1,
                generate=True,
                wind_profile=WIND_PROFILE_OUTDOOR,
                gustiness=GUSTINESS,
                short_term_var=True,
                start_seed=SEED
            )

def generate_noise(duration:int=1.0, source=None): 
    if source is not None:
        original_data = np.load(source, allow_pickle=True)
        return dsp.normalize_signal(original_data)
    
    t = np.linspace(0, duration, int(SAMPLING_RATE * duration))

    # 1. Background noise (white noise at low level)
    white_noise = np.random.normal(0, 0.05, len(t))  # Gaussian noise
    
    # 2. Power line interference (50/60 Hz hum)
    power_line_freq = 60  # Hz (60 Hz in US, 50 Hz in many other countries)
    power_line_noise = 0.08 * np.sin(2 * np.pi * power_line_freq * t)
    
    # 3. Microphone self-noise (colored noise, more energy in lower frequencies)
    # Create pink noise (1/f noise)
    pink_noise = np.zeros_like(t)
    num_samples = len(t)
    
    # Using FFT-based method to generate pink noise
    X_white = np.random.randn(num_samples//2 + 1) + 1j * np.random.randn(num_samples//2 + 1)
    S = np.sqrt(np.arange(len(X_white)) + 1.)  # +1 to avoid division by zero
    X_pink = X_white / S
    pink_noise = np.fft.irfft(X_pink, n=num_samples) * 0.03  # Scale down the amplitude
    pink_noise
    
    # 4. Intermittent impulse noise (e.g., clicks)
    impulse_noise = np.zeros_like(t)
    # Add random clicks/pops
    num_clicks = np.random.randint(3, 8)  # 3 to 7 random clicks
    for _ in range(num_clicks):
        click_pos = np.random.randint(0, len(t))
        click_width = np.random.randint(5, 30)  # Width of the click in samples
        impulse_noise[click_pos:click_pos+click_width] = np.random.uniform(0.2, 0.5)

    # Combine all signals
    composite_noise = white_noise + power_line_noise + pink_noise + impulse_noise
    
    # Normalize to avoid clipping
    max_amplitude = np.max(np.abs(composite_noise))
    if max_amplitude > 1.0:
        composite_noise = composite_noise / max_amplitude * 0.95
        
    composite_noise -= np.mean(composite_noise)
    composite_noise = scale_to_dB(composite_noise, 60)
        
    return composite_noise

def calculate_xyz_random(min_dist:int, distance:int):
    # Randomly choose 1 coordinate and calculate the other based on distance.
    rand_num = random.random()
    if rand_num < 0.3:
        x = random.randint(min_dist, distance)
        y = random.randint(min_dist, distance)
        z = (distance**2 - x**2 - y**2)**0.5
    elif (rand_num >= 0.3) and (rand_num < 0.6):
        x = random.randint(min_dist, distance)
        z = random.randint(min_dist, distance)
        y = (distance**2 - x**2 - z**2)**0.5
    else:
        y = random.randint(min_dist, distance)
        z = random.randint(min_dist, distance)
        x = (distance**2 - y**2 - z**2)**0.5
    return (x,y,z)

def calculate_xyz_outsideBeam(min_distance, max_distance, R):
    for _ in range(MAX_CHOICES_FOR_NOISE_COORDS):
        r = random.uniform(min_distance, max_distance)
        theta = random.uniform(0, math.pi)          # angle from z-axis
        phi = random.uniform(0, 2 * math.pi)        # angle around z-axis

        x = r * math.sin(theta) * math.cos(phi)
        y = r * math.sin(theta) * math.sin(phi)
        z = r * math.cos(theta)

        # Reject point if it's inside the beam (e.g., a cone or cylinder around z-axis)
        if x**2 + y**2 > R**2:
            return [x, y, z]

    return None


def calculate_xyz_insideBeam(min_R, max_R):
    return [0,0,random.uniform(min_R, max_R)]

def process_signal(signal: np.ndarray):
    # offset_signal = dsp.pad_signal_np(
    #     signal=dsp.offset_signal(signal, -2.5),
    #     pad_size=300
    # )
    
    filtered_signal = dsp.butterworth_filter(
        signal=signal,
        sample_rate=SAMPLING_RATE,
        low_cutoff=1000,
        high_cutoff=5300,
        band_type="bandpass",
        order=5
    )         

    amplified_signal = dsp.amplify_signal(
        filtered_signal,
        gain=16,
        max_value=float('inf'),
        min_value=float('-inf')
    )
    return amplified_signal

def get_STFT_data(signal: np.ndarray):
    frame_length = int(0.02 * SAMPLING_RATE)  # ≈ 664 samples

    n_fft = frame_length  # As specified in your requirements
    hop_length = frame_length // 2  # 50% overlap ≈ 332 samples
    win_length = frame_length  # Match the 20ms frame length

    return dsp.calc_stft_normalized(audio=process_signal(signal), n_fft=n_fft, hop_length=hop_length, win_length=win_length)

def plot_preprocessed_stft(X, original_sr=33203, hop_length=332, name='Preprocessed STFT Spectrogram'):
        # Take the first example (removing the batch dimension)
        stft_matrix = X#.squeeze()

        # print(stft_matrix.shape)
        
        # Calculate time axis parameters
        num_frames = stft_matrix.shape[1]
        duration = (num_frames * hop_length) / original_sr  # Total duration in seconds
        
        plt.figure(figsize=(12, 8))
        
        # Add small epsilon to avoid division by zero
        stft_matrix = np.where(stft_matrix == 0, 1e-10, stft_matrix)
        
        img = plt.imshow(
            stft_matrix,
            aspect='auto',
            origin='lower',
            interpolation='nearest',
            cmap='viridis'
        )
        
        # Create tick positions and labels ensuring they match in length
        num_ticks = 10
        y_positions = np.linspace(0, stft_matrix.shape[0]-1, num_ticks)
        y_labels = np.round(np.linspace(0, original_sr/2, num_ticks)).astype(int)
        
        x_positions = np.linspace(0, num_frames-1, num_ticks)
        x_labels = np.round(np.linspace(0, duration, num_ticks), 2)
        
        # Set axis ticks with matching positions and labels
        plt.yticks(y_positions, y_labels)
        plt.xticks(x_positions, x_labels)
        
        plt.colorbar(img, format='%.2f')
        plt.title(name)
        plt.xlabel('Time (seconds)')
        plt.ylabel('Frequency (Hz)')
        
        plt.tight_layout()
        plt.show()

def generate_drone_signal(duration: float = 1.0, needed_db: int = 80, pitch_variation: float = 0.05):
    t = np.linspace(0, duration, int(SAMPLING_RATE * duration), endpoint=False)

    # Top frequencies (Hz) and their corresponding counts from the histogram
    top_frequencies = [1151.52, 1030.30, 1393.94, 1515.15, 1272.73,
                       3333.33, 2969.70, 3090.91, 3454.55, 3939.39,
                       3212.12, 2848.48, 5030.30, 2606.06, 2484.85,
                       1636.36, 3818.18, 3696.97, 1757.58, 1878.79]
    
    counts = [540.0, 149.0, 137.0, 95.0, 91.0,
              58.0, 49.0, 49.0, 44.0, 37.0,
              34.0, 34.0, 32.0, 30.0, 29.0,
              29.0, 24.0, 21.0, 21.0, 20.0]

    # Normalize counts to create amplitude weights (range [0, 1])
    amplitudes = np.array(counts) / np.max(counts)

    # Create the mixture signal by summing weighted sine waves with slight frequency variations.
    signal_mixture = np.zeros_like(t)
    for freq, amp in zip(top_frequencies, amplitudes):
        # Apply a random variation within ±pitch_variation
        variation = np.random.uniform(1 - pitch_variation, 1 + pitch_variation)
        freq_variant = freq * variation
        signal_mixture += amp * np.sin(2 * np.pi * freq_variant * t)

    signal_mixture -= np.mean(signal_mixture) # Remove DC Offset
    signal_mixture = scale_to_dB(signal=signal_mixture, target_db=needed_db)
    return signal_mixture

def write_wav(processed, file_name, sample_rate = 33203):
    signal = processed / np.max(np.abs(processed))  # Normalize to [-1, 1]
    signal_16bit = (signal * 32767).astype(np.int16)  # Scale to 16-bit range

    write(file_name, sample_rate, signal_16bit)

def load_noise_as_float(noise_int16_array):
    # Convert from int16 to float32 (or float64)
    noise_float = noise_int16_array.astype(np.float32)
    # Normalize to [-1, 1]
    noise_float /= 32768.0
    return noise_float

def scale_to_dB(signal, target_db):
    current_rms = np.sqrt(np.mean(signal**2))
    desired_rms = 20e-6 * 10**(target_db / 20.0)  # e.g., 0.2 Pa for 80 dB
    scaling_factor = desired_rms / current_rms
    return signal * scaling_factor

def generate_wind(target_db: float):
    waveform, wind_profile = generator.generate_wind_noise()
    wind = scale_to_dB(signal=waveform, target_db=target_db)
    return wind