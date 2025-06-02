import numpy as np
from scipy.fft import fft
from scipy.fftpack import dct

def calculate_ste(frames):
    # L is the length of each frame
    L = frames.shape[1]
    
    # Calculate STE for each frame
    # 1. Square the absolute values: |s(i)|^2
    squared_values = np.abs(frames) ** 2
    
    # 2. Sum over each frame
    frame_sums = np.sum(squared_values, axis=1)
    
    # 3. Divide by frame length L
    ste = (1/L) * frame_sums
    
    return ste

def calculate_zcr(frames):
    num_frames = frames.shape[0]
    frame_length = frames.shape[1]
    zcr_values = np.zeros(num_frames)
    
    for i in range(num_frames):
        signs = np.sign(frames[i])
        sign_changes = np.abs(np.diff(signs))

        # Calculate ZCR
        zcr_values[i] = np.sum(sign_changes) / (2 * (frame_length - 1))
    
    return zcr_values

def calculate_temporal_centroid(frames):
    num_frames = frames.shape[0]
    frame_length = frames.shape[1]
    temporal_centroids = np.zeros(num_frames)
    
    # Create time index array (h in the formula)
    time_index = np.arange(frame_length)
    
    for i in range(num_frames):
        frame = frames[i]
        # Calculate numerator: sum(h * s(i))
        numerator = np.sum(time_index * frame)
        # Calculate denominator: sum(s(i))
        denominator = np.sum(frame)
        
        # Avoid division by zero
        if denominator != 0:
            temporal_centroids[i] = numerator / denominator
        else:
            temporal_centroids[i] = 0
            
    return temporal_centroids

def calculate_spectral_centroid(frames, sample_rate):
    num_frames = frames.shape[0]
    frame_length = frames.shape[1]
    centroids = np.zeros(num_frames)
    
    frequencies = np.fft.rfftfreq(frame_length, d=1/sample_rate)
    
    for i in range(num_frames):
        # Calculate FFT (no need to window as frames are pre-windowed)
        spectrum = np.fft.rfft(frames[i])
        
        # Calculate magnitude spectrum (power spectrum)
        magnitudes = np.abs(spectrum) ** 2
        
        # Calculate spectral centroid for this frame
        if np.sum(magnitudes) > 0:  # Avoid division by zero
            centroids[i] = np.sum(frequencies * magnitudes) / np.sum(magnitudes)
        else:
            centroids[i] = 0
            
    return centroids

def calculate_spectral_rolloff(frames, sample_rate=33203, beta=0.9):
    num_frames = frames.shape[0]
    rolloffs = np.zeros(num_frames)
    
    for i in range(num_frames):
        # Calculate power spectrum
        spectrum = np.abs(fft(frames[i]))
        # We only need the positive frequencies (first half)
        spectrum = spectrum[:len(spectrum)//2]
        
        # Calculate power spectrum
        power_spectrum = np.square(spectrum)
        
        # Calculate total energy
        total_energy = np.sum(power_spectrum)
        
        # Calculate cumulative sum of the power spectrum
        cumsum = np.cumsum(power_spectrum)
        
        # Normalize cumulative sum
        cumsum_normalized = cumsum / total_energy
        
        # Find the first frequency bin where cumulative energy exceeds beta
        # Using np.searchsorted to find the index efficiently
        rolloff_bin = np.searchsorted(cumsum_normalized, beta)
        
        # Convert bin index to frequency
        # frequency = bin_index * (sample_rate / frame_length)
        rolloff_freq = rolloff_bin * (sample_rate / (2 * len(spectrum)))
        
        rolloffs[i] = rolloff_freq
    
    return rolloffs

def mel_to_hz(mel):
    """Convert mel frequency to Hz using the formula from the paper"""
    return 700 * (10**(mel/2595) - 1)

def hz_to_mel(freq):
    """Convert Hz to mel frequency using the formula from the paper"""
    return 2595 * np.log10(1 + freq/700)

def create_mel_filterbanks(num_filters=13, fft_size=512, sample_rate=33203):
    # Convert lowest and highest frequencies to mel scale
    low_freq_mel = hz_to_mel(0)
    high_freq_mel = hz_to_mel(sample_rate/2)
    
    # Generate equally spaced points in mel scale
    mel_points = np.linspace(low_freq_mel, high_freq_mel, num_filters + 2)
    
    # Convert mel points back to Hz
    hz_points = mel_to_hz(mel_points)
    
    # Convert Hz frequencies to FFT bin numbers
    bin_numbers = np.floor((fft_size + 1) * hz_points / sample_rate).astype(int)
    
    # Create filterbanks
    filterbanks = np.zeros((num_filters, fft_size//2 + 1))
    
    for i in range(num_filters):
        for j in range(bin_numbers[i], bin_numbers[i+1]):
            filterbanks[i, j] = (j - bin_numbers[i]) / (bin_numbers[i+1] - bin_numbers[i])
        for j in range(bin_numbers[i+1], bin_numbers[i+2]):
            filterbanks[i, j] = (bin_numbers[i+2] - j) / (bin_numbers[i+2] - bin_numbers[i+1])
            
    return filterbanks

def calculate_mfcc(frames, sample_rate=33203, num_coeffs=13):
    num_frames = frames.shape[0]
    frame_length = frames.shape[1]
    
    # Create mel filterbanks
    filterbanks = create_mel_filterbanks(num_filters=num_coeffs, 
                                       fft_size=frame_length, 
                                       sample_rate=sample_rate)
    
    # Initialize MFCC array
    mfccs = np.zeros((num_frames, num_coeffs))
    
    for i in range(num_frames):
        # Calculate power spectrum
        spectrum = np.abs(fft(frames[i]))[:frame_length//2 + 1]
        power_spectrum = np.square(spectrum)
        
        # Apply mel filterbanks
        mel_energies = np.dot(filterbanks, power_spectrum)
        
        # Take log of mel energies
        log_mel_energies = np.log(mel_energies + 1e-10)  # Add small constant to avoid log(0)
        
        # Apply DCT to get MFCCs
        mfccs[i] = dct(log_mel_energies, type=2, norm='ortho')[:num_coeffs]
    
    return mfccs

def calculate_midterm_features(frames, sample_rate=33203, midterm_window=0.2):
    ste = calculate_ste(frames)
    zcr = calculate_zcr(frames)
    temporal_centroid = calculate_temporal_centroid(frames)
    spectral_centroid = calculate_spectral_centroid(frames, sample_rate)
    spectral_rolloff = calculate_spectral_rolloff(frames, sample_rate)
    mfcc = calculate_mfcc(frames, sample_rate)
    
    frame_step = 0.01  # 10ms step between frames
    frames_per_midterm = int(midterm_window / frame_step)
    
    # Calculate number of mid-term windows
    num_midterm_windows = len(frames) // frames_per_midterm
    
    # Initialize array for mid-term features
    midterm_features = []
    
    for i in range(num_midterm_windows):
        start_idx = i * frames_per_midterm
        end_idx = start_idx + frames_per_midterm
        
        feature_stats = []
        
        # STE statistics
        feature_stats.extend([
            np.mean(ste[start_idx:end_idx]),
            np.std(ste[start_idx:end_idx])
        ])
        
        # ZCR statistics
        feature_stats.extend([
            np.mean(zcr[start_idx:end_idx]),
            np.std(zcr[start_idx:end_idx])
        ])
        
        # Temporal Centroid statistics
        feature_stats.extend([
            np.mean(temporal_centroid[start_idx:end_idx]),
            np.std(temporal_centroid[start_idx:end_idx])
        ])
        
        # Spectral Centroid statistics
        feature_stats.extend([
            np.mean(spectral_centroid[start_idx:end_idx]),
            np.std(spectral_centroid[start_idx:end_idx])
        ])
        
        # Spectral Rolloff statistics
        feature_stats.extend([
            np.mean(spectral_rolloff[start_idx:end_idx]),
            np.std(spectral_rolloff[start_idx:end_idx])
        ])
        
        # MFCC statistics (for each coefficient)
        for j in range(mfcc.shape[1]):  # For each MFCC coefficient
            feature_stats.extend([
                np.mean(mfcc[start_idx:end_idx, j]),
                np.std(mfcc[start_idx:end_idx, j])
            ])
        
        midterm_features.append(feature_stats)
    
    return np.array(midterm_features)

def get_feature_vector_size():
    """
    Returns the size of the feature vector for each mid-term window
    """
    # Count number of features:
    # 5 basic features (STE, ZCR, TC, SC, SRO) * 2 statistics = 10
    # 13 MFCC coefficients * 2 statistics = 26
    return 10 + 26  # Total 36 features