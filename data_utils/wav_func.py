from scipy.io.wavfile import write
import librosa
import numpy as np

def load_wav(file_path: str):
    return librosa.load(file_path, sr=None)

def load_wav_mono(file_path: str):
    return librosa.load(file_path, sr=None, mono=True)

def write_wav(processed, file_name, sample_rate = 33203):
    signal = processed / np.max(np.abs(processed))  # Normalize to [-1, 1]
    signal_16bit = (signal * 32767).astype(np.int16)  # Scale to 16-bit range

    write(file_name, sample_rate, signal_16bit)