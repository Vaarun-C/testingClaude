import matplotlib.pyplot as plt
import numpy as np
import librosa

class Plotter:
    def plot_waveform(self, audio_data, sr, ax=None):
        if ax is None:
            fig = plt.figure(figsize=(15, 5))
            ax = fig.add_subplot(111)

        librosa.display.waveshow(audio_data, sr=sr, ax=ax)

    def plot_spectrogram(self, audio_data, sr, ax=None, window_size=1024, overlap=0.75):
        if ax is None:
            fig = plt.figure(figsize=(15, 5))
            ax = fig.add_subplot(111)
        
        S = np.abs(librosa.stft(audio_data))
        img = librosa.display.specshow(librosa.amplitude_to_db(S,

                                                       ref=np.max),

                               y_axis='log', x_axis='time', ax=ax)
        
        return audio_data, sr, ax
    
    def plot_cqt(self, audio_data, sr, ax=None):
        C = np.abs(librosa.cqt(audio_data, sr=sr))
        img = librosa.display.specshow(librosa.amplitude_to_db(C, ref=np.max),
                               sr=sr, x_axis='time', y_axis='cqt_note', ax=ax)
        return audio_data, sr, ax
    
    def plot_vqt(self, audio_data, sr, ax=None):
        n_bins = 36
        chroma_vq = librosa.feature.chroma_vqt(y=audio_data, sr=sr,
                                       intervals='ji5',
                                       bins_per_octave=n_bins)
        
        img = librosa.display.specshow(chroma_vq, y_axis='chroma_fjs', x_axis='time',
                               ax=ax, bins_per_octave=n_bins,
                               intervals='ji5')
        
        return audio_data, sr, ax
    
    def plot_preprocessed_stft(self, X, fig=None, ax=None, original_sr=33203, hop_length=332):
        # Take the first example (removing the batch dimension)
        stft_matrix = X#.squeeze()
        
        # Calculate time axis parameters
        num_frames = stft_matrix.shape[1]
        duration = (num_frames * hop_length) / original_sr  # Total duration in seconds
        
        if fig is None or ax is None:
            fig = plt.figure(figsize=(12, 8))
            ax = plt.gca()
        else:
            ax.clear()

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
        ax.set_yticks(y_positions)
        ax.set_yticklabels(y_labels)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels)
        
        # Only add colorbar if it doesn't exist
        if not hasattr(fig, 'colorbar'):
            fig.colorbar = plt.colorbar(img, format='%.2f')

        ax.set_title('Preprocessed STFT Spectrogram')
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Frequency (Hz)')
        
        plt.tight_layout()
        
        return fig, ax

    def plot_chroma_cens(self, audio_data, sr, ax=None):
        chroma_cens = librosa.feature.chroma_cens(y=audio_data, sr=sr)
        librosa.display.specshow(chroma_cens, y_axis='chroma', x_axis='time', ax=ax)        
        return audio_data, sr, ax