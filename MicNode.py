from helper import generate_noise, calculate_xyz_outsideBeam, generate_drone_signal, generate_wind
from constants import SAMPLING_RATE, SPEED_OF_SOUND, PHYLLOTAXIS_D, PHYLLOTAXIS_S, PHYLLOTAXIS_Z, MicArrangements
from Exceptions.NoSuchArrangement import NoSuchArrangementException
from Mic import Microphone
from Source import Source

import matplotlib.pyplot as plt
import numpy as np

class Node():
    def __init__(
            self,
            num_mics:int=1,
            mic_sensitivity:int=float('inf'),
            sources_coords:list[list[int,int, int]]=[],
            num_background_noises:int=0,
            max_distance_for_noise=0,
            min_distance_for_noise=0,
            const_noise_floor = 0,
            level_of_noise = 0,
            z_percent_for_noise = 0.66,
            mic_arrangement = MicArrangements.SPIRAL
        ):

        self.__num_mics = num_mics+1 # accounting for reference mic at center
        self.combined_const_noise = generate_wind(const_noise_floor)
        self.__noise_level = level_of_noise

        self.__mics = [
            Microphone(
                sensitivity_threshold=mic_sensitivity,
                const_noise=self.combined_const_noise
            )
            for _ in range(self.__num_mics)
        ]

        self.__sources: list[Source] = []
        self.__sources_pos = sources_coords
        self.__previous_mic_overflow_data = [[] for _ in range(self.__num_mics)]
        self.__num_sources = len(sources_coords)
        self.__num_noises_bg = num_background_noises

        for _ in range(self.__num_sources):
            self.__sources.append(
                Source(
                    sound_source=generate_drone_signal(),
                    target_sr=SAMPLING_RATE
                )
            )

        for _ in range(num_background_noises):
            self.__sources.append(
                Source(
                    sound_source=generate_noise(),
                    target_sr=SAMPLING_RATE
                )
            )
            # dis = random.randint(min_distance_for_noise, max_distance_for_noise)
            noise_coords = calculate_xyz_outsideBeam(R=10, min_distance=min_distance_for_noise, max_distance=max_distance_for_noise)

            if noise_coords is None:
                print("Failed when placing noise skipping...")
            else:
                self.__sources_pos.append(noise_coords)

        print("Total number of noises placed:", len(self.__sources_pos)-self.__num_sources)
            
        self.__sources_pos = np.array(self.__sources_pos)

        if mic_arrangement == MicArrangements.SPIRAL:
            self.place_mics_spiral()
        elif mic_arrangement == MicArrangements.LINEAR:
            self.place_mics_linear()
        elif mic_arrangement == MicArrangements.LINEAR_SAME_SIZE:
            self.place_mics_linear_same_size_as_spiral()
        else:
            raise NoSuchArrangementException()

    def get_mics(self):
        return self.__mics
    
    def get_sources(self):
        return self.__sources
    
    def place_mics_linear(self):
        coords = [(0,i,0) for i in range(-(self.__num_mics-1)//2, (self.__num_mics+1)//2)]
        self.__mic_pos = np.array(coords)
        
    def place_mics_linear_same_size_as_spiral(self):
        N = self.__num_mics
        Δ = np.sqrt(N-1 + PHYLLOTAXIS_Z) / (N-1)
        coords = [
            (0, i*Δ - np.sqrt(N-1 + PHYLLOTAXIS_Z)/2, 0)
            for i in range(N)
        ]
        self.__mic_pos = np.array(coords)

    
    def place_mics_spiral(self):
        coords = [(0,0,0)] # place the reference mic at origin
        for a in range(1, self.__num_mics):
            # z coordinate is fixed at 0
            z = 0.0
            r = np.sqrt(a+PHYLLOTAXIS_Z)
            angle = ((1 + np.sqrt(5)) / PHYLLOTAXIS_D) * np.pi * a + PHYLLOTAXIS_S
            x = r * np.cos(angle)
            y = r * np.sin(angle)
            coords.append((x, y, z))
        self.__mic_pos = np.array(coords)

    def delay_and_sum_beamforming(self, signals, compensation_delays):
        aligned_signals = []
        for signal, delay in zip(signals, compensation_delays):
            # Shift in the opposite direction to compensate the introduced delay.
            aligned = np.roll(signal, -int(delay))
            if delay > 0:
                aligned[-int(delay):] = 0  # zero out wrapped-around part
            aligned_signals.append(aligned)

        self.aligned_outputs = aligned_signals
        # Average the aligned signals
        beamformed = np.mean(aligned_signals, axis=0)
        return beamformed

    def estimate_delay(self, reference, signal):
        # Compute full cross-correlation
        corr = np.correlate(signal, reference, mode='full')
        # The zero-lag index is at len(reference) - 1 in 'full' mode.
        zero_lag_index = len(reference) - 1
        # Find the lag that maximizes the cross-correlation.
        lag = np.argmax(corr) - zero_lag_index
        return lag

    def estimate_delays(self, signals):
        reference = signals[0]
        delays = []  # Reference mic has zero delay.
        for sig in signals:
            delay = self.estimate_delay(reference, sig)
            delays.append(delay)

        delays[0] = 0
        return np.array(delays)

    def beamforming(self):
        mic_outputs = self.getMicOutputs()
        estimated_delays = self.estimate_delays(mic_outputs)
        beam = self.delay_and_sum_beamforming(mic_outputs, estimated_delays)
        return beam
        
    def getSourceSounds(self):
        distances = np.linalg.norm(self.__sources_pos[:, None, :] - self.__mic_pos[None, :, :], axis=2)

        time_delays = distances/SPEED_OF_SOUND
        sample_delays = np.round(time_delays * SAMPLING_RATE).astype(int)

        for i in range(len(self.__sources)):
            sample_delays[i] = sample_delays[i] - np.min(sample_delays[i])

        # Create a buffer with enough length to store the time delayed signals
        max_possible_delay_samples = np.max(sample_delays) + 1
        buffer_length = SAMPLING_RATE + max_possible_delay_samples
        mic_inputs = [
            np.zeros((buffer_length,))
            for _ in range(self.__num_mics)
        ]

        for i in range(self.__num_mics):
            mic_inputs[i][:len(self.__previous_mic_overflow_data[i])] = self.__previous_mic_overflow_data[i]

        for source_idx, mic_idx in np.ndindex(distances.shape):
            d = distances[source_idx, mic_idx]
            delay = sample_delays[source_idx, mic_idx]
            sound = self.__sources[source_idx].produce_sound(distance=d)
            mic_inputs[mic_idx][delay:delay + len(sound)] += sound

        # Trim buffers to original length and store overflow for next time ( To simulate the mics picking up data continuously )
        for i in range(self.__num_mics):
            self.__previous_mic_overflow_data[i] = mic_inputs[i][SAMPLING_RATE:]
            mic_inputs[i] = mic_inputs[i][:SAMPLING_RATE]
            self.__mics[i].input = mic_inputs[i]

    def getMicOutputs(self):
        return np.array([mic.getOutput(noise_level=self.__noise_level) for mic in self.__mics])

     # Plotter functions
    
    def plot_microphone_array(self, ax):
        x = self.__mic_pos[:, 0]
        y = self.__mic_pos[:, 1]
        z = self.__mic_pos[:, 2]
        ax.scatter(x, y, z, color='blue', marker='o')

    def plot_sources(self, ax, plot_lines=False, plot_center_lines=True):
        sourcex = self.__sources_pos[:, 0]
        sourcey = self.__sources_pos[:, 1]
        sourcez = self.__sources_pos[:, 2]

        micx = self.__mic_pos[:, 0]
        micy = self.__mic_pos[:, 1]
        micz = self.__mic_pos[:, 2]

        ax.scatter(
            sourcex[:self.__num_sources],
            sourcey[:self.__num_sources],
            sourcez[:self.__num_sources],
            color='green',
            marker='o',
        )

        ax.scatter(
            sourcex[self.__num_sources:self.__num_sources+self.__num_noises_bg],
            sourcey[self.__num_sources:self.__num_sources+self.__num_noises_bg],
            sourcez[self.__num_sources:self.__num_sources+self.__num_noises_bg],
            color='red',
            marker='x',
        )

        if plot_center_lines:
            # Plot lines connecting the source to the center microphone
            for i in range(len(sourcex)):
                ax.plot([sourcex[i], 0], [sourcey[i], 0], [sourcez[i], 0], linestyle='dotted', color='black')

        if plot_lines:
            # Plot lines connecting the source to each microphone
            for i in range(len(sourcex)):
                for j in range(len(micx)):
                    ax.plot([sourcex[i], micx[j]], [sourcey[i], micy[j]], [sourcez[i], micz[j]], linestyle='dotted', color='black')

        ax.legend()

    def plot_preprocessed_stft(self, stft_matrix, ax, hop_length=332, name='Preprocessed STFT Spectrogram'):
        # Calculate time axis parameters
        num_frames = stft_matrix.shape[1]
        duration = (num_frames * hop_length) / SAMPLING_RATE  # Total duration in seconds

        # Add a small epsilon to avoid division by zero issues
        stft_matrix = np.where(stft_matrix == 0, 1e-10, stft_matrix)

        # Plot the STFT matrix on the provided axis
        img = ax.imshow(
            stft_matrix,
            aspect='auto',
            origin='lower',
            interpolation='nearest',
            cmap='viridis'
        )

        # Create tick positions and labels ensuring they match in length
        num_ticks = 10
        y_positions = np.linspace(0, stft_matrix.shape[0] - 1, num_ticks)
        y_labels = np.round(np.linspace(0, SAMPLING_RATE / 2, num_ticks)).astype(int)
        
        x_positions = np.linspace(0, num_frames - 1, num_ticks)
        x_labels = np.round(np.linspace(0, duration, num_ticks), 2)
        
        # Set axis ticks with matching positions and labels
        ax.set_yticks(y_positions)
        ax.set_yticklabels(y_labels)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels)

        # ax.set_ylim([0, 100])  # or adjust your tick labels
        
        # Attach a colorbar to the axis
        plt.colorbar(img, ax=ax, format='%.2f')
        
        # Set title and axis labels
        ax.set_title(name)
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Frequency (Hz)')