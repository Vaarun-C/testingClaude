from helper import get_STFT_data, plot_preprocessed_stft
import matplotlib.pyplot as plt
import numpy as np

import os

folder = "Simulator/Simulations/noise_floor_60dB_NoiseInsideBeam/spiral/128mics_5noises/np_data"
beamformed_output = sorted(os.listdir(f"{folder}/beam"), key=lambda x: int(x[:-4]))
ref_output = sorted(os.listdir(f"{folder}/ref"), key=lambda x: int(x[:-4]))

for file in beamformed_output:
    data = np.load(f"{folder}/beam/{file}", allow_pickle=True)
    stft = get_STFT_data(data)
    
    plot_preprocessed_stft(stft, name=file)
    plt.show()