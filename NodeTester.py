from constants import MicArrangements
from MicNode import Node

import matplotlib.pyplot as plt
import numpy as np

import os

# distance = 500

noise_db = 60
noise_level = 1
samples_to_plot = 500
num_microphones = 128
num_bg_noises = 5
chosen_arrangement = MicArrangements.SPIRAL

parent_dir_path = f'Simulator/Simulations/noise_floor_{noise_db}dB_NoiseOutsideBeam'

os.makedirs(parent_dir_path, exist_ok=True)
os.makedirs(f'{parent_dir_path}/{chosen_arrangement.value}', exist_ok=True)

res_path = f'{parent_dir_path}/{chosen_arrangement.value}/{num_microphones}mics_{num_bg_noises}noises/'
img_res_path = f'{res_path}/imgs'
np_res_path = f'{res_path}/np_data'
np_res_path_ref_mic = f'{np_res_path}/ref'
np_res_path_beamformed = f'{np_res_path}/beam'

os.makedirs(img_res_path, exist_ok=True)
os.makedirs(np_res_path, exist_ok=True)
os.makedirs(np_res_path_ref_mic, exist_ok=True)
os.makedirs(np_res_path_beamformed, exist_ok=True)

for distance in range(0, 1001, 100):
    print("Simulating distance", distance, 'meters...', end='\r')
    n = Node(
        sources_coords=[[0,0,distance]],
        num_mics=num_microphones,
        num_background_noises=num_bg_noises,
        const_noise_floor = noise_db,
        level_of_noise = noise_level,
        mic_arrangement = chosen_arrangement,
        max_distance_for_noise=distance//4,
        min_distance_for_noise=distance//20,
        z_percent_for_noise = 0.66 # A percentage of z where the circle on the circumference of sphere (radius=distance) will fall
    )

    sources = n.get_sources()
    mics = n.get_mics()
    n.getSourceSounds()

    # Set up the figure
    fig, (ax0, ax1, ax2, ax3, ax4) = plt.subplots(5, 1, figsize=(10, 8))

    # -----------------------------
    # First subplot: original vs. scaled Drone Signal
    # -----------------------------
    ax0.plot(sources[0].produce_sound(0)[:samples_to_plot], label='Original Drone Signal')
    scaled_signal = sources[0].produce_sound(distance=distance)
    ax0.plot(scaled_signal[:samples_to_plot], label=f'Scaled Signal')

    ax0.set_title(f'Drone Signal Comparison at Distance {distance}')
    ax0.set_xlabel('Sample Index')
    ax0.set_ylabel('Amplitude')
    ax0.legend()
    ax0.grid(True)
    
    np.save(f"{np_res_path_ref_mic}/{distance}.npy", scaled_signal)

    # -----------------------------
    # Second subplot: Noise Floor
    # -----------------------------

    ax1.plot(n.combined_const_noise[:samples_to_plot])
    ax1.set_title(f'Constant Noise Floor: {noise_db}db ({noise_level*100}%)')
    ax1.set_xlabel('Sample Index')
    ax1.set_ylabel('Amplitude')
    ax1.grid(True)

    # -----------------------------
    # Third subplot: Microphone Output
    # -----------------------------

    ax2.set_title(f'Microphone Output')
    for i,m in enumerate(mics):
        mic_output = m.getOutput(noise_level=noise_level)
        if i == 0:
            ax2.plot(mic_output[:samples_to_plot], label='Reference Microphone', color="red", alpha=1)
        else:
            ax2.plot(mic_output[:samples_to_plot], color="blue", alpha=0.5)

    ax2.set_xlabel('Sample Index')
    ax2.set_ylabel('Amplitude')
    ax2.legend()
    ax2.grid(True)

    b = n.beamforming()
    np.save(f"{np_res_path_beamformed}/{distance}.npy", b)

    # -----------------------------
    # Fourth subplot: Aligned Microphone Output
    # -----------------------------

    ax3.set_title(f'Aligned Microphone Output')
    for i,s in enumerate(n.aligned_outputs):
        if i == 0:
            ax3.plot(s[:samples_to_plot], label='Reference Microphone', color="red", alpha=1)
        else:
            ax3.plot(s[:samples_to_plot], color="blue", alpha=0.5)

    ax3.set_xlabel('Sample Index')
    ax3.set_ylabel('Amplitude')
    ax3.legend()
    ax3.grid(True)

    # -----------------------------
    # Fifth subplot: beamformed signal
    # -----------------------------
    max_amp_beamformed = np.max(b[:samples_to_plot]).item()
    max_orig = np.max(sources[0].produce_sound(0)[:samples_to_plot]).item()
    scaling = max_amp_beamformed/max_orig
    ax4.plot(sources[0].produce_sound(0)[:samples_to_plot] * scaling, label='Original Drone Signal')
    ax4.plot(b[:samples_to_plot], label='Beamformed')
    ax4.set_title('Beamformed Signal')
    ax4.set_xlabel('Sample Index')
    ax4.set_ylabel('Amplitude')
    ax4.legend()
    ax4.grid(True)

    plt.savefig(f"{img_res_path}/{distance}_results.png")
    plt.close()

    fig = plt.figure()
    ax3d = fig.add_subplot(111, projection='3d')

    n.plot_microphone_array(ax=ax3d)
    n.plot_sources(ax=ax3d)
    ax3d.set_title('Source and Microphone Array Positions')

    plt.savefig(f"{img_res_path}/{distance}_mic_arr.png")
    
    del n
    plt.close()