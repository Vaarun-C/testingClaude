import wave
import numpy as np

import os

def analyze_wav(filename):
    try:
        # Open the WAV file
        with wave.open(filename, 'rb') as wav_file:
            # Get basic properties
            n_channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            frame_rate = wav_file.getframerate()
            n_frames = wav_file.getnframes()
            comp_type = wav_file.getcomptype()
            comp_name = wav_file.getcompname()
            
            # Calculate duration
            duration = n_frames / float(frame_rate)
            
            # Read the audio data
            audio_data = wav_file.readframes(n_frames)
            audio_np = np.frombuffer(audio_data, dtype=np.int16)
            
            # Calculate some basic statistics
            max_amplitude = np.max(np.abs(audio_np))
            rms = np.sqrt(np.mean(np.square(audio_np)))

            # raw_data = wav_file.readframes(n_frames)

            # print(raw_data)
        
            # Convert to numpy array
            if sample_width == 1:
                dtype = np.uint8  # 8-bit unsigned
            elif sample_width == 2:
                dtype = np.int16  # 16-bit signed
            elif sample_width == 4:
                dtype = np.int32  # 32-bit signed
                
            # audio_np = np.frombuffer(raw_data, dtype=dtype)
            
            # Print the analysis
            print("\nWAV File Analysis")
            print("-----------------")
            print(f"Filename: {filename}")
            print(f"Number of channels: {n_channels}")
            print(f"Sample width: {sample_width} bytes")
            print(f"Frame rate: {frame_rate} Hz")
            print(f"Number of frames: {n_frames}")
            print(f"Duration: {duration:.2f} seconds")
            print(f"Compression type: {comp_type}")
            print(f"Compression name: {comp_name}")
            print("\nAudio Statistics")
            print("----------------")
            print(f"Maximum amplitude: {max_amplitude}")
            print(f"RMS value: {rms:.2f}")
            print(f"Bit depth: {sample_width * 8} bits")
            
            if n_channels == 2:
                # If stereo, separate channels and show stats for each
                left_channel = audio_np[::2]
                right_channel = audio_np[1::2]
                print("\nChannel Statistics")
                print("------------------")
                print(f"Left Channel - Max amplitude: {np.max(np.abs(left_channel))}")
                print(f"Right Channel - Max amplitude: {np.max(np.abs(right_channel))}")

            return {
                'n_channels': n_channels,
                'sample_width': sample_width,
                'frame_rate': frame_rate,
                'n_frames': n_frames,
                'duration': duration,
                'max_amplitude': max_amplitude,
                'rms': rms,
                'audio_data': audio_np
            }
            
    except wave.Error as e:
        print(f"Error reading WAV file: {str(e)}")
    except FileNotFoundError:
        print(f"File not found: {filename}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

def save_audio_segments(audio_data, sample_rate, output_dir, prefix='segment'):    
    os.makedirs(output_dir, exist_ok=True)
    samples_per_second = int(sample_rate)
    total_seconds = len(audio_data) // samples_per_second

    # Split and save each second
    for i in range(total_seconds):
        # Extract one second of data
        start_idx = i * samples_per_second
        end_idx = start_idx + samples_per_second
        segment = audio_data[start_idx:end_idx]
        
        # Save to file
        filename = f"{prefix}_{i}.npy"
        filepath = os.path.join(output_dir, filename)
        np.save(filepath, segment)
        
    # Handle remaining samples if any
    remaining_samples = len(audio_data) % samples_per_second
    if remaining_samples > 0:
        # Save remaining samples as the last segment
        last_segment = audio_data[-remaining_samples:]
        filename = f"{prefix}_{total_seconds}.npy"
        filepath = os.path.join(output_dir, filename)
        np.save(filepath, last_segment)
    
    return total_seconds + (1 if remaining_samples > 0 else 0)