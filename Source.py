from typeguard import typechecked
import numpy as np
import librosa

@typechecked
class Source:
    def __init__(self, sound_source: str | np.ndarray, target_sr: int = 33203):
        self.sr = target_sr

        if type(sound_source) is str:
            if sound_source.endswith('.npy'):
                self.signal = np.load(sound_source) 

                if len(self.signal.shape) == 2:
                    self.signal = self.signal.reshape(self.signal.shape[0]*self.signal.shape[1])
                elif len(self.signal.shape) == 1 and self.signal.shape[0] % target_sr != 0:
                    raise Exception('Given numpy array not of target sr')
                else:
                    raise Exception(f'Unknown shape type: {self.signal.shape}')

            elif sound_source.endswith('.wav'):
                self.signal, _ = librosa.load(sound_source, sr=target_sr, mono=True)
        else:
            self.signal = sound_source

        self.initial_db = self.__get_spl_db()
        
    def __process_audio_distance(self, distance: float):
        # Calculate attenuated dB level at the given distance
        db_at_distance = self.initial_db - 20 * np.log10(distance)
    
        # Calculate amplitude scaling factor
        original_amplitude = 10 ** (self.initial_db / 20.0)
        new_amplitude = 10 ** (db_at_distance / 20.0)
        scaling_factor = new_amplitude / original_amplitude
    
        # Apply scaling to signal
        return self.signal * scaling_factor 

    def __get_spl_db(self):
        sig = self.signal - np.mean(self.signal)
        rms = np.sqrt(np.mean(sig**2))
        db_spl = 20.0 * np.log10(rms / 20e-6)
        return db_spl

    def produce_sound(self, distance: float):
        if distance <= 1.0:
            return self.signal
        else:
            return self.__process_audio_distance(distance)