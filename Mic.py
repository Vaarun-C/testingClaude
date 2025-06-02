from Exceptions.NullMicrophoneInput import NullMicrophoneInputException
from helper import process_signal

import numpy as np

class Microphone():
    def __init__(self, sensitivity_threshold: int, const_noise: np.ndarray = None):
        self.sensitivity:int = sensitivity_threshold
        self.input:np.ndarray = None
        self.__const_noise = const_noise
    
    def getOutput(self, noise_level: int = 1):
        if(self.input is None):
            raise NullMicrophoneInputException()
        
        if self.__const_noise is not None:
            signal = self.__apply_constant_noise(noise_level=noise_level)
        else:
            signal = self.input#self.__simulateSensitivity(self.input)
        processed_signal = process_signal(signal=signal)
        return processed_signal
    
    def __apply_constant_noise(self, noise_level: int = 1):
        """Apply constant background noise to the microphone input signal."""
        if self.input is None:
            raise NullMicrophoneInputException()
        
        noisy_signal = self.input + (self.__const_noise * noise_level)
        return noisy_signal
