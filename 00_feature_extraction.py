import pandas as pd 
import numpy as np
from abc import ABC, abstractmethod 
from scipy.fft import rfft, rfftfreq

class FeatureExtraction(ABC):
    """Store Feature Extraction methods"""

    @abstractmethod
    def transform(self):
        NotImplemented


class FourierTransform(FeatureExtraction):
    """Fourier Transform Extractor"""

    def __init__(self, sampling_rate, frequency_res):
        self.fs = sampling_rate
        self.f_res = frequency_res

    def transform(self, sample, epoch_size):
        """Transform from time domain to frequency domain"""
        n_fft = int(epoch_size * self.fs / self.f_res)
        fft_output = rfft(sample, n_fft)
        freq = rfftfreq(n_fft, 1/self.fs)

        return fft_output, freq 
    
    


        