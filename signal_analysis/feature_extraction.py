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

    def __init__(self, fs, f_res):
        """Init FT extractor
        
        Parameters
        -----------
        fs: 
            Sampling rate
        f_res: 
            frequency resolution 
        """
        self.fs = fs
        self.f_res = f_res

    def transform(self, sample, epoch_size=None):
        """Transform from time domain to frequency domain
        
        Parameters
        -----------
        sample: np.darray 
            Time domain data 
        epoch_size: int 
            Processing window in second. E.g: 2s epoch
        
        Returns
        --------
        fft_output:
            Power of frequency bins 
        freq:
            Frequency bins accordingly
        """
        if not epoch_size:
            epoch_size = int(len(sample / self.fs))

        n_fft = int(self.fs / self.f_res)
        fft_output = rfft(sample, n_fft)
        freq = rfftfreq(n_fft, 1/self.fs)

        self.fft_output = fft_output 
        self.freq = freq

        return abs(fft_output), freq 

    
    


        