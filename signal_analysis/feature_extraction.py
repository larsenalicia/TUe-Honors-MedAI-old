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

    
    


        import pandas as pd 
import numpy as np
from abc import ABC, abstractmethod 
from scipy.fft import rfft, rfftfreq
from mne.time_frequency import psd_array_multitaper
from typing import Optional
from sklearn.decomposition import PCA

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

    def transform(self, sample: np.ndarray, epoch_size=None):
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


class MultitaperSpectralTransform(FeatureExtraction):
    """ Multitaper Spectral Analysis"""

    def __init__(self, fs=400, fmin=0, fmax=50, epoch_size=2, f_res=0.5):
        self.fs = fs 
        self.fmin = fmin 
        self.fmax = fmax 
        self.epoch_size = epoch_size 
        self.f_res = f_res

    def transform(self, sample: np.ndarray, epoch_size=None):
        """Transofrm from time domain to frequency domain 
        using Multitaper spectral method 

        Parameters
        ----------

        Returns
        -------
        psds: raw PSD estimate for each epoch. For example, if we process with 36 epochs 
            then `psds.shape = (36, 101)`. 101 are real values for each freqs bin
        freqs: frequency bins

        Notes
        -------
        The input is the whole signal not just 1 epoch. The multitaper method assigned
        a different taper to different epochs similar to window function.
        """
        if epoch_size:
            self.epoch_size = epoch_size
        n = len(sample)
        n_per_epoch = self.epoch_size * self.fs
        n_epoch = (n // n_per_epoch)
        X = sample[:n_epoch * n_per_epoch].reshape(n // n_per_epoch, n_per_epoch)

        psds, freqs = psd_array_multitaper(X, self.fs, fmin=self.fmin, 
                                           fmax=self.fmax, n_jobs=1, output="power")

        self.psds = psds 
        self.freqs = freqs 

        return psds, freqs


class BandwisePowerTransform(FeatureExtraction):
    """Extract bandwise power in canonical spectral ranges"""
    
    def transform(self, psds: np.ndarray | MultitaperSpectralTransform | FourierTransform, 
                freqs: Optional[np.ndarray] = None):
        if isinstance(psds, np.ndarray):
            return self.raw_transform(psds, freqs)
            print("sth")
        
        elif isinstance(psds, MultitaperSpectralTransform):
            ms_transformer = psds 
            psds = ms_transformer.psds
            freqs = ms_transformer.freqs 
            return self.raw_transform(psds, freqs)
        
        elif isinstance(psds, FourierTransform):
            ft_output = psds.fft_output
            ft_output = ft_output.reshape(1, ft_output.shape[0]) # Reshape to 2-D since FT returns 1-D
            freqs = psds.freq
            return self.raw_transform(ft_output, freqs)
            
    @staticmethod
    def raw_transform(psds: np.ndarray, freqs: np.ndarray) -> np.ndarray:
        """Compute avg estimated power in selected frequency band. There are in 
        order slow, delta, theta, beta, and gamma wave.
        """
        slow = BandwisePowerTransform.extract_bandwise_power(psds, freqs, 0, 1)
        delta = BandwisePowerTransform.extract_bandwise_power(psds, freqs, 4, 8)
        theta = BandwisePowerTransform.extract_bandwise_power(psds, freqs, 8, 13)
        beta = BandwisePowerTransform.extract_bandwise_power(psds, freqs, 13, 25)
        gamma = BandwisePowerTransform.extract_bandwise_power(psds, freqs, 25, 50)

        return np.array([slow, delta, theta, 
                        beta, gamma])
    
    @staticmethod
    def extract_bandwise_power(psds: np.ndarray, freqs: np.ndarray, 
                            fmin: int | float, fmax: int | float) -> np.ndarray:
        """
        Note:
        ---------
        Take in an array of psd and its frequency bins. Compute filtering on the selected
        band frequency. Take the average of estimated power.
        """
        
        filter = (freqs <= fmax) & (freqs >= fmin)
        if sum(filter) == 0: print("Not available bandwise freqs") # Add warning if missing freq band
        return np.average(psds[:, filter], axis=1) 


class PCATransform(FeatureExtraction):
    """Extract n-th first principle components from psds. So from 100 dims to n dims"""

    def __init__(self, n_components) -> None:
        self.n_components = n_components

    def transform(self, psds_transform: MultitaperSpectralTransform | np.ndarray):
        """Using PCA method in sklearn to extract nth first princicple components"""
        
        if isinstance(psds_transform, MultitaperSpectralTransform):
            pca = PCA(n_components=self.n_components)
            return pca.fit_transform(psds_transform.psds)
        
        if isinstance(psds_transform, np.ndarray):
            NotImplemented

    