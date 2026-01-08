import numpy as np
from scipy import signal
from scipy.stats import skew, kurtosis

def compute_psd(data, fs=256, nperseg=256):
    """
    Compute Power Spectral Density using Welch's method.
    
    Args:
        data: EEG signal array (channels x samples)
        fs: Sampling frequency (default: 256 Hz)
        nperseg: Length of each segment (default: 256)
    
    Returns:
        freqs: Array of sample frequencies
        psd: Power spectral density array
    """
    freqs, psd = signal.welch(data, fs=fs, nperseg=nperseg, axis=-1)
    return freqs, psd

def extract_band_powers(psd, freqs):
    """
    Extract power in different frequency bands.
    
    Args:
        psd: Power spectral density array
        freqs: Frequency array
    
    Returns:
        dict: Dictionary containing power in each band
    """
    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 50)
    }
    
    band_powers = {}
    for band_name, (low, high) in bands.items():
        idx = np.logical_and(freqs >= low, freqs <= high)
        band_powers[band_name] = np.mean(psd[:, idx], axis=1)
    
    return band_powers

def extract_statistical_features(data):
    """
    Extract statistical features from EEG signals.
    
    Args:
        data: EEG signal array (channels x samples)
    
    Returns:
        dict: Dictionary containing statistical features
    """
    features = {
        'mean': np.mean(data, axis=1),
        'std': np.std(data, axis=1),
        'variance': np.var(data, axis=1),
        'skewness': skew(data, axis=1),
        'kurtosis': kurtosis(data, axis=1),
        'min': np.min(data, axis=1),
        'max': np.max(data, axis=1)
    }
    return features

def extract_all_features(data, fs=256):
    """
    Extract all features from EEG data.
    
    Args:
        data: EEG signal array (channels x samples)
        fs: Sampling frequency (default: 256 Hz)
    
    Returns:
        np.array: Concatenated feature vector
    """
    # Compute PSD
    freqs, psd = compute_psd(data, fs=fs)
    
    # Extract band powers
    band_powers = extract_band_powers(psd, freqs)
    
    # Extract statistical features
    stat_features = extract_statistical_features(data)
    
    # Concatenate all features
    all_features = []
    for band_power in band_powers.values():
        all_features.append(band_power)
    for stat_feature in stat_features.values():
        all_features.append(stat_feature)
    
    return np.concatenate(all_features)
