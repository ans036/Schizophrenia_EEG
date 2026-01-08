import numpy as np
import tensorflow as tf

def add_gaussian_noise(data, noise_level=0.1):
    """
    Add Gaussian noise to EEG data for augmentation.
    
    Args:
        data: EEG signal array
        noise_level: Standard deviation of noise (default: 0.1)
    
    Returns:
        Augmented data with added noise
    """
    noise = np.random.normal(0, noise_level, data.shape)
    return data + noise

def time_shift(data, shift_max=50):
    """
    Randomly shift EEG signal in time.
    
    Args:
        data: EEG signal array (channels x samples)
        shift_max: Maximum shift in samples (default: 50)
    
    Returns:
        Time-shifted data
    """
    shift = np.random.randint(-shift_max, shift_max)
    return np.roll(data, shift, axis=-1)

def amplitude_scale(data, scale_range=(0.8, 1.2)):
    """
    Randomly scale the amplitude of EEG signals.
    
    Args:
        data: EEG signal array
        scale_range: Tuple of (min_scale, max_scale)
    
    Returns:
        Amplitude-scaled data
    """
    scale = np.random.uniform(scale_range[0], scale_range[1])
    return data * scale

def channel_dropout(data, dropout_prob=0.1):
    """
    Randomly drop out EEG channels for augmentation.
    
    Args:
        data: EEG signal array (channels x samples)
        dropout_prob: Probability of dropping each channel (default: 0.1)
    
    Returns:
        Data with randomly dropped channels
    """
    augmented = data.copy()
    n_channels = data.shape[0]
    dropout_mask = np.random.random(n_channels) > dropout_prob
    augmented[~dropout_mask] = 0
    return augmented

def augment_data(data, labels, augmentation_factor=2):
    """
    Apply multiple augmentation techniques to EEG data.
    
    Args:
        data: Array of EEG signals
        labels: Corresponding labels
        augmentation_factor: Number of augmented samples per original sample
    
    Returns:
        Augmented data and labels
    """
    augmented_data = [data]
    augmented_labels = [labels]
    
    for _ in range(augmentation_factor):
        aug_data = data.copy()
        
        # Randomly apply augmentations
        if np.random.random() > 0.5:
            aug_data = add_gaussian_noise(aug_data)
        if np.random.random() > 0.5:
            aug_data = np.array([time_shift(sample) for sample in aug_data])
        if np.random.random() > 0.5:
            aug_data = amplitude_scale(aug_data)
        if np.random.random() > 0.5:
            aug_data = np.array([channel_dropout(sample) for sample in aug_data])
        
        augmented_data.append(aug_data)
        augmented_labels.append(labels)
    
    return np.concatenate(augmented_data), np.concatenate(augmented_labels)
