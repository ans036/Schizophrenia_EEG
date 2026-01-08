"""data_loading.py - Module for loading EEG data from .eea and .edf files"""

import os
import numpy as np
import pandas as pd
import pyedflib


def read_eea_file(eea_file, num_channels=16, samples_per_channel=7680):
    """
    Read an .eea file and return EEG data as numpy array.
    
    Args:
        eea_file (str): Path to .eea file
        num_channels (int): Number of EEG channels
        samples_per_channel (int): Number of samples per channel
    
    Returns:
        numpy.ndarray: EEG data of shape (num_channels, samples_per_channel)
    """
    with open(eea_file, 'r') as f:
        data = f.read().split('\n')
    data = [float(item) for item in data if item]
    return np.reshape(data, (num_channels, samples_per_channel))


def read_edf(edf_file, channels_to_drop=['Fp1', 'Fz', 'Fp2']):
    """
    Read an .edf file using pyedflib and return as pandas DataFrame.
    
    Args:
        edf_file (str): Path to .edf file
        channels_to_drop (list): List of channel names to exclude
    
    Returns:
        pandas.DataFrame: EEG data with channels as columns
    """
    f = pyedflib.EdfReader(edf_file)
    n = f.signals_in_file
    signal_labels = f.getSignalLabels()
    sigbufs = np.zeros((n, f.getNSamples()[0]))
    
    for i in range(n):
        sigbufs[i, :] = f.readSignal(i)
    
    df = pd.DataFrame(sigbufs.transpose(), columns=signal_labels)
    
    # Drop specified channels
    for ch in channels_to_drop:
        if ch in df.columns:
            df = df.drop(columns=[ch])
    
    return df


def load_eea_dataset(directory, channels, subdirs_labels=[('SCZ-Dataset2-Normal', 0), ('SCZ-Dataset2-SCZ', 1)]):
    """
    Load all .eea files from subdirectories.
    
    Args:
        directory (str): Root directory containing subdirectories
        channels (list): List of channel names
        subdirs_labels (list): List of (subdirectory_name, label) tuples
    
    Returns:
        tuple: (data_arrays, labels)
            data_arrays: list of numpy arrays
            labels: list of corresponding labels
    """
    all_data = []
    labels = []
    
    for subdir, label in subdirs_labels:
        subdir_path = os.path.join(directory, subdir)
        for filename in os.listdir(subdir_path):
            if filename.endswith(".eea"):
                file_path = os.path.join(subdir_path, filename)
                data_array = read_eea_file(file_path)
                all_data.append(data_array)
                labels.append(label)
    
    return all_data, np.array(labels)


def load_edf_dataset(directory, channels):
    """
    Load all .edf files from a directory.
    Labels based on filename prefix: 'h' for class 0, 's' for class 1.
    
    Args:
        directory (str): Directory containing .edf files
        channels (list): List of channel names to extract
    
    Returns:
        tuple: (data_arrays, labels)
            data_arrays: list of pandas DataFrames
            labels: list of corresponding labels
    """
    all_data = []
    labels = []
    
    for filename in os.listdir(directory):
        if filename.endswith(".edf"):
            # Determine label from filename
            if filename.startswith("h"):
                label = 0
            elif filename.startswith("s"):
                label = 1
            else:
                continue
            
            df = read_edf(os.path.join(directory, filename))
            all_data.append(df[channels].values)  # shape: (num_samples, num_channels)
            labels.append(label)
    
    return all_data, np.array(labels)
