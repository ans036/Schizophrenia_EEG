import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import pandas as pd

def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance on test data.
    
    Args:
        model: Trained model
        X_test: Test data
        y_test: Test labels
    
    Returns:
        Dictionary containing evaluation metrics
    """
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    accuracy = accuracy_score(y_test, y_pred_classes)
    
    return {
        'accuracy': accuracy,
        'y_pred': y_pred_classes,
        'y_true': y_test
    }

def plot_confusion_matrix(y_true, y_pred, class_names=['Healthy', 'Schizophrenia'],
                         save_path='results/confusion_matrix.png'):
    """
    Plot and save confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def generate_classification_report(y_true, y_pred, class_names=['Healthy', 'Schizophrenia'],
                                  save_path='results/classification_report.txt'):
    """
    Generate and save classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save the report
    """
    report = classification_report(y_true, y_pred, target_names=class_names)
    
    with open(save_path, 'w') as f:
        f.write(report)
    
    return report

def plot_feature_heatmap(features, labels, feature_names=None,
                        save_path='results/feature_heatmap.png'):
    """
    Plot heatmap of features for different classes.
    
    Args:
        features: Feature array (samples x features)
        labels: Label array
        feature_names: List of feature names
        save_path: Path to save the plot
    """
    # Calculate mean features for each class
    unique_labels = np.unique(labels)
    mean_features = []
    
    for label in unique_labels:
        mask = labels == label
        mean_features.append(np.mean(features[mask], axis=0))
    
    mean_features = np.array(mean_features)
    
    plt.figure(figsize=(12, 6))
    sns.heatmap(mean_features, cmap='coolwarm', center=0,
                yticklabels=['Healthy', 'Schizophrenia'],
                xticklabels=feature_names if feature_names else range(features.shape[1]))
    plt.title('Mean Feature Values by Class')
    plt.xlabel('Features')
    plt.ylabel('Class')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_psd_distribution(psd_data, labels, frequency_bands,
                         save_path='results/PSD_distribution.png'):
    """
    Plot PSD distribution across frequency bands for different classes.
    
    Args:
        psd_data: PSD array (samples x frequency_bins)
        labels: Label array
        frequency_bands: Dictionary of frequency bands
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, (band_name, (low, high)) in enumerate(frequency_bands.items()):
        ax = axes[idx]
        
        for label in [0, 1]:
            mask = labels == label
            label_name = 'Healthy' if label == 0 else 'Schizophrenia'
            ax.hist(psd_data[mask, idx], bins=30, alpha=0.5, label=label_name)
        
        ax.set_title(f'{band_name.capitalize()} Band ({low}-{high} Hz)')
        ax.set_xlabel('Power')
        ax.set_ylabel('Frequency')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
