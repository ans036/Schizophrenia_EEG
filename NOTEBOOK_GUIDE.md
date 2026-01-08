# Schizophrenia EEG Classification - Notebook Guide

## Overview

This guide explains the structure and contents of the Schizophrenia EEG classification project notebooks and how to use them.

## Project Notebooks

### Main Project Notebook: schizophrenia-v1.ipynb

The primary notebook for this project contains your complete schizophrenia EEG classification pipeline.

**What it includes:**
- Data loading and preprocessing for EEG signals
- - Exploratory Data Analysis (EDA)
  - - Feature extraction and representation
    - - TimeDistributed LSTM model architecture implementation
      - - Model training with validation
        - - Performance evaluation and metrics
          - - Confusion matrix and classification reports visualization
            - - PSD (Power Spectral Density) analysis across frequency bands
             
              - ### Reference Notebooks
             
              - - **Input Data Representation.ipynb** - Demonstrates how EEG data is structured and visualized
                - - **Training.ipynb** - Shows training procedures and logging (reference implementation)
                 
                  - ## Running the Project
                 
                  - ### Prerequisites
                  - ```bash
                    pip install -r requirements.txt
                    ```

                    Required packages:
                    - TensorFlow/Keras
                    - - NumPy
                      - - Pandas
                        - - Matplotlib/Seaborn
                          - - Scikit-learn
                           
                            - ### Step-by-Step Execution
                           
                            - 1. **Data Preparation**
                              2.    - Load EEG data from your dataset
                                    -    - Preprocess signals (normalize, filter, etc.)
                                         -    - Split into train/validation/test sets
                                          
                                              - 2. **Model Building**
                                                3.    - Define TimeDistributed LSTM architecture
                                                      -    - Configure layers (TimeDistributed → LSTM → Dense → Dropout)
                                                           -    - Compile with appropriate loss and optimizer
                                                            
                                                                - 3. **Training**
                                                                  4.    - Run training for specified epochs (100 in our case)
                                                                        -    - Monitor training and validation metrics
                                                                             -    - Save best model checkpoints
                                                                              
                                                                                  - 4. **Evaluation**
                                                                                    5.    - Generate confusion matrix
                                                                                          -    - Calculate classification metrics (precision, recall, F1-score)
                                                                                               -    - Visualize PSD analysis by frequency bands
                                                                                                    -    - Generate performance plots
                                                                                                     
                                                                                                         - ## Model Architecture Details
                                                                                                     
                                                                                                         - ```
                                                                                                           Input: EEG signals in temporal format

                                                                                                           TimeDistributed Layers (141-147):
                                                                                                           ├─ Transform sequence data
                                                                                                           ├─ Output shape: (None, 8, 512)
                                                                                                           │
                                                                                                           LSTM Layer (lstm_20):
                                                                                                           ├─ Units: 128
                                                                                                           ├─ Captures temporal patterns
                                                                                                           ├─ Output shape: (None, 128)
                                                                                                           │
                                                                                                           Dropout (dropout_61):
                                                                                                           ├─ Regularization
                                                                                                           │
                                                                                                           Dense Layer (dense_46):
                                                                                                           ├─ Units: 512
                                                                                                           ├─ Feature extraction
                                                                                                           ├─ Output shape: (None, 512)
                                                                                                           │
                                                                                                           Dropout (dropout_62):
                                                                                                           ├─ Regularization
                                                                                                           │
                                                                                                           Dense Output Layer (dense_47):
                                                                                                           ├─ Units: 2 (Binary classification)
                                                                                                           ├─ Activation: Softmax
                                                                                                           └─ Output shape: (None, 2)
                                                                                                           ```
                                                                                                           
                                                                                                           ## Key Results
                                                                                                           
                                                                                                           - **Overall Accuracy**: 93%
                                                                                                           - - **Class 0 Recall**: 85% (Control cases)
                                                                                                             - - **Class 1 Recall**: 100% (Schizophrenia cases)
                                                                                                               - - **Training Epochs**: 100
                                                                                                                 - - **Final Training Accuracy**: 99.92%
                                                                                                                  
                                                                                                                   - ## Output Files Generated
                                                                                                                  
                                                                                                                   - - `graphics/confusion_matrix_schizophrenia.png` - Confusion matrix visualization
                                                                                                                     - - `graphics/model_architecture_timedistributed.png` - Model layer diagram
                                                                                                                       - - `graphics/psd_analysis_all_bands.png` - Power spectral density plots by frequency band
                                                                                                                        
                                                                                                                         - ## Troubleshooting
                                                                                                                        
                                                                                                                         - **Issue**: Data loading fails
                                                                                                                         - - **Solution**: Check that your dataset path is correct and files are in the expected format
                                                                                                                          
                                                                                                                           - **Issue**: Out of memory during training
                                                                                                                           - - **Solution**: Reduce batch size or use data generators for large datasets
                                                                                                                            
                                                                                                                             - **Issue**: Poor model performance
                                                                                                                             - - **Solution**: Review hyperparameters, check data preprocessing, ensure balanced class distribution
                                                                                                                              
                                                                                                                               - ## Future Enhancements
                                                                                                                              
                                                                                                                               - See README.md for planned improvements including:
                                                                                                                               - - Cross-subject validation
                                                                                                                                 - - Data augmentation techniques
                                                                                                                                   - - Hyperparameter optimization
                                                                                                                                     - - Ensemble methods
                                                                                                                                       - - Real-time processing capabilities
                                                                                                                                        
                                                                                                                                         - ## Support
                                                                                                                                        
                                                                                                                                         - For questions about the implementation, refer to the main README.md and inline notebook comments.
                                                                                                                                        
                                                                                                                                         - ## Citation
                                                                                                                                        
                                                                                                                                         - If you use this project in your research, please cite accordingly and acknowledge the use of EEG data for schizophrenia classification.
