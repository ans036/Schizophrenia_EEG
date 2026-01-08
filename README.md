**Status:** Active - Schizophrenia EEG Classification Project

# Schizophrenia Classification Using EEG Brain Signals

A novel deep learning approach to classify Schizophrenia using EEG (electroencephalography) brain signals with LSTM and Dense neural networks. This project demonstrates the use of TimeDistributed layers with temporal feature extraction from EEG data for binary classification (Schizophrenia vs. Control).

## Project Overview

This project implements a sequential neural network model using TimeDistributed LSTM layers combined with Dense layers to classify EEG signals into two categories:
- **Class 0**: Control (Non-Schizophrenia)
- - **Class 1**: Schizophrenia
 
  - ## Model Architecture
 
  - The model consists of the following layers:
  - - **TimeDistributed Layers (141-147)**: Transform input EEG signals with output shapes progressing from (None, 8, 9, 8, 64) to (None, 8, 512)
    - - **LSTM Layer (lstm_20)**: Extracts temporal patterns from EEG sequences with 128 units, outputting (None, 128)
      - - **Dropout Layer (dropout_61)**: Regularization with dropout rate to prevent overfitting
        - - **Dense Layer (dense_46)**: Hidden layer with 512 units for feature extraction
          - - **Dropout Layer (dropout_62)**: Additional regularization
            - - **Dense Output Layer (dense_47)**: Binary classification output (None, 2)
             
              - ## Results
             
              - ### Model Performance
              - - **Overall Accuracy**: 93%
                - - **Training Epochs**: 100
                  - - **Final Training Accuracy**: 0.9992
                    - - **Final Validation Accuracy**: 0.9998
                     
                      - ### Classification Metrics
                     
                      - ```
                        Classification Report:
                                        precision    recall  f1-score   support
                               Class 0      1.00      0.85      0.92        53
                               Class 1      0.88      1.00      0.94        59

                               accuracy                         0.93       112
                            macro avg      0.94      0.92      0.93       112
                         weighted avg      0.94      0.93      0.93       112
                        ```

                        ### Confusion Matrix Results
                        - **True Negatives**: 45 (Control correctly classified)
                        - - **False Positives**: 8 (Control misclassified as Schizophrenia)
                          - - **False Negatives**: 0 (No Schizophrenia cases missed)
                            - - **True Positives**: 59 (Schizophrenia correctly classified)
                             
                              - ## Data Analysis
                             
                              - ### Power Spectral Density (PSD) Analysis
                              - The model analyzes PSD across multiple frequency bands:
                              - - **Band 0** (Delta/Low frequencies): Clear separation between control and schizophrenia groups
                                - - **Band 1** (Theta): Distinct distribution differences
                                  - - **Band 2** (Alpha): Visible class discrimination
                                    - - **Band 3** (Beta): Observable patterns distinguishing the two classes
                                     
                                      - PSD plots show significant differences in power distribution between Class 0 (Control - blue) and Class 1 (Schizophrenia - orange) across all frequency bands.
                                     
                                      - ## Dataset
                                     
                                      - The project uses an EEG dataset for schizophrenia classification with:
                                      - - Multi-channel EEG recordings
                                        - - Binary labels (Schizophrenia vs. Control)
                                          - - Preprocessed data in standard formats
                                            - - Data split for training, validation, and testing
                                             
                                              - ## Key Features
                                             
                                              - 1. **Temporal Processing**: TimeDistributed layers capture temporal dynamics in EEG signals
                                                2. 2. **LSTM Architecture**: Effective handling of sequential data with memory mechanisms
                                                   3. 3. **Regularization**: Dropout layers prevent overfitting and improve generalization
                                                      4. 4. **High Performance**: Achieves 93% overall accuracy with balanced precision and recall
                                                         5. 5. **Class Balance Awareness**: Equal sensitivity to both classes (100% recall for schizophrenia)
                                                           
                                                            6. ## Files and Structure
                                                           
                                                            7. - `schizophrenia_classification.ipynb` - Main notebook with complete analysis and model training
                                                               - - `graphics/` - Visualization outputs including confusion matrix and PSD plots
                                                                 - - `requirements.txt` - Python dependencies
                                                                   - - `README.md` - This file
                                                                    
                                                                     - ## Usage
                                                                    
                                                                     - ### Training the Model
                                                                     - ```python
                                                                       # Load the notebook and run all cells to:
                                                                       # 1. Prepare EEG data
                                                                       # 2. Build the TimeDistributed LSTM model
                                                                       # 3. Train on the schizophrenia dataset
                                                                       # 4. Evaluate performance metrics
                                                                       # 5. Generate visualizations
                                                                       ```

                                                                       ## Dependencies

                                                                       Required packages:
                                                                       - TensorFlow/Keras
                                                                       - - NumPy
                                                                         - - Pandas
                                                                           - - Matplotlib/Seaborn (for visualizations)
                                                                             - - Scikit-learn (for metrics)
                                                                              
                                                                               - See `requirements.txt` for complete dependencies.
                                                                              
                                                                               - ## Future Improvements
                                                                              
                                                                               - 1. **Cross-subject Validation**: Implement subject-independent classification
                                                                                 2. 2. **Data Augmentation**: Enhance training data with augmentation techniques
                                                                                    3. 3. **Hyperparameter Tuning**: Optimize model parameters for improved accuracy
                                                                                       4. 4. **Feature Analysis**: Deep investigation of important EEG features
                                                                                          5. 5. **Ensemble Methods**: Combine multiple models for enhanced predictions
                                                                                             6. 6. **Real-time Processing**: Adapt model for real-time EEG monitoring
                                                                                               
                                                                                                7. ## Author
                                                                                               
                                                                                                8. Created as a novel approach to schizophrenia classification using deep learning on EEG signals.
                                                                                               
                                                                                                9. ## License
                                                                                               
                                                                                                10. MIT License - See LICENSE file for details
