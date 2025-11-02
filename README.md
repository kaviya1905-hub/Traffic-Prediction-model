## Traffic Flow Prediction with LSTM
-Predicting short-term traffic speeds on LA highways using deep learning (LSTM networks) and the widely used https://www.kaggle.com/datasets/annnnguyen/metr-la-dataset.

## Project Overview
-This repository contains an end-to-end pipeline for traffic speed forecasting:
  *Data Loading & Preprocessing: Automated handling of METR-LA data <br>
  *LSTM Model Training: Two-layer deep sequence model  <br>
  *Evaluation: MAE and RMSE on test set, visual sample predictions <br>
  *Outputs: Trained model saved, results and error analysis illustrated<br>
This project provides a robust, reproducible LSTM baseline for time-series traffic forecasting—ideal for research, learning, and extension.<br>

## Dataset
-Name: METR-LA (Metropolitan Los Angeles Traffic)
-Source: https://www.kaggle.com/datasets/annnnguyen/metr-la-dataset
-Description: 207 freeway sensors recording vehicle speed every 5 mins (~34,272 timepoints)
-Format: HDF5 file (METR-LA.h5)
-Local path example:
## D:\OneDrive\Documents\Traffic prediction model\METR-LA.h5

## Installation & Setup
1.Clone the repository:<br>
 git clone https://github.com/kaviya1905-hub/Traffic-Prediction-model.git<br>
3cd traffic-flow-prediction<br>
2.Download METR-LA dataset:<br>
 https://www.kaggle.com/datasets/annnnguyen/metr-la-dataset<br>
3.Install requirements:<br>
pip install torch h5py numpy pandas scikit-learn matplotlib<br>
4.Run the main script:<br>
 python traffic_prediction.py<br>
5.Repository Structure:<br>
-traffic_prediction.py  <br>     
-traffic_lstm_model.pth     <br>  
-METR-LA.h5<br>                   
-<output screenshots>.jpg<br>     
-README.md <br>                  

## Model & Preprocessing Details
 -Model: Stacked LSTM (2 layers, 64 hidden units)
      *Good for learning sequential/temporal patterns in traffic<br>
      *Trained using Mean Squared Error loss (MSE)<br>
 -Preprocessing:
      *Data loaded and scaled to with MinMaxScaler​<br>
      *Sliding window: last 12 timesteps predict next 12 (1 hour ahead)<br>
      *Univariate mode (first sensor)—multi-sensor possible with minor code edits<br>

## Training & Evaluation:
  -Training progress: Loss decreases smoothly <img width="1229" height="654" alt="Screenshot 2025-11-02 200447" src="https://github.com/user-attachments/assets/25bb885b-b100-4aaf-af2f-59c4b83bfda9" /> 

  -Test Set Performance:
MAE : 5.77<br>
RMSE: 12.60
<img width="699" height="651" alt="Screenshot 2025-11-02 200456" src="https://github.com/user-attachments/assets/e5ca3132-8f31-4529-ada1-a07f08f782a5" /> 

 -Saved Model:Model weights: "D:\OneDrive\Documents\Traffic Prediction Model\processed_data"
 -Sample Results and Visualizations:
      *Training/Validation curves<br>
      *Ground truth vs. predicted speeds for sample sensor(s)<br>
      *Error distribution<br>
<img width="1539" height="746" alt="Figure_1" src="https://github.com/user-attachments/assets/d4197041-dd0a-48e9-9509-6352072976f8" />

## Strengths:
  -Clean, reproducible code with interpretable outputs
  -Strong baseline for further research (TCN, GNN, multivariate, exogenous feature integration)

## Limitations:
-Currently univariate—accuracy improves with more sensors/features
-No explicit data augmentation (not common for time series)

## References:
Kaaggle - https://www.kaggle.com/datasets/annnnguyen/metr-la-dataset

## Author
Kaviya M
