# Traffic Flow Prediction with LSTM
-This repository contains code and results for short-term traffic speed forecasting using LSTM networks on the METR-LA dataset.

# Code
- Main script: `traffic\_prediction.py`
- Requires: Python, PyTorch, h5py, numpy, pandas, scikit-learn, matplotlib

# Setup Instructions
1. Clone/download this repository.
2. Place the METR-LA dataset (`METR-LA.h5`) in:
**D:\\OneDrive\\Documents\\Traffic prediction model\\METR-LA.csv\\**
3. Install requirements:
&nbsp;```

 **pip install torch h5py numpy pandas scikit-learn matplotlib**

&nbsp;```
4. Run:

&nbsp;```

&nbsp;**python traffic\_prediction.py**

&nbsp;```

# Dataset Details
Source: METR-LA public dataset – traffic speed readings from 207 LA highway sensors, 34,272 timesteps.
Format: HDF5 file (`METR-LA.h5`), loaded as a DataFrame.
Usage:  Example uses the first sensor for univariate LSTM modeling (edit for multivariate support).

# Trained Model File
- Final trained model is saved as:  

**`traffic\_lstm\_model.pth`**

# Results
-Uploaded as screenshot

# Training and Data Inspection
- Data loaded with shape `(34272, 207)`
- First few rows and statistics printed for confirmation

# Training Progress
- 20 epochs, loss decreases smoothly, confirming learning:

**Epoch 01/20, Loss: 0.061735**

**...**

**Epoch 20/20, Loss: 0.029305**

# Evaluation
-Test MAE: 5.7701
-Test RMSE: 12.5997

# Predictions
- Predictions tracked true values, as shown below (first sample, scatter):
-[Sample Predictions](Screenshot-2025-10-30-103712.jpg)
-[Scatter Plot (Predicted vs Actual)](Screenshot-2025-10-30-103722.jpg)

# Model & Preprocessing Explanation
Model architecture: 
-A two-layer LSTM with 64 hidden units was chosen due to its effectiveness at modeling time-series data with sequential dependencies, such as traffic. LSTMs handle long-range temporal relationships and are widely used in traffic forecasting research.

Preprocessing:  
-Data loaded and converted to a scaled \[0,1] range for stable training (using MinMaxScaler).
- Time-series windowing: For each prediction, the last 12 five-minute readings predict the next 12 readings (1 hour ahead).
- (No explicit augmentation, as time-series usually do not use classic image/data augmentations.)

Interpretation of Performance:  
- The training loss falls and plateaus quickly, indicating good convergence.
- MAE and RMSE are reasonable for basic univariate prediction.
- Sample prediction plots show the LSTM can track the general trend, but greater accuracy may be reached by using more complex models, more sensors, or additional features.

Strengths:
-Simple pipeline, reproducible results, interpretable error plots, good baseline for upgrades.

Weaknesses:
-Univariate; real-world scenarios need more features, richer architectures (TCN, GNN) for noisy or highly variable data.

# Author
Kaviya M

# Output:
<img width="862" height="713" alt="Screenshot 2025-10-30 103722" src="https://github.com/user-attachments/assets/b581a5ff-a5c3-4da1-8c8d-f83a3b7e8c0f" />
<img width="925" height="741" alt="Screenshot 2025-10-30 103712" src="https://github.com/user-attachments/assets/049aedfd-c1d2-4266-a6a2-0aca553e8bc8" />
<img width="1245" height="702" alt="Screenshot 2025-10-30 103655" src="https://github.com/user-attachments/assets/344ac69e-bc52-4b91-82f9-3b6f9457c5f2" />


