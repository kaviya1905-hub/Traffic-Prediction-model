import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

DATA_PATH = r"D:\OneDrive\Documents\Traffic prediction model\METR-LA.csv\METR-LA.h5"
LOOKBACK = 12  
FORECAST_HORIZON = 12 
BATCH_SIZE = 64
NUM_EPOCHS = 20
LEARNING_RATE = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Loading data from .h5 file...")
with h5py.File(DATA_PATH, 'r') as f:
    print("Root keys:", list(f.keys()))
    
    if 'df' in f.keys():
        group = f['df']
        print("Keys in 'df' group:", list(group.keys()))
        
        if 'block0_values' in group:
            data = group['block0_values'][:]
        elif 'data' in group:
            data = group['data'][:]
        elif 'values' in group:
            data = group['values'][:]
        else:
            subkey = list(group.keys())[0]
            print(f"Using subkey: {subkey}")
            data = group[subkey][:]
    elif 'speed' in f.keys():
        data = f['speed'][:]
    else:
        key = list(f.keys())[0]
        print(f"Using key: {key}")
        data = f[key][:]

print(f"Data shape: {data.shape}")
if data.shape[0] < data.shape[1]:
    print("Transposing data to have timesteps as rows...")
    data = data.T
    print(f"New shape: {data.shape}")
df = pd.DataFrame(data)
print("First few rows:")
print(df.head())
print(f"Data statistics:\n{df.describe()}")

sensor_data = df.iloc[:, 0].values.reshape(-1, 1)
print(f"\nSelected sensor data shape: {sensor_data.shape}")
print(f"Data range: [{sensor_data.min():.2f}, {sensor_data.max():.2f}]")
scaler = MinMaxScaler()
sensor_data_scaled = scaler.fit_transform(sensor_data)
def create_sequences(data, lookback, forecast_horizon):
    X, y = [], []
    for i in range(len(data) - lookback - forecast_horizon + 1):
        X.append(data[i:i+lookback])
        y.append(data[i+lookback:i+lookback+forecast_horizon])
    return np.array(X), np.array(y)

X, y = create_sequences(sensor_data_scaled, LOOKBACK, FORECAST_HORIZON)
print(f"\nSequence shapes - X: {X.shape}, y: {y.shape}")
split_idx = int(0.8 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Convert to PyTorch tensors
X_train_t = torch.FloatTensor(X_train).to(DEVICE)
y_train_t = torch.FloatTensor(y_train).to(DEVICE)
X_test_t = torch.FloatTensor(X_test).to(DEVICE)
y_test_t = torch.FloatTensor(y_test).to(DEVICE)

train_dataset = TensorDataset(X_train_t, y_train_t)
test_dataset = TensorDataset(X_test_t, y_test_t)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

print(f"Training samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
class TrafficLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=12):
        super(TrafficLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out.unsqueeze(-1)

model = TrafficLSTM(input_size=1, hidden_size=64, num_layers=2, output_size=FORECAST_HORIZON).to(DEVICE)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

print(f"\nModel created on device: {DEVICE}")
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print("\n" + "="*60)
print("TRAINING STARTED")
print("="*60)
train_losses = []
for epoch in range(NUM_EPOCHS):
    model.train()
    epoch_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    avg_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_loss)
    print(f"Epoch {epoch+1:02d}/{NUM_EPOCHS}, Loss: {avg_loss:.6f}")
print("\n" + "="*60)
print("EVALUATION")
print("="*60)
model.eval()
predictions = []
actuals = []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        y_pred = model(X_batch)
        predictions.append(y_pred.cpu().numpy())
        actuals.append(y_batch.cpu().numpy())

predictions = np.concatenate(predictions, axis=0)
actuals = np.concatenate(actuals, axis=0)
predictions = scaler.inverse_transform(predictions.reshape(-1, 1)).reshape(predictions.shape)
actuals = scaler.inverse_transform(actuals.reshape(-1, 1)).reshape(actuals.shape)
mae = mean_absolute_error(actuals.flatten(), predictions.flatten())
rmse = np.sqrt(mean_squared_error(actuals.flatten(), predictions.flatten()))

print(f"\n{'='*60}")
print(f"FINAL RESULTS")
print(f"{'='*60}")
print(f"Test MAE:  {mae:.4f}")
print(f"Test RMSE: {rmse:.4f}")
print(f"{'='*60}\n")
model_save_path = r"D:\OneDrive\Documents\Traffic prediction model\traffic_lstm_model.pth"
torch.save(model.state_dict(), model_save_path)
print(f"✅ Model saved: {model_save_path}\n")
plt.figure(figsize=(10, 5))
plt.plot(range(1, NUM_EPOCHS+1), train_losses, marker='o', linewidth=2)
plt.title('Training Loss Over Epochs', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss (MSE)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
sample_idx = min(50, len(actuals)-1)
plt.figure(figsize=(12, 5))
time_steps = range(FORECAST_HORIZON)
plt.plot(time_steps, actuals[sample_idx], label='Actual', marker='o', linewidth=2, markersize=8)
plt.plot(time_steps, predictions[sample_idx], label='Predicted', marker='x', linewidth=2, markersize=8)
plt.title(f'Traffic Prediction Sample #{sample_idx}', fontsize=14, fontweight='bold')
plt.xlabel('Time Step (5-min intervals)', fontsize=12)
plt.ylabel('Traffic Speed', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
sample_indices = [0, min(100, len(actuals)-1), min(200, len(actuals)-1), min(300, len(actuals)-1)]
for i, ax in enumerate(axes.flat):
    idx = sample_indices[i]
    ax.plot(actuals[idx], label='Actual', marker='o', linewidth=2)
    ax.plot(predictions[idx], label='Predicted', marker='x', linewidth=2)
    ax.set_title(f'Sample #{idx}', fontweight='bold')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Speed')
    ax.legend()
    ax.grid(True, alpha=0.3)
plt.suptitle('Traffic Predictions: Multiple Samples', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()
plt.figure(figsize=(8, 8))
plt.scatter(actuals.flatten(), predictions.flatten(), alpha=0.3, s=2, c='blue')
plt.plot([actuals.min(), actuals.max()], [actuals.min(), actuals.max()], 'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual Speed', fontsize=12)
plt.ylabel('Predicted Speed', fontsize=12)
plt.title('Predicted vs Actual Traffic Speed', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("\n Model training and evaluation complete!")
print(f"Final Results: MAE={mae:.4f}, RMSE={rmse:.4f}")
