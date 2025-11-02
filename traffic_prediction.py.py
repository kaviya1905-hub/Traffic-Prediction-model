import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle
import json
import argparse
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
np.random.seed(42)
torch.manual_seed(42)

class TrafficDataPreprocessor:
    
    def __init__(self, data_path, output_dir='processed_data'):
        self.data_path = data_path
        self.output_dir = output_dir
        self.scaler = StandardScaler()
        os.makedirs(output_dir, exist_ok=True)
    
    def load_data(self):
        print(f"Loading data from: {self.data_path}")
        
        data_files = []
        if os.path.isdir(self.data_path):
            for file in os.listdir(self.data_path):
                if file.endswith(('.csv', '.h5', '.pkl', '.npy')):
                    data_files.append(os.path.join(self.data_path, file))
        elif os.path.isfile(self.data_path):
            data_files = [self.data_path]
        
        if not data_files:
            print("No data files found. Generating synthetic data...")
            return self.generate_synthetic_data()
        
        for file_path in data_files:
            try:
                if file_path.endswith('.csv'):
                    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                elif file_path.endswith('.h5'):
                    import h5py
                    with h5py.File(file_path, 'r') as f:
                        keys = list(f.keys())
                        print(f"Available keys in H5 file: {keys}")
                        
                        data_key = None
                        for key in ['speed', 'flow', 'data', 'traffic']:
                            if key in keys:
                                data_key = key
                                break
                        
                        if data_key is None:
                            data_key = keys[0]  # Use first available key
                        
                        data = f[data_key][:]
                        print(f"Loaded data from key '{data_key}' with shape: {data.shape}")
                        
                        start_date = datetime(2012, 3, 1)
                        timestamps = [start_date + timedelta(minutes=5*i) for i in range(len(data))]
                        df = pd.DataFrame(
                            data,
                            index=timestamps,
                            columns=[f'sensor_{i}' for i in range(data.shape[1])]
                        )
                elif file_path.endswith('.pkl'):
                    df = pd.read_pickle(file_path)
                elif file_path.endswith('.npy'):
                    data = np.load(file_path)
                    start_date = datetime(2012, 3, 1)
                    timestamps = [start_date + timedelta(minutes=5*i) for i in range(len(data))]
                    df = pd.DataFrame(
                        data,
                        index=timestamps,
                        columns=[f'sensor_{i}' for i in range(data.shape[1])]
                    )
                
                print(f"Successfully loaded data with shape: {df.shape}")
                return df
                
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
        
        print("Failed to load any data files. Generating synthetic data...")
        return self.generate_synthetic_data()
    
    def generate_synthetic_data(self, num_sensors=50, num_days=90, freq='5T'):
        print(f"Generating synthetic traffic data: {num_sensors} sensors, {num_days} days")
        
        start_date = datetime(2024, 1, 1)
        end_date = start_date + timedelta(days=num_days)
        timestamps = pd.date_range(start_date, end_date, freq=freq)
        
        data = []
        for sensor_id in range(num_sensors):
            base_speed = np.random.uniform(35, 70)
            
            trend = np.linspace(0, 3, len(timestamps))
           
            hours = np.array([t.hour + t.minute/60 for t in timestamps])
            morning_rush = -12 * np.exp(-((hours - 8)**2) / 6)
            evening_rush = -15 * np.exp(-((hours - 17)**2) / 8)
            daily_pattern = morning_rush + evening_rush
        
            day_of_week = np.array([t.dayofweek for t in timestamps])
            weekend_boost = np.where(day_of_week >= 5, 8, 0)
        
            random_events = np.random.choice([-20, 0], len(timestamps), p=[0.02, 0.98])
            noise = np.random.normal(0, 4, len(timestamps))
            speed = base_speed + trend + daily_pattern + weekend_boost + random_events + noise
            speed = np.clip(speed, 5, 85) 
            data.append(speed)
        
        df = pd.DataFrame(
            np.array(data).T,
            index=timestamps,
            columns=[f'sensor_{i}' for i in range(num_sensors)]
        )
        
        print(f"Generated synthetic data shape: {df.shape}")
        return df
    
    def preprocess_data(self, df, input_len=12, horizon=12, train_ratio=0.7, val_ratio=0.1):
        print("Starting preprocessing...")
        
        df = df.interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
    
        features_df = pd.DataFrame(index=df.index)
        hour = df.index.hour
        features_df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        features_df['hour_cos'] = np.cos(2 * np.pi * hour / 24)
        
        day = df.index.dayofweek
        features_df['day_sin'] = np.sin(2 * np.pi * day / 7)
        features_df['day_cos'] = np.cos(2 * np.pi * day / 7)
    
        features_df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
        features_df['is_morning_rush'] = ((df.index.hour >= 7) & (df.index.hour <= 9)).astype(int)
        features_df['is_evening_rush'] = ((df.index.hour >= 16) & (df.index.hour <= 19)).astype(int)
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        train_df = df.iloc[:train_end]
        val_df = df.iloc[train_end:val_end]
        test_df = df.iloc[val_end:]
        
        train_feat = features_df.iloc[:train_end]
        val_feat = features_df.iloc[train_end:val_end]
        test_feat = features_df.iloc[val_end:]
        self.scaler.fit(train_df.values)
        train_norm = self.scaler.transform(train_df.values)
        val_norm = self.scaler.transform(val_df.values)
        test_norm = self.scaler.transform(test_df.values)
        def create_sequences(data, features, input_len, horizon):
            X, Y, X_feat = [], [], []
            for i in range(len(data) - input_len - horizon + 1):
                X.append(data[i:i+input_len])
                Y.append(data[i+input_len:i+input_len+horizon])
                X_feat.append(features[i:i+input_len])
            return np.array(X), np.array(Y), np.array(X_feat)
        
        X_train, Y_train, X_train_feat = create_sequences(train_norm, train_feat.values, input_len, horizon)
        X_val, Y_val, X_val_feat = create_sequences(val_norm, val_feat.values, input_len, horizon)
        X_test, Y_test, X_test_feat = create_sequences(test_norm, test_feat.values, input_len, horizon)
        num_sensors = df.shape[1]
        adj_matrix = self.create_adjacency_matrix(num_sensors)
        np.save(f'{self.output_dir}/X_train.npy', X_train)
        np.save(f'{self.output_dir}/Y_train.npy', Y_train)
        np.save(f'{self.output_dir}/X_train_feat.npy', X_train_feat)
        np.save(f'{self.output_dir}/X_val.npy', X_val)
        np.save(f'{self.output_dir}/Y_val.npy', Y_val)
        np.save(f'{self.output_dir}/X_val_feat.npy', X_val_feat)
        np.save(f'{self.output_dir}/X_test.npy', X_test)
        np.save(f'{self.output_dir}/Y_test.npy', Y_test)
        np.save(f'{self.output_dir}/X_test_feat.npy', X_test_feat)
        np.save(f'{self.output_dir}/adj_matrix.npy', adj_matrix)
        
        with open(f'{self.output_dir}/scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        metadata = {
            'num_sensors': num_sensors,
            'input_len': input_len,
            'horizon': horizon,
            'num_features': features_df.shape[1],
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'test_samples': len(X_test)
        }
        
        with open(f'{self.output_dir}/metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Preprocessing complete! Saved to {self.output_dir}")
        print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        return X_train, Y_train, X_train_feat, X_val, Y_val, X_val_feat, X_test, Y_test, X_test_feat, adj_matrix, metadata
    
    def create_adjacency_matrix(self, num_sensors):
        locations = np.random.rand(num_sensors, 2) * 100
        
        distances = np.zeros((num_sensors, num_sensors))
        for i in range(num_sensors):
            for j in range(num_sensors):
                distances[i, j] = np.linalg.norm(locations[i] - locations[j])
        
        threshold = np.percentile(distances[distances > 0], 15)
        adj = (distances <= threshold).astype(float)
        
        np.fill_diagonal(adj, 1)
        degree = np.sum(adj, axis=1)
        degree_inv_sqrt = np.power(degree + 1e-6, -0.5)
        adj_norm = adj * degree_inv_sqrt[:, None] * degree_inv_sqrt[None, :]
        
        return adj_norm


class LSTMModel(nn.Module):
    """LSTM baseline model"""
    
    def __init__(self, num_sensors, input_len, horizon, hidden_dim=64, num_layers=2, dropout=0.2, num_features=7):
        super(LSTMModel, self).__init__()
        
        self.num_sensors = num_sensors
        self.input_len = input_len
        self.horizon = horizon
    
        self.lstm = nn.LSTM(
            input_size=num_sensors + num_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
    
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_sensors * horizon)
        )
        
    def forward(self, x, features=None):
        if features is not None:
            x = torch.cat([x, features], dim=-1)
        
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        out = self.fc(last_hidden)
        predictions = out.view(-1, self.horizon, self.num_sensors)
        
        return predictions


class GraphConvolution(nn.Module):
    """Graph convolution layer"""
    
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)
    
    def forward(self, x, adj):
        support = torch.matmul(x, self.weight)
        output = torch.matmul(adj, support) + self.bias
        return output


class STGNNBlock(nn.Module):
    """Spatio-Temporal Graph Neural Network Block"""
    
    def __init__(self, in_channels, out_channels, num_nodes):
        super(STGNNBlock, self).__init__()
        
        self.graph_conv = GraphConvolution(in_channels, out_channels)
        self.temporal_conv = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.LayerNorm([num_nodes, out_channels])
        
    def forward(self, x, adj):
        batch_size, seq_len, num_nodes, _ = x.shape
        
        spatial_out = []
        for t in range(seq_len):
            xt = x[:, t, :, :]
            st = F.relu(self.graph_conv(xt, adj))
            spatial_out.append(st)
        
        spatial_out = torch.stack(spatial_out, dim=1)
        spatial_out = spatial_out.permute(0, 2, 3, 1).contiguous()
        spatial_out = spatial_out.view(batch_size * num_nodes, -1, seq_len)
        temporal_out = F.relu(self.temporal_conv(spatial_out))
        temporal_out = temporal_out.view(batch_size, num_nodes, -1, seq_len)
        temporal_out = temporal_out.permute(0, 3, 1, 2).contiguous()

        out = temporal_out + x
        out = self.norm(out)
        
        return out


class GNNTrafficModel(nn.Module):
    """Graph Neural Network for traffic prediction"""
    
    def __init__(self, num_sensors, input_len, horizon, hidden_dim=64, num_blocks=2, num_features=7):
        super(GNNTrafficModel, self).__init__()
        
        self.num_sensors = num_sensors
        self.input_len = input_len
        self.horizon = horizon
        
        self.input_embed = nn.Linear(1 + num_features, hidden_dim)
    
        self.st_blocks = nn.ModuleList([
            STGNNBlock(hidden_dim, hidden_dim, num_sensors)
            for _ in range(num_blocks)
        ])
        
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, horizon)
        )
    
    def forward(self, x, adj, features=None):
        batch_size = x.size(0)
    
        x = x.unsqueeze(-1) 
        
        if features is not None:
            features = features.unsqueeze(2).expand(-1, -1, self.num_sensors, -1)
            x = torch.cat([x, features], dim=-1)
        
        x = self.input_embed(x)
        
    
        for block in self.st_blocks:
            x = block(x, adj)
        
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size * self.num_sensors, self.input_len, -1)
        
        _, h_n = self.gru(x)
        h_n = h_n.squeeze(0)
   
        predictions = self.output_proj(h_n)
        predictions = predictions.view(batch_size, self.num_sensors, self.horizon)
        predictions = predictions.permute(0, 2, 1).contiguous()
        
        return predictions


class TrafficPredictor:
    """Main training and prediction class"""
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
    def load_processed_data(self, processed_dir='processed_data'):
        """Load preprocessed data"""
        X_train = np.load(f'{processed_dir}/X_train.npy')
        Y_train = np.load(f'{processed_dir}/Y_train.npy')
        X_train_feat = np.load(f'{processed_dir}/X_train_feat.npy')
        X_val = np.load(f'{processed_dir}/X_val.npy')
        Y_val = np.load(f'{processed_dir}/Y_val.npy')
        X_val_feat = np.load(f'{processed_dir}/X_val_feat.npy')
        X_test = np.load(f'{processed_dir}/X_test.npy')
        Y_test = np.load(f'{processed_dir}/Y_test.npy')
        X_test_feat = np.load(f'{processed_dir}/X_test_feat.npy')
        adj_matrix = np.load(f'{processed_dir}/adj_matrix.npy')
        
        with open(f'{processed_dir}/metadata.json', 'r') as f:
            metadata = json.load(f)
        
        with open(f'{processed_dir}/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        return (X_train, Y_train, X_train_feat, X_val, Y_val, X_val_feat, 
                X_test, Y_test, X_test_feat, adj_matrix, metadata, scaler)
    
    def train_model(self, model, train_loader, val_loader, adj_matrix=None, epochs=50, lr=0.001):
        """Training loop"""
        model = model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.L1Loss()  # MAE loss
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        
        if adj_matrix is not None:
            adj_matrix = torch.FloatTensor(adj_matrix).to(self.device)
        
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses, val_losses = [], []
        
        print(f"Training for {epochs} epochs...")
        
        for epoch in range(epochs):
           
            model.train()
            train_loss = 0
            for batch_x, batch_y, batch_feat in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                batch_feat = batch_feat.to(self.device)
                
                optimizer.zero_grad()
                
                if adj_matrix is not None:
                    pred = model(batch_x, adj_matrix, batch_feat)
                else:
                    pred = model(batch_x, batch_feat)
                
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
    
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_x, batch_y, batch_feat in val_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)
                    batch_feat = batch_feat.to(self.device)
                    
                    if adj_matrix is not None:
                        pred = model(batch_x, adj_matrix, batch_feat)
                    else:
                        pred = model(batch_x, batch_feat)
                    
                    loss = criterion(pred, batch_y)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            scheduler.step(val_loss)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            if epoch % 10 == 0:
                print(f'Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            
    
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= 10:
                    print(f'Early stopping at epoch {epoch+1}')
                    break
        
      
        model.load_state_dict(torch.load('best_model.pth'))
        
        return model, train_losses, val_losses
    
    def evaluate_model(self, model, test_loader, adj_matrix=None, scaler=None):
        """Evaluate model performance"""
        model.eval()
        predictions = []
        actuals = []
        
        if adj_matrix is not None:
            adj_matrix = torch.FloatTensor(adj_matrix).to(self.device)
        
        with torch.no_grad():
            for batch_x, batch_y, batch_feat in test_loader:
                batch_x = batch_x.to(self.device)
                batch_feat = batch_feat.to(self.device)
                
                if adj_matrix is not None:
                    pred = model(batch_x, adj_matrix, batch_feat)
                else:
                    pred = model(batch_x, batch_feat)
                
                predictions.append(pred.cpu().numpy())
                actuals.append(batch_y.numpy())
        
        predictions = np.concatenate(predictions, axis=0)
        actuals = np.concatenate(actuals, axis=0)
        
        if scaler is not None:
            pred_shape = predictions.shape
            actual_shape = actuals.shape
            
            predictions_reshaped = predictions.reshape(-1, predictions.shape[-1])
            actuals_reshaped = actuals.reshape(-1, actuals.shape[-1])
            
            predictions_denorm = scaler.inverse_transform(predictions_reshaped)
            actuals_denorm = scaler.inverse_transform(actuals_reshaped)
            
            predictions = predictions_denorm.reshape(pred_shape)
            actuals = actuals_denorm.reshape(actual_shape)
     
        mae = mean_absolute_error(actuals.flatten(), predictions.flatten())
        rmse = np.sqrt(mean_squared_error(actuals.flatten(), predictions.flatten()))
        mape = np.mean(np.abs((actuals - predictions) / (actuals + 1e-5))) * 100
        
  
        horizon_metrics = {}
        for h in [3, 6, 12]:  # 15min, 30min, 60min
            if h <= predictions.shape[1]:
                pred_h = predictions[:, :h, :]
                actual_h = actuals[:, :h, :]
                
                mae_h = mean_absolute_error(actual_h.flatten(), pred_h.flatten())
                rmse_h = np.sqrt(mean_squared_error(actual_h.flatten(), pred_h.flatten()))
                mape_h = np.mean(np.abs((actual_h - pred_h) / (actual_h + 1e-5))) * 100
                
                horizon_metrics[f'{h*5}min'] = {
                    'MAE': mae_h,
                    'RMSE': rmse_h,
                    'MAPE': mape_h
                }
        
        metrics = {
            'overall': {'MAE': mae, 'RMSE': rmse, 'MAPE': mape},
            'by_horizon': horizon_metrics
        }
        
        return predictions, actuals, metrics
    
    def plot_results(self, predictions, actuals, train_losses, val_losses, metrics):
        """Plot training curves and predictions"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
   
        axes[0, 0].plot(train_losses, label='Train Loss')
        axes[0, 0].plot(val_losses, label='Val Loss')
        axes[0, 0].set_title('Training Curves')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss (MAE)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
      
        sample_idx = np.random.randint(0, predictions.shape[0])
        sensor_idx = np.random.randint(0, predictions.shape[2])
        
        axes[0, 1].plot(actuals[sample_idx, :, sensor_idx], 'o-', label='Actual', alpha=0.7)
        axes[0, 1].plot(predictions[sample_idx, :, sensor_idx], 's-', label='Predicted', alpha=0.7)
        axes[0, 1].set_title(f'Sample Prediction (Sensor {sensor_idx})')
        axes[0, 1].set_xlabel('Time Steps (5-min intervals)')
        axes[0, 1].set_ylabel('Traffic Speed')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
       
        errors = predictions - actuals
        axes[1, 0].hist(errors.flatten(), bins=50, alpha=0.7, density=True)
        axes[1, 0].set_title('Prediction Error Distribution')
        axes[1, 0].set_xlabel('Error')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].grid(True, alpha=0.3)

        horizons = list(metrics['by_horizon'].keys())
        maes = [metrics['by_horizon'][h]['MAE'] for h in horizons]
        
        axes[1, 1].bar(horizons, maes, alpha=0.7)
        axes[1, 1].set_title('MAE by Prediction Horizon')
        axes[1, 1].set_xlabel('Horizon')
        axes[1, 1].set_ylabel('MAE')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('traffic_prediction_results.png', dpi=150, bbox_inches='tight')
        plt.show()
        

        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        print(f"\nOverall Metrics:")
        for metric, value in metrics['overall'].items():
            print(f"  {metric}: {value:.4f}")
        
        print(f"\nMetrics by Horizon:")
        for horizon, horizon_metrics in metrics['by_horizon'].items():
            print(f"\n{horizon}:")
            for metric, value in horizon_metrics.items():
                print(f"  {metric}: {value:.4f}")
    
    def run_complete_pipeline(self, model_type='lstm', epochs=50, batch_size=64):
        """Run the complete pipeline"""
        print("="*60)
        print("TRAFFIC PREDICTION MODEL - COMPLETE PIPELINE")
        print("="*60)
        
        print("\n1. PREPROCESSING DATA...")
        preprocessor = TrafficDataPreprocessor(self.data_path)
        df = preprocessor.load_data()
        
        (X_train, Y_train, X_train_feat, X_val, Y_val, X_val_feat, 
         X_test, Y_test, X_test_feat, adj_matrix, metadata) = preprocessor.preprocess_data(df)

        print("\n2. CREATING DATA LOADERS...")
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(Y_train),
            torch.FloatTensor(X_train_feat)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.FloatTensor(Y_val),
            torch.FloatTensor(X_val_feat)
        )
        test_dataset = TensorDataset(
            torch.FloatTensor(X_test),
            torch.FloatTensor(Y_test),
            torch.FloatTensor(X_test_feat)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        print(f"\n3. CREATING {model_type.upper()} MODEL...")
        if model_type == 'lstm':
            model = LSTMModel(
                num_sensors=metadata['num_sensors'],
                input_len=metadata['input_len'],
                horizon=metadata['horizon'],
                num_features=metadata['num_features']
            )
            adj = None
        elif model_type == 'gnn':
            model = GNNTrafficModel(
                num_sensors=metadata['num_sensors'],
                input_len=metadata['input_len'],
                horizon=metadata['horizon'],
                num_features=metadata['num_features']
            )
            adj = adj_matrix
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"\n4. TRAINING MODEL...")
        model, train_losses, val_losses = self.train_model(
            model, train_loader, val_loader, adj, epochs
        )
        print(f"\n5. EVALUATING MODEL...")
        predictions, actuals, metrics = self.evaluate_model(
            model, test_loader, adj, preprocessor.scaler
        )
        print(f"\n6. GENERATING RESULTS...")
        self.plot_results(predictions, actuals, train_losses, val_losses, metrics)
        
        print(f"\nPipeline complete! Results saved to 'traffic_prediction_results.png'")
        
        return model, predictions, actuals, metrics


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Traffic Prediction Model')
    parser.add_argument('--data_path', type=str, 
                       default=r'D:\OneDrive\Documents\Traffic Prediction Model\METR.LA',
                       help='Path to data directory')
    parser.add_argument('--model', type=str, default='lstm', choices=['lstm', 'gnn'],
                       help='Model type')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    
    args = parser.parse_args()

    predictor = TrafficPredictor(args.data_path)
    model, predictions, actuals, metrics = predictor.run_complete_pipeline(
        model_type=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)


if __name__ == '__main__':
    main()
