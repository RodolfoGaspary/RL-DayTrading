
import numpy as np
import pandas as pd
import joblib
import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
try:
    from stable_baselines3 import PPO
except ImportError:
    PPO = None

class BaseModel:
    def train(self, X, y):
        raise NotImplementedError
    
    def predict(self, X):
        raise NotImplementedError
    
    def save(self, path):
        raise NotImplementedError
    
    def load(self, path):
        raise NotImplementedError

    def evaluate(self, X, y):
        predictions = self.predict(X)
        return accuracy_score(y, predictions)

class RandomForestModel(BaseModel):
    def __init__(self, n_estimators=100, max_depth=10):
        self.model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        
    def train(self, X, y):
        # Flatten X for Random Forest if it's 3D (samples, window, features)
        if len(X.shape) == 3:
            nsamples, nx, ny = X.shape
            X_flat = X.reshape((nsamples, nx*ny))
        else:
            X_flat = X
            
        print("Training Random Forest...")
        self.model.fit(X_flat, y)
        print("Training complete.")
        
    def predict(self, X):
        if len(X.shape) == 3:
            nsamples, nx, ny = X.shape
            X_flat = X.reshape((nsamples, nx*ny))
        else:
            X_flat = X
        return self.model.predict(X_flat)

    def get_confidence(self, X):
        if len(X.shape) == 3:
            nsamples, nx, ny = X.shape
            X_flat = X.reshape((nsamples, nx*ny))
        else:
            X_flat = X
        # Return probability of class 1 (Bullish)
        return self.model.predict_proba(X_flat)[:, 1]
        
    def save(self, path):
        joblib.dump(self.model, path)
        
    def load(self, path):
        self.model = joblib.load(path)

# PyTorch LSTM Module
class LSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid() # For binary classification
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return self.sigmoid(out)

class LSTMModel(BaseModel):
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=1, epochs=10, learning_rate=0.001):
        self.input_size = input_size # Number of features per timestep
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        
    def train(self, X, y, batch_size=64):
        # X shape: (samples, window_size, features)
        # y shape: (samples,)
        
        # Ensure we have the input size matching the data
        if self.model is None:
             self.model = LSTMNet(X.shape[2], self.hidden_size, self.num_layers, self.output_size).to(self.device)
        
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Use DataLoader for minibatches
        dataset = torch.utils.data.TensorDataset(
            torch.tensor(X, dtype=torch.float32) if not torch.is_tensor(X) else X.clone().detach(), 
            torch.tensor(y, dtype=torch.float32).unsqueeze(1) if not torch.is_tensor(y) else y.clone().detach().unsqueeze(1)
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        print(f"Training LSTM on {self.device} with batch size {batch_size}...")
        self.model.train()
        pbar = tqdm(range(self.epochs), desc="LSTM Training")
        for epoch in pbar:
            epoch_loss = 0
            for X_batch, y_batch in loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            pbar.set_postfix(loss=f"{epoch_loss/len(loader):.4f}")
                
    def predict(self, X):
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            outputs = self.model(X_tensor)
            predicted = (outputs > 0.5).float()
        return predicted.cpu().numpy().flatten()
    
    def get_confidence(self, X):
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            outputs = self.model(X_tensor)
        return outputs.cpu().numpy().flatten()

    def save(self, path):
        torch.save(self.model.state_dict(), path)
        
    def load(self, path):
        # We need to initialize the model first with correct dimensions. 
        # For simplicity, we assume dimensions are handled or known.
        # Ideally, save config along with weights.
        if self.model is None:
             self.model = LSTMNet(self.input_size, self.hidden_size, self.num_layers, self.output_size).to(self.device)
        self.model.load_state_dict(torch.load(path))
        self.model.to(self.device)

# PyTorch Transformer Module
class TransformerNet(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_layers, output_size, window_size):
        super(TransformerNet, self).__init__()
        self.d_model = d_model
        # Project input to d_model
        self.embedding = nn.Linear(input_size, d_model)
        # Use simple positional encoding (could be more complex)
        self.pos_encoder = nn.Parameter(torch.zeros(1, window_size, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc = nn.Linear(d_model, output_size)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x shape: (batch, window, features)
        x = self.embedding(x) + self.pos_encoder
        x = self.transformer_encoder(x)
        # Use last time step
        out = self.fc(x[:, -1, :])
        return self.sigmoid(out)

class TransformerModel(BaseModel):
    def __init__(self, input_size, window_size=60, d_model=64, nhead=4, num_layers=2, output_size=1, epochs=10, learning_rate=0.001):
        self.input_size = input_size
        self.window_size = window_size
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.output_size = output_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        
    def train(self, X, y, batch_size=64):
        if self.model is None:
            self.model = TransformerNet(X.shape[2], self.d_model, self.nhead, self.num_layers, self.output_size, self.window_size).to(self.device)
        
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        dataset = torch.utils.data.TensorDataset(
            torch.tensor(X, dtype=torch.float32) if not torch.is_tensor(X) else X.clone().detach(), 
            torch.tensor(y, dtype=torch.float32).unsqueeze(1) if not torch.is_tensor(y) else y.clone().detach().unsqueeze(1)
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        print(f"Training Transformer on {self.device} with batch size {batch_size}...")
        self.model.train()
        pbar = tqdm(range(self.epochs), desc="Transformer Training")
        for epoch in pbar:
            epoch_loss = 0
            for X_batch, y_batch in loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            pbar.set_postfix(loss=f"{epoch_loss/len(loader):.4f}")
                
    def predict(self, X):
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            outputs = self.model(X_tensor)
            predicted = (outputs > 0.5).float()
        return predicted.cpu().numpy().flatten()
    
    def get_confidence(self, X):
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            outputs = self.model(X_tensor)
        return outputs.cpu().numpy().flatten()

    def save(self, path):
        torch.save(self.model.state_dict(), path)
        
    def load(self, path):
        if self.model is None:
             self.model = TransformerNet(self.input_size, self.d_model, self.nhead, self.num_layers, self.output_size, self.window_size).to(self.device)
        self.model.load_state_dict(torch.load(path))
        self.model.to(self.device)

class PPOModel(BaseModel):
    def __init__(self, model_path=None):
        self.model = None
        if PPO is None:
            print("Warning: stable-baselines3 not installed. RL models disabled.")
        if model_path:
            self.load(model_path)
            
    def train(self, env, total_timesteps=50000):
        if PPO is None: return
        self.model = PPO("MlpPolicy", env, verbose=1)
        self.model.learn(total_timesteps=total_timesteps)
        
    def predict(self, X):
        """
        X should be the current observation vector
        """
        if self.model is None: return np.array([0])
        action, _ = self.model.predict(X, deterministic=True)
        return action
        
    def save(self, path):
        if self.model:
            self.model.save(path)
            
    def load(self, path):
        if PPO is None: return
        self.model = PPO.load(path)

def get_model(model_type, input_shape=None):
    if model_type == "RandomForest":
        return RandomForestModel()
    elif model_type == "LSTM":
        if input_shape is None:
             raise ValueError("Input shape required for LSTM")
        return LSTMModel(input_size=input_shape[2]) # (batch, window, features)
    elif model_type == "Transformer":
        if input_shape is None:
             raise ValueError("Input shape required for Transformer")
        return TransformerModel(input_size=input_shape[2], window_size=input_shape[1])
    elif model_type == "PPO":
        return PPOModel()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
