import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# --- PART 1: DATASET CLASS ---
class RadiomicsDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# --- PART 2: NEURAL NETWORK MODEL ---
class CancerDetectorMLP(nn.Module):
    def __init__(self, input_dim):
        super(CancerDetectorMLP, self).__init__()
        # Reduced network size since input features are compressed and uncorrelated
        self.network = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)

# --- PART 3: DATA PROCESSING PIPELINE ---
def load_and_process_data(csv_path, n_components=0.95):
    """
    Loads data, splits it, scales it, and runs PCA.
    n_components=0.95 means 'keep enough components to explain 95% of variance'
    """
    # Load Data
    df = pd.read_csv(csv_path)
    
    # Filter for only numeric radiomics features
    feature_cols = [c for c in df.columns if "original_" in c]
    X = df[feature_cols].values.astype(np.float32)
    
    # --- LABEL HANDLING ---
    # TODO: Replace this with real label logic (e.g., reading from a separate clinical CSV)
    print("WARNING: Generating random dummy labels for demonstration.")
    y = np.random.randint(0, 2, size=len(df)).astype(np.float32)
    # ----------------------

    # 1. SPLIT (Must happen before Scaling/PCA to prevent leakage)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. SCALE (Fit on Train, Apply to Test)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 3. PCA (Fit on Train, Apply to Test)
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    print(f"Original feature count: {X.shape[1]}")
    print(f"PCA feature count:      {X_train_pca.shape[1]} (retained {n_components*100}% variance)")

    return X_train_pca, X_test_pca, y_train, y_test

# --- PART 4: TRAINING LOOP ---
def train_model():
    CSV_PATH = 'pyradiomics/lung_left_data.csv'
    BATCH_SIZE = 4
    LEARNING_RATE = 0.001
    EPOCHS = 20
    
    try:
        X_train, X_test, y_train, y_test = load_and_process_data(CSV_PATH)
    except FileNotFoundError:
        print(f"Error: Could not find {CSV_PATH}. Run the extraction pipeline first.")
        return
    except ValueError as e:
        print(f"Error processing data: {e}")
        return

    # Create Datasets
    train_dataset = RadiomicsDataset(X_train, y_train)
    # test_dataset = RadiomicsDataset(X_test, y_test) # Reserved for validation

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Initialize Model
    input_dim = X_train.shape[1] # This is now the number of PCA components
    model = CancerDetectorMLP(input_dim)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Starting training on {len(train_dataset)} samples...")

    # Train
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for features, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Print average loss every 5 epochs
        if (epoch + 1) % 5 == 0:
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}")

    print("Training complete.")

if __name__ == "__main__":
    train_model()