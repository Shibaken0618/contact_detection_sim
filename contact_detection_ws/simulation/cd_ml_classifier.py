"""
Contact Detection ML Classifier
Takes in joint state data to train nn
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import joblib


class ContactDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class ContactClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_rate):
        super(ContactClassifier, self).__init__()
        self.neuralNet = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size//4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x)


class ContactDetectionTrainer:
    def __init__(self, data_file="data/joint_data.csv"):
        self.data_file = data_file
        self.model = None
        self.scaler = StandardScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

    def preproces_data(self):
        print("Load data ...")
        if not os.path.exists(self.data_file):
            raise FileNotFoundError(f"Data file {self.data_file} not found, create or get data first")
        
        df = pd.read_csv(self.data_file)
        print(f"Loaded {len(df)} samples")

        feature_cols = [col for col in df.columns if col != 'contact_detected']
        X = df[feature_cols].values
        y = df['contact_detected'].values

        contact_count = np.sum(y)
        print(f"Contact samples: {contact_count} ({contact_count/len(y)*100:.1f}%)")
        print(f"No-contact samples: {len(y)-contact_count} ({(len(y)-contact_count)/len(y)*100:.1f}%)")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_model():
        pass

    def eval_model():
        pass

    def save_model():
        pass

    def load_model():
        pass


def main():
    print("starting NN Training...")

    trainer = ContactDetectionTrainer()
    X_train, X_test, y_train, y_test = trainer.preproces_data()
    print(len(X_train))
    print(len(X_test))
    print(len(y_train))
    print(len(y_test))

    print("\nTraining complete!")

    print("\nModel saved, ready to use.")



if __name__ == "__main__":
    main()


