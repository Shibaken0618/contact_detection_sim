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


EPOCHS = 10
BATCH_SIZE = 32
DATA_FILE = "data/joint_data.csv"
MODEL_PATH = "models/contact_detector.pth"
SCALER_PATH = "models/scaler.pkl"


class ContactDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]



class ContactClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=64, dropout_rate=0.2):
        super(ContactClassifier, self).__init__()
        self.neuralNet = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size//4),
            nn.ReLU(),
            nn.Linear(hidden_size//4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.neuralNet(x)


class ContactDetectionTrainer:
    def __init__(self, data_file=DATA_FILE):
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
    

    def train_model(self, X_train, y_train, X_test, y_test, epochs=EPOCHS, batch_size=BATCH_SIZE):
        print("Training started ...")
        train_dataset = ContactDataset(X_train, y_train)
        test_dataset = ContactDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        input_size=X_train.shape[1]
        self.model = ContactClassifier(input_size).to(self.device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        train_losses = []
        test_losses = []

        for epoch in range(epochs):
            # train
            self.model.train()
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
        
            self.model.eval()

            # val
            test_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    outputs = self.model(batch_X).squeeze()
                    loss = criterion(outputs, batch_y)
                    test_loss += loss.item()
            
            train_loss /= len(train_loader)
            test_loss /= len(test_loader)

            train_losses.append(train_loss)
            test_losses.append(test_loss)

            if epoch % 1 == 0:
                print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
        
        return train_losses, test_losses


    def eval_model(self, X_test, y_test):
        print("Evaluating model ...")
        self.model.eval()
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test).to(self.device)
            predictions = self.model(X_test_tensor).squeeze().cpu().numpy()
            y_pred = (predictions > 0.5).astype(int)
            print("\nClassification results:")
            print(classification_report(y_test, y_pred))

            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(8,6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matirx')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.savefig('confusion_matrix.png')
            plt.show()

            return predictions, y_pred


    def save_model(self, model_path=MODEL_PATH, scaler_path=SCALER_PATH):
        os.makedirs("models", exist_ok=True)
        torch.save(self.model.state_dict(), model_path)
        joblib.dump(self.scaler, scaler_path)
        print("model and scaler saved")


    def load_model(self, model_path=MODEL_PATH, scaler_path=SCALER_PATH):
        self.scaler = joblib.load(scaler_path)
        input_size = len(self.scaler.feature_names_in_) if hasattr(self.scaler, 'feature_names_in_') else 7*3
        self.model = ContactClassifier(input_size).to(self.device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        print("Model loaded.")

    def predict_contact(self, joint_pos, joint_vel, joint_tor):
        if self.model is None:
            raise ValueError("Model not trained or loaded")

        features = np.concatenate([joint_pos, joint_vel, joint_tor])
        features_scaled = self.scaler.transform(features.reshape(1, -1))

        with torch.no_grad():
            features_tensor = torch.FloatTensor(features_scaled).to(self.device)
            prediction = self.model(features_tensor).squeeze().cpu().numpy()
        
        return prediction


def main():
    print("starting NN Training...")

    trainer = ContactDetectionTrainer()
    try:
        X_train, X_test, y_train, y_test = trainer.preproces_data()
        # print(len(X_train))
        # print(len(X_test))
        # print(len(y_train))
        # print(len(y_test))

        train_losses, test_losses = trainer.train_model(X_train, y_train, X_test, y_test)
        # print(train_losses[0])
        # print(test_losses[0])
        plt.figure(figsize=(10,5))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(test_losses, label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training History')
        plt.savefig('training_history.png')
        plt.show()


        print("\nTraining complete!")

        predictions, y_pred = trainer.eval_model(X_test, y_test)
        # print(predictions[:5])
        # print(y_pred[:5])
        trainer.save_model()

        print("\nModel saved, ready to use.")
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure to run the simulator script to generate training data")



if __name__ == "__main__":
    main()


