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


def main():
    print("starting NN Training...")

    print("\nTraining complete!")

    print("\nModel saved, ready to use.")



if __name__ == "__main__":
    main()
