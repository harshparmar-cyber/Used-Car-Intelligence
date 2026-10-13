
import numpy as np
import pandas as pd
import matplotlib
import sklearn
import scipy
import streamlit

print("All libraries imported successfully!")

df = pd.read_csv("data/train.csv")

print("\nDataset loaded successfully!")
print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])

print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst five rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())
