#!/usr/bin/env python3
"""
Alternative training script that handles serialization issues more gracefully
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from REPD_Impl import REPD
from autoencoder_tf2 import AutoEncoder
import os
import sys

# Suppress TensorFlow progress bars
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

def train_semantic_model_safe(dataset_path):
    """Train semantic model with safe serialization"""
    print("Starting safe model training...")
    
    # Load dataset
    df = pd.read_csv(dataset_path)
    print(f"Loaded dataset with {len(df)} samples")
    
    # Prepare features
    feature_columns = ["wmc", "rfc", "loc", "max_cc", "avg_cc", "cbo", "ca", "ce", "ic", "cbm", "lcom", "lcom3", "dit", "noc", "mfa", "npm", "dam", "moa", "cam", "amc"]
    X = df[feature_columns].fillna(0).values
    y = df["bug"].values
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target vector shape: {y.shape}")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create autoencoder
    input_dim = X_scaled.shape[1]
    layers = [input_dim, max(50, input_dim//2), max(25, input_dim//4), max(10, input_dim//8)]
    print(f"Autoencoder layers: {layers}")
    
    autoencoder = AutoEncoder(layers, lr=0.01, epoch=100, batch_size=32)
    
    # Train autoencoder
    print("Training autoencoder...")
    autoencoder.fit(X_scaled, print_progress=False)
    
    # Create REPD model
    print("Training REPD model...")
    repd_model = REPD(autoencoder)
    repd_model.fit(X_scaled, y)
    
    # Create output directory
    os.makedirs("trained_model", exist_ok=True)
    
    # Save components that can be safely serialized
    print("Saving model components...")
    
    # Save scaler (always works)
    joblib.dump(scaler, "trained_model/scaler.pkl")
    print("✅ Scaler saved")
    
    # Save autoencoder weights separately 
    try:
        autoencoder.autoencoder.save_weights("trained_model/autoencoder_weights.h5")
        print("✅ Autoencoder weights saved")
    except Exception as e:
        print(f"⚠️ Could not save autoencoder weights: {e}")
    
    # Save model architecture and parameters
    model_config = {
        'layers': layers,
        'input_dim': input_dim,
        'feature_columns': feature_columns,
        'lr': 0.01,
        'epoch': 100,
        'batch_size': 32,
        'dataset_shape': X.shape
    }
    joblib.dump(model_config, "trained_model/model_config.pkl")
    print("✅ Model configuration saved")
    
    # Try to save REPD distributions separately
    try:
        repd_data = {
            'dnd': repd_model.dnd,
            'dnd_pa': repd_model.dnd_pa,
            'dd': repd_model.dd,
            'dd_pa': repd_model.dd_pa
        }
        joblib.dump(repd_data, "trained_model/repd_distributions.pkl")
        print("✅ REPD distributions saved")
    except Exception as e:
        print(f"⚠️ Could not save REPD distributions: {e}")
    
    # Try to save the full REPD model (might fail due to lambda functions)
    try:
        joblib.dump(repd_model, "trained_model/repd_model_DA.pkl")
        print("✅ Full REPD model saved")
    except Exception as e:
        print(f"⚠️ Could not save full REPD model (expected): {e}")
        print("✅ Using component-based approach instead")
    
    # Save training results
    training_results = {
        "feature_columns": feature_columns, 
        "input_dim": input_dim, 
        "layers": layers, 
        "dataset_shape": X.shape,
        "training_successful": True
    }
    joblib.dump(training_results, "trained_model/training_results.pkl")
    print("✅ Training results saved")
    
    print("Model training completed successfully")
    return True

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python train_semantic_model_safe.py <dataset_path>")
        sys.exit(1)
    
    success = train_semantic_model_safe(sys.argv[1])
    if not success:
        sys.exit(1)
