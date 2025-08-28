#!/usr/bin/env python3
"""
Train REPD Model on Non-Defective Examples

This script trains the REPD (Reconstruction Error-based Probabilistic Detection) model
on non-defective examples from the semantic features that were previously encoded.

The REPD model works by:
1. Training a dimensionality reduction model (autoencoder) on non-defective examples
2. Computing reconstruction errors for both defective and non-defective examples
3. Fitting probability distributions to the reconstruction errors
4. Using these distributions for classification

Author: REPD Implementation
"""

import numpy as np
import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Import our custom modules
from REPD_Impl import REPD
from simple_autoencoder import DeepAutoencoder, ConvolutionalAutoencoder, SimpleAutoencoder
import sys
sys.path.append('semantic-dataset-creation')
from dbn import DBN

class REPDTrainer:
    def __init__(self, data_path="data/openj9_metrics.csv"):
        self.data_path = data_path
        self.scaler = StandardScaler()
        self.models = {}
        self.training_results = {}
        
    def load_data(self):
        """Load the original metrics data and semantic features"""
        print("Loading data...")
        
        # Load original metrics data
        self.original_data = pd.read_csv(self.data_path)
        print(f"Original data shape: {self.original_data.shape}")
        
        # Create binary labels (0: non-defective, 1: defective)
        self.labels = (self.original_data['bug'] > 0).astype(int)
        print(f"Class distribution: {np.bincount(self.labels)}")
        
        # Load semantic features
        self.semantic_features = {}
        feature_files = {
            'DA': 'openj9_metrics_DA_features.npy',
            'CA': 'openj9_metrics_CA_features.npy', 
            'DBN': 'openj9_metrics_DBN_features.npy'
        }
        
        for feature_type, filename in feature_files.items():
            if os.path.exists(filename):
                self.semantic_features[feature_type] = np.load(filename)
                print(f"Loaded {feature_type} features: {self.semantic_features[feature_type].shape}")
            else:
                print(f"Warning: {filename} not found")
                
        return self
    
    def prepare_features(self, feature_type='DA'):
        """Prepare features for training"""
        if feature_type not in self.semantic_features:
            raise ValueError(f"Feature type {feature_type} not available")
            
        X = self.semantic_features[feature_type]
        y = self.labels
        
        # Handle DBN features which have shape (n_samples, 1, n_features)
        if feature_type == 'DBN' and len(X.shape) == 3:
            X = X.reshape(X.shape[0], X.shape[2])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Create a new scaler for each feature type to avoid conflicts
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test, scaler
    
    def create_autoencoder(self, feature_type='DA', input_dim=None):
        """Create appropriate autoencoder based on feature type"""
        if input_dim is None:
            if feature_type == 'DBN' and len(self.semantic_features[feature_type].shape) == 3:
                input_dim = self.semantic_features[feature_type].shape[2]
            else:
                input_dim = self.semantic_features[feature_type].shape[1]
            
        if feature_type == 'DA':
            # Create SimpleAutoencoder instead of DeepAutoencoder for DA features
            return SimpleAutoencoder(n_components=int(input_dim * 0.5))  # Use 50% of features
            layers = [input_dim, input_dim//2, input_dim//4, input_dim//2, input_dim]
            return DeepAutoencoder(compression_ratio=0.25)  # Use 25% compression ratio for better feature extraction
        elif feature_type == 'CA':
            # Use ConvolutionalAutoencoder
            return ConvolutionalAutoencoder(input_dim=input_dim)
        elif feature_type == 'DBN':
            # Use SimpleAutoencoder for DBN features
            return SimpleAutoencoder(n_components=input_dim//2)
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")
    
    def train_repd_model(self, feature_type='DA'):
        """Train REPD model on non-defective examples"""
        print(f"\nTraining REPD model with {feature_type} features...")
        
        try:
            # Prepare data
            X_train, X_test, y_train, y_test, scaler = self.prepare_features(feature_type)
            
            # Create autoencoder
            autoencoder = self.create_autoencoder(feature_type, X_train.shape[1])
            
            # Create and train REPD model
            repd_model = REPD(dim_reduction_model=autoencoder)
            repd_model.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = repd_model.predict(X_test)
            
            # Store results
            self.models[feature_type] = repd_model
            self.training_results[feature_type] = {
                'y_test': y_test,
                'y_pred': y_pred,
                'X_test': X_test,
                'X_train': X_train,
                'y_train': y_train,
                'scaler': scaler
            }
            
            # Print results
            print(f"\nResults for {feature_type} features:")
            print(classification_report(y_test, y_pred))
            
            return repd_model
            
        except Exception as e:
            print(f"Error training {feature_type} model: {e}")
            return None
    
    def train_all_models(self):
        """Train REPD models with all available semantic features"""
        print("Training REPD models with all semantic features...")
        
        for feature_type in self.semantic_features.keys():
            try:
                self.train_repd_model(feature_type)
            except Exception as e:
                print(f"Error training {feature_type} model: {e}")
    
    def save_models(self, output_dir="trained_models"):
        """Save trained models and results"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save models (only if they exist and are not None)
        for feature_type, model in self.models.items():
            if model is not None:
                try:
                    model_path = os.path.join(output_dir, f"repd_model_{feature_type}.pkl")
                    with open(model_path, 'wb') as f:
                        pickle.dump(model, f)
                    print(f"Saved {feature_type} model to {model_path}")
                except Exception as e:
                    print(f"Could not save {feature_type} model: {e}")
        
        # Save training results
        results_path = os.path.join(output_dir, "training_results.pkl")
        with open(results_path, 'wb') as f:
            pickle.dump(self.training_results, f)
        print(f"Saved training results to {results_path}")
    
    def plot_results(self, output_dir="plots"):
        """Plot training results"""
        os.makedirs(output_dir, exist_ok=True)
        
        for feature_type, results in self.training_results.items():
            # Confusion matrix
            plt.figure(figsize=(8, 6))
            cm = confusion_matrix(results['y_test'], results['y_pred'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix - {feature_type} Features')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.savefig(os.path.join(output_dir, f'confusion_matrix_{feature_type}.png'))
            plt.close()
            
            # Distribution plots
            if hasattr(self.models[feature_type], 'get_probability_data'):
                errors, nd_p, d_p = self.models[feature_type].get_probability_data()
                
                plt.figure(figsize=(10, 6))
                plt.plot(errors, nd_p, label='Non-Defective Probability', color='blue')
                plt.plot(errors, d_p, label='Defective Probability', color='red')
                plt.xlabel('Reconstruction Error')
                plt.ylabel('Probability')
                plt.title(f'Probability Distributions - {feature_type} Features')
                plt.legend()
                plt.grid(True)
                plt.savefig(os.path.join(output_dir, f'probability_distributions_{feature_type}.png'))
                plt.close()
    
    def print_summary(self):
        """Print training summary"""
        print("\n" + "="*50)
        print("REPD MODEL TRAINING SUMMARY")
        print("="*50)
        
        for feature_type in self.semantic_features.keys():
            if feature_type in self.training_results:
                results = self.training_results[feature_type]
                y_test = results['y_test']
                y_pred = results['y_pred']
                
                # Calculate metrics
                accuracy = np.mean(y_test == y_pred)
                precision = np.sum((y_test == 1) & (y_pred == 1)) / np.sum(y_pred == 1) if np.sum(y_pred == 1) > 0 else 0
                recall = np.sum((y_test == 1) & (y_pred == 1)) / np.sum(y_test == 1) if np.sum(y_test == 1) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                print(f"\n{feature_type} Features:")
                print(f"  Accuracy: {accuracy:.4f}")
                print(f"  Precision: {precision:.4f}")
                print(f"  Recall: {recall:.4f}")
                print(f"  F1-Score: {f1:.4f}")
                print(f"  Non-defective samples: {np.sum(y_test == 0)}")
                print(f"  Defective samples: {np.sum(y_test == 1)}")

def main():
    """Main training function"""
    print("Starting REPD model training...")
    
    # Initialize trainer
    trainer = REPDTrainer()
    
    # Load data
    trainer.load_data()
    
    # Train models
    trainer.train_all_models()
    
    # Save models and results
    trainer.save_models()
    
    # Plot results
    trainer.plot_results()
    
    # Print summary
    trainer.print_summary()
    
    print("\nTraining completed successfully!")
    print("Models saved in 'trained_models' directory")
    print("Plots saved in 'plots' directory")

if __name__ == "__main__":
    main() 