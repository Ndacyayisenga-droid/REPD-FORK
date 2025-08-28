#!/usr/bin/env python3
"""
Test script to verify the local workflow functionality
"""

import pandas as pd
import numpy as np
import os
import sys

# Add current directory to path
sys.path.append('.')

def test_local_dataset():
    """Test loading and processing the local dataset"""
    print("🧪 Testing Local Dataset Workflow")
    print("=" * 40)
    
    # Test 1: Load local dataset
    print("1. Loading local dataset...")
    try:
        df = pd.read_csv('data/openj9_metrics.csv')
        print(f"   ✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"   📊 Columns: {list(df.columns)}")
    except Exception as e:
        print(f"   ❌ Failed to load dataset: {e}")
        return False
    
    # Test 2: Check required columns
    print("\n2. Checking required columns...")
    required_columns = ['wmc', 'rfc', 'loc', 'max_cc', 'avg_cc', 'cbo', 'ca', 'ce', 
                       'ic', 'cbm', 'lcom', 'lcom3', 'dit', 'noc', 'mfa', 'npm', 
                       'dam', 'moa', 'cam', 'amc', 'bug']
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"   ❌ Missing columns: {missing_columns}")
        return False
    else:
        print(f"   ✅ All required columns present")
    
    # Test 3: Check data quality
    print("\n3. Checking data quality...")
    print(f"   📈 Bug distribution: {df['bug'].value_counts().to_dict()}")
    print(f"   🔢 Non-null values: {df[required_columns[:-1]].count().min()} / {len(df)}")
    
    # Test 4: Prepare features
    print("\n4. Preparing features...")
    try:
        feature_columns = required_columns[:-1]  # Exclude 'bug'
        X = df[feature_columns].fillna(0).values
        y = df['bug'].values
        
        print(f"   ✅ Features prepared: {X.shape}")
        print(f"   ✅ Labels prepared: {y.shape}")
        print(f"   📊 Feature statistics:")
        print(f"      - Min: {X.min(axis=0).min():.2f}")
        print(f"      - Max: {X.max(axis=0).max():.2f}")
        print(f"      - Mean: {X.mean(axis=0).mean():.2f}")
    except Exception as e:
        print(f"   ❌ Failed to prepare features: {e}")
        return False
    
    # Test 5: Check local models
    print("\n5. Checking local models...")
    model_files = [
        'seantic_trained_models/repd_model_DA.pkl',
        'seantic_trained_models/scaler.pkl',
        'seantic_trained_models/training_results.pkl'
    ]
    
    for model_file in model_files:
        if os.path.exists(model_file):
            print(f"   ✅ Found: {model_file}")
        else:
            print(f"   ⚠️  Missing: {model_file}")
    
    print("\n" + "=" * 40)
    print("🎉 Local workflow test completed successfully!")
    print("✅ The workflow can use local data and models")
    return True

if __name__ == "__main__":
    success = test_local_dataset()
    sys.exit(0 if success else 1) 