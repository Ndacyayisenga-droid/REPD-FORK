#!/usr/bin/env python3
"""
Test script for GlitchWitcher Semantic Workflow
This script tests the semantic analysis components to ensure they work correctly.
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import javalang
import tempfile
import subprocess
from collections import defaultdict
import csv
import re

# Add current directory to path to import REPD modules
sys.path.append('.')

try:
    from REPD_Impl import REPD
    from autoencoder_tf2 import AutoEncoder
    print("✅ Successfully imported REPD and AutoEncoder modules")
except ImportError as e:
    print(f"❌ Failed to import REPD modules: {e}")
    sys.exit(1)

def test_javalang_parsing():
    """Test javalang parsing with a simple Java class"""
    print("\n🔍 Testing javalang parsing...")
    
    java_code = """
package com.example;

public class TestClass {
    private int field1;
    public String field2;
    
    public TestClass() {
        this.field1 = 0;
    }
    
    public int getField1() {
        return field1;
    }
    
    public void setField1(int value) {
        if (value > 0) {
            this.field1 = value;
        }
    }
    
    public void complexMethod() {
        for (int i = 0; i < 10; i++) {
            if (i % 2 == 0) {
                System.out.println("Even: " + i);
            } else {
                System.out.println("Odd: " + i);
            }
        }
    }
}
"""
    
    try:
        tree = javalang.parse.parse(java_code)
        print("✅ Successfully parsed Java code")
        
        # Count classes
        classes = list(tree.filter(javalang.tree.ClassDeclaration))
        print(f"✅ Found {len(classes)} class(es)")
        
        # Count methods
        methods = list(classes[0].methods) if hasattr(classes[0], 'methods') else []
        print(f"✅ Found {len(methods)} method(s)")
        
        return True
    except Exception as e:
        print(f"❌ Failed to parse Java code: {e}")
        return False

def test_metrics_extraction():
    """Test metrics extraction from Java code"""
    print("\n📊 Testing metrics extraction...")
    
    java_code = """
package com.example;

public class TestClass {
    private int field1;
    public String field2;
    
    public TestClass() {
        this.field1 = 0;
    }
    
    public int getField1() {
        return field1;
    }
    
    public void setField1(int value) {
        if (value > 0) {
            this.field1 = value;
        }
    }
}
"""
    
    try:
        tree = javalang.parse.parse(java_code)
        
        for _, class_node in tree.filter(javalang.tree.ClassDeclaration):
            # Basic metrics
            methods = class_node.methods
            fields = [f for f in class_node.fields if isinstance(f, javalang.tree.FieldDeclaration)]
            
            metrics = {
                'wmc': len(methods),
                'loc': len(java_code.splitlines()),
                'npm': sum(1 for m in methods if m.modifiers and 'public' in m.modifiers),
                'dit': 1 if class_node.extends else 0,
                'moa': sum(1 for f in fields if f.type and isinstance(f.type, javalang.tree.ReferenceType))
            }
            
            print(f"✅ Extracted metrics: {metrics}")
            return True
            
    except Exception as e:
        print(f"❌ Failed to extract metrics: {e}")
        return False

def test_autoencoder():
    """Test autoencoder creation and training"""
    print("\n🤖 Testing AutoEncoder...")
    
    try:
        # Create sample data
        X = np.random.rand(100, 20)
        
        # Create autoencoder
        layers = [20, 15, 10, 5]
        autoencoder = AutoEncoder(layers, lr=0.01, epoch=5, batch_size=10)
        
        # Train autoencoder
        autoencoder.fit(X, print_progress=False)
        
        # Test transform
        encoded = autoencoder.transform(X)
        decoded = autoencoder.inverse_transform(encoded)
        
        print(f"✅ AutoEncoder created and trained successfully")
        print(f"   Input shape: {X.shape}")
        print(f"   Encoded shape: {encoded.shape}")
        print(f"   Decoded shape: {decoded.shape}")
        
        # Clean up
        autoencoder.close()
        
        return True
    except Exception as e:
        print(f"❌ AutoEncoder test failed: {e}")
        return False

def test_repd_model():
    """Test REPD model creation and training"""
    print("\n🧠 Testing REPD model...")
    
    try:
        # Create sample data
        X = np.random.rand(100, 20)
        y = np.random.randint(0, 2, 100)
        
        # Create autoencoder
        layers = [20, 15, 10, 5]
        autoencoder = AutoEncoder(layers, lr=0.01, epoch=5, batch_size=10)
        
        # Create REPD model
        repd_model = REPD(autoencoder)
        
        # Train REPD model
        repd_model.fit(X, y)
        
        # Test prediction
        predictions = repd_model.predict(X)
        
        print(f"✅ REPD model created and trained successfully")
        print(f"   Input shape: {X.shape}")
        print(f"   Predictions shape: {predictions.shape}")
        print(f"   Unique predictions: {np.unique(predictions)}")
        
        # Clean up
        autoencoder.close()
        
        return True
    except Exception as e:
        print(f"❌ REPD model test failed: {e}")
        return False

def test_csv_generation():
    """Test CSV generation with semantic metrics"""
    print("\n📄 Testing CSV generation...")
    
    try:
        # Create sample data
        data = {
            'project_name': ['test-project'] * 5,
            'version': ['1.0'] * 5,
            'class_name': [f'TestClass{i}' for i in range(5)],
            'wmc': [3, 5, 2, 7, 4],
            'rfc': [8, 12, 5, 15, 10],
            'loc': [50, 80, 30, 120, 60],
            'max_cc': [2, 4, 1, 6, 3],
            'avg_cc': [1.5, 2.5, 1.0, 3.0, 2.0],
            'cbo': [2, 4, 1, 5, 3],
            'ca': [1, 2, 0, 3, 1],
            'ce': [1, 2, 1, 2, 2],
            'ic': [0, 1, 0, 1, 0],
            'cbm': [2, 3, 1, 4, 2],
            'lcom': [1, 3, 0, 5, 2],
            'lcom3': [0.5, 1.0, 0.0, 1.67, 0.67],
            'dit': [0, 1, 0, 2, 1],
            'noc': [0, 1, 0, 2, 0],
            'mfa': [0.0, 0.2, 0.0, 0.4, 0.1],
            'npm': [2, 3, 1, 4, 2],
            'dam': [0.5, 0.7, 0.3, 0.8, 0.6],
            'moa': [1, 2, 0, 3, 1],
            'cam': [0.3, 0.5, 0.2, 0.6, 0.4],
            'amc': [16.7, 16.0, 15.0, 17.1, 15.0],
            'bug': [0, 1, 0, 2, 1]
        }
        
        df = pd.DataFrame(data)
        
        # Save to CSV
        csv_file = 'test_semantic_metrics.csv'
        df.to_csv(csv_file, index=False)
        
        # Read back and verify
        df_read = pd.read_csv(csv_file)
        
        print(f"✅ CSV generation successful")
        print(f"   Original shape: {df.shape}")
        print(f"   Read shape: {df_read.shape}")
        print(f"   Columns: {list(df.columns)}")
        
        # Clean up
        os.remove(csv_file)
        
        return True
    except Exception as e:
        print(f"❌ CSV generation test failed: {e}")
        return False

def test_end_to_end_workflow():
    """Test the complete semantic analysis workflow"""
    print("\n🔄 Testing end-to-end workflow...")
    
    try:
        # Create sample dataset
        data = {
            'project_name': ['test-project'] * 10,
            'version': ['1.0'] * 10,
            'class_name': [f'TestClass{i}' for i in range(10)],
            'wmc': [3, 5, 2, 7, 4, 6, 3, 8, 5, 4],
            'rfc': [8, 12, 5, 15, 10, 13, 7, 16, 11, 9],
            'loc': [50, 80, 30, 120, 60, 90, 40, 150, 70, 55],
            'max_cc': [2, 4, 1, 6, 3, 5, 2, 7, 4, 3],
            'avg_cc': [1.5, 2.5, 1.0, 3.0, 2.0, 2.8, 1.8, 3.5, 2.2, 1.9],
            'cbo': [2, 4, 1, 5, 3, 4, 2, 6, 3, 2],
            'ca': [1, 2, 0, 3, 1, 2, 1, 4, 2, 1],
            'ce': [1, 2, 1, 2, 2, 3, 1, 3, 2, 1],
            'ic': [0, 1, 0, 1, 0, 1, 0, 2, 1, 0],
            'cbm': [2, 3, 1, 4, 2, 3, 2, 5, 3, 2],
            'lcom': [1, 3, 0, 5, 2, 4, 1, 6, 3, 2],
            'lcom3': [0.5, 1.0, 0.0, 1.67, 0.67, 1.33, 0.5, 2.0, 1.0, 0.67],
            'dit': [0, 1, 0, 2, 1, 1, 0, 2, 1, 0],
            'noc': [0, 1, 0, 2, 0, 1, 0, 3, 1, 0],
            'mfa': [0.0, 0.2, 0.0, 0.4, 0.1, 0.3, 0.0, 0.5, 0.2, 0.1],
            'npm': [2, 3, 1, 4, 2, 3, 2, 5, 3, 2],
            'dam': [0.5, 0.7, 0.3, 0.8, 0.6, 0.7, 0.4, 0.9, 0.6, 0.5],
            'moa': [1, 2, 0, 3, 1, 2, 1, 4, 2, 1],
            'cam': [0.3, 0.5, 0.2, 0.6, 0.4, 0.5, 0.3, 0.7, 0.4, 0.3],
            'amc': [16.7, 16.0, 15.0, 17.1, 15.0, 15.0, 13.3, 18.8, 14.0, 13.8],
            'bug': [0, 1, 0, 2, 1, 1, 0, 3, 1, 0]
        }
        
        df = pd.DataFrame(data)
        
        # Prepare features
        feature_columns = ['wmc', 'rfc', 'loc', 'max_cc', 'avg_cc', 'cbo', 'ca', 'ce', 
                          'ic', 'cbm', 'lcom', 'lcom3', 'dit', 'noc', 'mfa', 'npm', 
                          'dam', 'moa', 'cam', 'amc']
        
        X = df[feature_columns].fillna(0).values
        y = df['bug'].values
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Create and train autoencoder
        input_dim = X_scaled.shape[1]
        layers = [input_dim, max(50, input_dim//2), max(25, input_dim//4), max(10, input_dim//8)]
        autoencoder = AutoEncoder(layers, lr=0.01, epoch=10, batch_size=5)
        autoencoder.fit(X_scaled, print_progress=False)
        
        # Create and train REPD model
        repd_model = REPD(autoencoder)
        repd_model.fit(X_scaled, y)
        
        # Test predictions
        predictions = repd_model.predict(X_scaled)
        test_errors = repd_model.calculate_reconstruction_error(X_scaled)
        p_nd = repd_model.get_non_defect_probability(test_errors)
        p_d = repd_model.get_defect_probability(test_errors)
        
        print(f"✅ End-to-end workflow successful")
        print(f"   Dataset shape: {df.shape}")
        print(f"   Feature columns: {len(feature_columns)}")
        print(f"   Predictions: {np.unique(predictions)}")
        print(f"   Probability densities calculated: {len(p_nd)}")
        
        # Clean up
        autoencoder.close()
        
        return True
    except Exception as e:
        print(f"❌ End-to-end workflow test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 GlitchWitcher Semantic Workflow Test Suite")
    print("=" * 50)
    
    tests = [
        ("JavaLang Parsing", test_javalang_parsing),
        ("Metrics Extraction", test_metrics_extraction),
        ("AutoEncoder", test_autoencoder),
        ("REPD Model", test_repd_model),
        ("CSV Generation", test_csv_generation),
        ("End-to-End Workflow", test_end_to_end_workflow)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The semantic workflow should work correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 