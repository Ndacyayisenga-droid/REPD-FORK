# GlitchWitcher Semantic Workflow

## Overview

The GlitchWitcher Semantic Workflow is an advanced bug prediction system that uses **semantic analysis** of Java source code to predict potential bugs in pull requests. Unlike traditional metrics-based approaches, this workflow extracts **semantic features** from Java Abstract Syntax Trees (AST) and uses deep learning models to provide more accurate bug predictions.

## Key Features

- **Semantic Analysis**: Extracts features from Java AST including method complexity, coupling, cohesion, and inheritance metrics
- **Deep Learning Models**: Uses autoencoders and REPD (Reconstruction Error-based Probability Distribution) models
- **Java-Specific**: Optimized for Java code analysis with comprehensive metric extraction
- **GitHub Integration**: Seamless integration with GitHub Actions for automated analysis
- **Real-time Analysis**: Analyzes pull requests in real-time and provides detailed reports

## Semantic Metrics Extracted

The workflow extracts the following semantic metrics from Java code:

### Complexity Metrics
- **WMC (Weighted Methods per Class)**: Number of methods in a class
- **RFC (Response for Class)**: Number of methods that can be executed in response to a message
- **LOC (Lines of Code)**: Total lines of code in the class
- **Max CC (Maximum Cyclomatic Complexity)**: Highest cyclomatic complexity among methods
- **Avg CC (Average Cyclomatic Complexity)**: Average cyclomatic complexity of methods

### Coupling Metrics
- **CBO (Coupling Between Objects)**: Number of classes coupled to a given class
- **CA (Afferent Coupling)**: Number of classes that depend on this class
- **CE (Efferent Coupling)**: Number of classes this class depends on
- **CBM (Coupling Between Methods)**: Number of intra-class method calls

### Cohesion Metrics
- **LCOM (Lack of Cohesion of Methods)**: Measure of class cohesion
- **LCOM3**: Normalized version of LCOM
- **CAM (Cohesion Among Methods)**: Method parameter similarity

### Inheritance Metrics
- **DIT (Depth of Inheritance Tree)**: Maximum inheritance depth
- **NOC (Number of Children)**: Number of direct subclasses
- **MFA (Measure of Functional Abstraction)**: Ratio of inherited methods

### Encapsulation Metrics
- **NPM (Number of Public Methods)**: Count of public methods
- **DAM (Data Access Metric)**: Ratio of private/protected attributes
- **MOA (Measure of Aggregation)**: Count of aggregation relationships

### Maintainability Metrics
- **AMC (Average Method Complexity)**: Average complexity per method
- **IC (Inheritance Coupling)**: Inheritance depth

## Workflow Architecture

```
GitHub PR Comment → Parse Command → Check Dataset → Generate/Train Model → Analyze Changes → Report Results
```

### 1. Command Parsing
- Triggers on `GlitchWitcher-Semantic` comment
- Extracts repository and PR information
- Validates Java files in the PR

### 2. Dataset Management
- Checks for existing semantic dataset
- Generates new dataset if missing using AST analysis
- Stores datasets in `aqa-triage-data` repository

### 3. Model Training
- Uses autoencoder for dimensionality reduction
- Trains REPD model for bug prediction
- Saves trained models for reuse

### 4. Semantic Analysis
- Extracts semantic features from changed Java files
- Compares base and head commits
- Calculates probability densities for bug prediction

### 5. Reporting
- Generates detailed analysis report
- Shows risk changes between commits
- Provides interpretation guidance

## Usage

### Triggering the Workflow

1. **On a Pull Request**: Simply comment `GlitchWitcher-Semantic` on any PR containing Java files
2. **With Specific PR**: Comment `GlitchWitcher-Semantic https://github.com/owner/repo/pull/123`

### Example Output

```
## 🔮 GlitchWitcher Semantic Analysis Results
**Target PR:** https://github.com/example/repo/pull/123
**Repository:** example-repo

### 📊 **Semantic Analysis Results**

| File | Base (Non-Defective) | Base (Defective) | Head (Non-Defective) | Head (Defective) | Risk Change |
|------|---------------------|------------------|---------------------|------------------|-------------|
| TestClass.java | 0.000123 | 0.000045 | 0.000098 | 0.000067 | ⬆️ Increased |

### 📋 Semantic Analysis Interpretation:
> This analysis uses **semantic features** extracted from Java AST (Abstract Syntax Tree) including:
> - **WMC (Weighted Methods per Class)**: Complexity of class methods
> - **RFC (Response for Class)**: Number of methods that can be executed
> - **LCOM (Lack of Cohesion of Methods)**: Measure of class cohesion
> - **CBO (Coupling Between Objects)**: Degree of coupling between classes
> - **DIT (Depth of Inheritance Tree)**: Inheritance depth
> - **NOC (Number of Children)**: Number of direct subclasses
> - **CAM (Cohesion Among Methods)**: Method parameter similarity

> The values shown are **Probability Densities (PDFs)**, not probabilities. Higher values indicate better fit for that category.
```

## Technical Implementation

### Dependencies

```yaml
# Python Dependencies
tensorflow==2.12.0
pandas
joblib
scipy
numpy
urllib3
scikit-learn
javalang
torch
keras

# System Dependencies
openjdk-11-jdk
git
cloc
```

### Key Components

1. **AST Parser**: Uses `javalang` library for Java AST parsing
2. **AutoEncoder**: TensorFlow 2.x compatible implementation for feature extraction
3. **REPD Model**: Reconstruction Error-based Probability Distribution for bug prediction
4. **Metrics Extractor**: Comprehensive semantic metrics calculation
5. **Git Integration**: Automated repository cloning and diff analysis

### File Structure

```
REPD/
├── .github/workflows/
│   └── glitchwitcher-semantic.yml    # Main workflow file
├── autoencoder_tf2.py                # TensorFlow 2.x autoencoder
├── REPD_Impl.py                      # REPD model implementation
├── analyze_semantic_metrics.sh       # Metrics extraction script
├── test_semantic_workflow.py         # Test suite
└── semantic-dataset-creation/        # Semantic analysis components
    ├── extractor.py                  # Feature extractors
    ├── ASTEncoder-v1.2.jar          # AST encoder
    └── config/                       # Configuration files
```

## Configuration

### Workflow Configuration

The workflow is configured in `.github/workflows/glitchwitcher-semantic.yml`:

- **Triggers**: `issue_comment` with `GlitchWitcher-Semantic`
- **Environment**: Ubuntu latest with Python 3.10
- **Permissions**: Pull requests, issues, and contents read access

### Model Configuration

- **AutoEncoder Layers**: Dynamic based on feature count
- **Training Epochs**: 100-200 epochs for convergence
- **Batch Size**: 32-512 depending on dataset size
- **Learning Rate**: 0.01 with Adam optimizer

## Testing

### Running Tests

```bash
# Run the complete test suite
python3 test_semantic_workflow.py

# Test individual components
python3 -c "from autoencoder_tf2 import AutoEncoder; print('AutoEncoder OK')"
python3 -c "from REPD_Impl import REPD; print('REPD OK')"
```

### Test Coverage

- ✅ JavaLang parsing
- ✅ Metrics extraction
- ✅ AutoEncoder training
- ✅ REPD model training
- ✅ CSV generation
- ✅ End-to-end workflow

## Comparison with Traditional Workflow

| Aspect | Traditional Workflow | Semantic Workflow |
|--------|---------------------|-------------------|
| **Language Support** | C/C++ | Java |
| **Feature Extraction** | Basic metrics | AST-based semantic features |
| **Model Type** | Traditional ML | Deep Learning (AutoEncoder + REPD) |
| **Analysis Depth** | Surface-level | Semantic understanding |
| **Accuracy** | Good | Better (semantic context) |
| **Complexity** | Lower | Higher (more sophisticated) |

## Troubleshooting

### Common Issues

1. **TensorFlow Compatibility**: Ensure using TensorFlow 2.x compatible code
2. **Java Parsing Errors**: Check for malformed Java code or unsupported syntax
3. **Memory Issues**: Reduce batch size for large datasets
4. **Model Training Failures**: Check dataset quality and feature scaling

### Debug Mode

Enable debug output by setting environment variables:
```bash
export GLITCHWITCHER_DEBUG=1
export TENSORFLOW_VERBOSE=1
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is part of the GlitchWitcher initiative and follows the same licensing terms as the main project.

## Support

For issues and questions:
- Create an issue in the repository
- Check the troubleshooting section
- Review the test suite for examples

---

**Note**: This semantic workflow provides more sophisticated analysis than the traditional workflow by leveraging deep learning and semantic understanding of Java code structure. 