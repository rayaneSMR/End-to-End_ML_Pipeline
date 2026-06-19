# End-to-End ML Pipeline: Telco Customer Churn Prediction


A production-ready machine learning pipeline that predicts customer churn for a telecom company using scikit-learn pipelines, hyperparameter tuning, and a Streamlit web interface.

---

## Problem Statement

Build a reusable, production-ready ML pipeline that predicts **customer churn** for a telecom company using the Telco Customer Churn dataset. The system identifies which customers are likely to leave the service so the company can take proactive retention measures.

---

## Features

### Core ML Pipeline
- **Dual Model Architecture**: Logistic Regression and Random Forest classifiers
- **Sklearn Pipelines**: Automated preprocessing with zero data leakage
- **Feature Engineering**: Automatic handling of numeric and categorical features
- **Hyperparameter Tuning**: GridSearchCV optimization for both models
- **Cross-Validation**: 5-fold cross-validation for robust evaluation

### Streamlit Web Interface
- **Interactive Prediction Page**: Real-time churn predictions with probability scores
- **Model Information Page**: Detailed model architecture and feature documentation
- **Model Selector**: Compare predictions between Logistic Regression and Random Forest
- **Visual Metrics**: Color-coded alerts (green/red) and confidence scores
- **Professional UI**: Responsive design with custom styling

---

## Dataset

**Telco Customer Churn Dataset**
- **Size**: 7,043 customer records
- **Features**: 20 features (7 numeric + 13 categorical)
- **Target**: Binary churn classification (Yes/No)
- **Churn Rate**: ~26.5%
- **Source**: IBM's public GitHub repository
- **Link**: [Telco-Customer-Churn.csv](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv)

### Features Used

**Numeric Features (7)**
- tenure: Customer tenure in months
- MonthlyCharges: Monthly billing amount
- TotalCharges: Total charges to customer
- SeniorCitizen: Senior citizen status (0/1)

**Categorical Features (13)**
- gender, Partner, Dependents
- PhoneService, MultipleLines
- InternetService, OnlineSecurity, OnlineBackup
- DeviceProtection, TechSupport
- StreamingTV, StreamingMovies
- Contract, PaperlessBilling, PaymentMethod

---

## Models

### 1. Logistic Regression
- **Type**: Linear classification
- **Pros**: Fast, interpretable, good baseline
- **Use Case**: Understanding feature importance and relationships
- **Training Time**: ~2 seconds

### 2. Random Forest
- **Type**: Ensemble of decision trees
- **Pros**: Captures non-linear patterns, robust, excellent generalization
- **Use Case**: Production predictions with higher accuracy
- **Training Time**: ~15 seconds
- **Parameters**: 100 estimators, max_depth optimized via GridSearchCV

---

## Project Structure

```
task2/
├── README.md                          # Project documentation
├── Task2_Churn_Pipeline.ipynb        # Jupyter notebook with full analysis
├── streamlit_app.py                  # Streamlit web application
├── churn_model_lr.pkl                # Saved Logistic Regression model
├── churn_model_rf.pkl                # Saved Random Forest model
└── requirements.txt                  # Python dependencies
```

---

## Quick Start

### Prerequisites
- Python 3.8+
- pip or conda

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/rayaneSMR/End-to-End_ML_Pipeline.git
cd End-to-End_ML_Pipeline
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Running the Application

**Start the Streamlit web interface:**
```bash
streamlit run streamlit_app.py
```

Then open your browser and navigate to:
```
http://localhost:8501
```

---

## Usage

### Web Interface - Prediction Page

1. **Select Model**: Choose between "Logistic Regression" or "Random Forest" from sidebar
2. **Enter Customer Info**: Fill in the customer details using the form inputs
   - Demographics: Gender, Age, Partner status, Dependents
   - Services: Internet, Phone, Streaming, Security, etc.
   - Charges: Monthly and Total charges
   - Contract: Contract type and payment method
3. **Click "Predict Churn"**: Get instant prediction with probability score
4. **View Results**: See color-coded alert (green/red) and confidence metric

### Web Interface - Model Info Page

- Compare Logistic Regression vs Random Forest characteristics
- View complete feature list
- See dataset statistics and model architecture details

### Jupyter Notebook Analysis

Run the notebook for complete pipeline walkthrough:
```bash
jupyter notebook Task2_Churn_Pipeline.ipynb
```

**Notebook Sections:**
1. Dataset Loading & EDA
2. Feature Engineering & Train/Test Split
3. Pipeline Architecture
4. Baseline Model Training
5. Hyperparameter Tuning with GridSearchCV
6. Final Model Evaluation
7. ROC Curve and Confusion Matrix Analysis

---

## Model Performance

### Metrics Evaluated
- **Accuracy**: Overall prediction correctness
- **F1-Score**: Balance between precision and recall
- **ROC-AUC**: Discrimination ability across thresholds
- **Cross-Validation**: 5-fold CV scores for robustness

### Example Results (on test set)

| Model | Accuracy | F1-Score | ROC-AUC | CV F1 (5-fold) |
|-------|----------|----------|---------|----------------|
| Logistic Regression | ~0.80 | ~0.64 | ~0.84 | ~0.64 |
| Random Forest | ~0.82 | ~0.67 | ~0.86 | ~0.67 |

*Note: Actual results depend on dataset version and hyperparameter tuning*

---

## Technical Stack

### Libraries
- **scikit-learn**: ML models, pipelines, preprocessing
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **streamlit**: Web interface framework
- **joblib**: Model serialization
- **matplotlib & seaborn**: Data visualization

### Machine Learning Pipeline

```
Input Data
    ↓
[Preprocessing]
    ├─ Numeric: Imputation (median) → Scaling (StandardScaler)
    └─ Categorical: Imputation (mode) → Encoding (OneHotEncoder)
    ↓
[Feature Combination]
    ↓
[Classification Model]
    ├─ Logistic Regression
    └─ Random Forest
    ↓
Output: Churn Prediction & Probability
```

---

## Key Implementation Details

### Data Leakage Prevention
- All preprocessing steps are inside the pipeline
- Scaler fitted only on training data
- Categorical encoder fitted only on training data

### Hyperparameter Tuning
- **GridSearchCV** with 5-fold cross-validation
- **Scoring Metric**: F1-score (better for imbalanced classes)
- **Parallel Processing**: n_jobs=-1 for faster computation

### Model Persistence
- Models saved as pickle files (.pkl)
- Automatic loading on app startup
- Retraining if saved models not found

---

## Evaluation Reports

### Classification Report
```
              precision    recall  f1-score   support

    No Churn       0.85      0.93      0.89      1127
      Churn        0.72      0.53      0.61       343

    accuracy                           0.82      1470
   macro avg       0.79      0.73      0.75      1470
weighted avg       0.82      0.82      0.81      1470
```

### Confusion Matrix
```
               Predicted
              No Churn | Churn
Actual No Churn   1050  |  77
       Churn       161  | 182
```

---

## Learning Outcomes

This project demonstrates:
- End-to-end ML pipeline development
- Scikit-learn Pipeline API for production ML
- Hyperparameter optimization with GridSearchCV
- Handling mixed data types (numeric + categorical)
- Model evaluation and comparison
- Web application development with Streamlit
- Version control with Git/GitHub

---

## Troubleshooting

### Streamlit app not starting
```bash
# Clear Streamlit cache
streamlit cache clear

# Run with verbose output
streamlit run streamlit_app.py --logger.level=debug
```

### Models not training
- Ensure internet connection for dataset download
- If URL fails, synthetic dataset will be generated
- Check available memory (training uses ~2-3GB)

### Port 8501 already in use
```bash
streamlit run streamlit_app.py --server.port 8502
```

---

## Files Description

| File | Purpose |
|------|---------|
| `Task2_Churn_Pipeline.ipynb` | Complete ML pipeline with analysis and visualization |
| `streamlit_app.py` | Web interface for predictions and model comparison |
| `churn_model_lr.pkl` | Trained Logistic Regression pipeline |
| `churn_model_rf.pkl` | Trained Random Forest pipeline |
| `README.md` | This file - project documentation |

---

## GitHub Repository

**Repository**: [End-to-End_ML_Pipeline](https://github.com/rayaneSMR/End-to-End_ML_Pipeline.git)

---

## Author

**Smara** - DevelopersHub Corporation AI/ML Engineering Internship

---

## License

This project is open source and available under the MIT License.

---

## Acknowledgments

- **Dataset Source**: IBM Telco Customer Churn Dataset
- **Framework**: Scikit-learn & Streamlit communities
- **Inspiration**: DevelopersHub ML Engineering Advanced Program

---

## Support & Contact

For issues, questions, or improvements:
1. Check the troubleshooting section
2. Review the Jupyter notebook for detailed explanations
3. Open an issue on GitHub

---

## Future Enhancements

- [ ] Add more classification models (XGBoost, LightGBM, SVM)
- [ ] Feature importance visualization
- [ ] Model comparison dashboard
- [ ] SHAP value explainability
- [ ] Batch prediction upload
- [ ] API deployment with FastAPI
- [ ] Docker containerization
- [ ] Real-time monitoring dashboard

---

**Last Updated**: June 19, 2026

**Status**: Production Ready
