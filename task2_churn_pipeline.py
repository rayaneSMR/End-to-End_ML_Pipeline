# =============================================================================
# Task 2: End-to-End ML Pipeline with Scikit-learn Pipeline API
# Objective: Build a reusable and production-ready ML pipeline for predicting
#            customer churn using the Telco Churn Dataset.
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, roc_auc_score, roc_curve
)


# =============================================================================
# STEP 1: Load & Explore the Dataset
# =============================================================================
print("=" * 60)
print("STEP 1: Loading and Exploring the Telco Churn Dataset")
print("=" * 60)

# Download dataset from the IBM/Kaggle Telco Churn source
url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"

try:
    df = pd.read_csv(url)
    print(f"✅ Dataset loaded from URL. Shape: {df.shape}")
except Exception:
    # Fallback: generate a realistic synthetic Telco Churn dataset
    print("⚠️  URL not reachable. Generating synthetic Telco Churn dataset...")
    np.random.seed(42)
    n = 7043
    df = pd.DataFrame({
        'customerID': [f'ID-{i:05d}' for i in range(n)],
        'gender': np.random.choice(['Male', 'Female'], n),
        'SeniorCitizen': np.random.choice([0, 1], n, p=[0.84, 0.16]),
        'Partner': np.random.choice(['Yes', 'No'], n),
        'Dependents': np.random.choice(['Yes', 'No'], n, p=[0.3, 0.7]),
        'tenure': np.random.randint(0, 72, n),
        'PhoneService': np.random.choice(['Yes', 'No'], n, p=[0.9, 0.1]),
        'MultipleLines': np.random.choice(['Yes', 'No', 'No phone service'], n),
        'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n, p=[0.34, 0.44, 0.22]),
        'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], n),
        'OnlineBackup': np.random.choice(['Yes', 'No', 'No internet service'], n),
        'DeviceProtection': np.random.choice(['Yes', 'No', 'No internet service'], n),
        'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], n),
        'StreamingTV': np.random.choice(['Yes', 'No', 'No internet service'], n),
        'StreamingMovies': np.random.choice(['Yes', 'No', 'No internet service'], n),
        'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n, p=[0.55, 0.24, 0.21]),
        'PaperlessBilling': np.random.choice(['Yes', 'No'], n, p=[0.59, 0.41]),
        'PaymentMethod': np.random.choice([
            'Electronic check', 'Mailed check',
            'Bank transfer (automatic)', 'Credit card (automatic)'
        ], n),
        'MonthlyCharges': np.round(np.random.uniform(18, 119, n), 2),
        'TotalCharges': np.round(np.random.uniform(18, 8500, n), 2),
        'Churn': np.random.choice(['Yes', 'No'], n, p=[0.265, 0.735])
    })
    # Inject some missing values (realistic)
    missing_idx = np.random.choice(n, 11, replace=False)
    df.loc[missing_idx, 'TotalCharges'] = np.nan
    print(f"✅ Synthetic dataset generated. Shape: {df.shape}")

print(f"\nDataset Info:")
print(f"  Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print(f"\nFirst 3 rows:")
print(df.head(3).to_string())
print(f"\nMissing Values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
print(f"\nChurn Distribution:\n{df['Churn'].value_counts()}")


# =============================================================================
# STEP 2: Preprocessing & Feature Engineering
# =============================================================================
print("\n" + "=" * 60)
print("STEP 2: Preprocessing & Feature Engineering")
print("=" * 60)

# Drop customerID (not predictive)
df.drop(columns=['customerID'], inplace=True)

# Convert TotalCharges to numeric (some entries may be blank strings)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Encode target variable
df['Churn'] = (df['Churn'] == 'Yes').astype(int)

# Separate features and target
X = df.drop(columns=['Churn'])
y = df['Churn']

# Identify column types
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

print(f"Numeric features  ({len(numeric_features)}): {numeric_features}")
print(f"Categorical features ({len(categorical_features)}): {categorical_features}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
print(f"Churn rate (train): {y_train.mean():.2%}, Churn rate (test): {y_test.mean():.2%}")


# =============================================================================
# STEP 3: Build Scikit-learn Pipelines
# =============================================================================
print("\n" + "=" * 60)
print("STEP 3: Building Scikit-learn Pipelines")
print("=" * 60)

# --- Sub-pipelines for preprocessing ---
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# --- ColumnTransformer ---
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# --- Full pipelines ---
lr_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42, max_iter=1000))
])

rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42, n_jobs=-1))
])

print("✅ Logistic Regression pipeline created")
print("✅ Random Forest pipeline created")


# =============================================================================
# STEP 4: Baseline Training & Evaluation
# =============================================================================
print("\n" + "=" * 60)
print("STEP 4: Baseline Training & Evaluation")
print("=" * 60)

def evaluate_model(name, pipeline, X_tr, y_tr, X_te, y_te):
    pipeline.fit(X_tr, y_tr)
    y_pred = pipeline.predict(X_te)
    y_prob = pipeline.predict_proba(X_te)[:, 1]
    acc   = accuracy_score(y_te, y_pred)
    f1    = f1_score(y_te, y_pred)
    auc   = roc_auc_score(y_te, y_prob)
    cv    = cross_val_score(pipeline, X_tr, y_tr, cv=5, scoring='f1', n_jobs=-1).mean()
    print(f"\n{'─'*40}")
    print(f"  Model: {name}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"  ROC-AUC   : {auc:.4f}")
    print(f"  CV F1 (5-fold): {cv:.4f}")
    print(f"\n  Classification Report:\n{classification_report(y_te, y_pred, target_names=['No Churn','Churn'])}")
    return acc, f1, auc

lr_acc, lr_f1, lr_auc = evaluate_model("Logistic Regression", lr_pipeline, X_train, y_train, X_test, y_test)
rf_acc, rf_f1, rf_auc = evaluate_model("Random Forest (baseline)", rf_pipeline, X_train, y_train, X_test, y_test)


# =============================================================================
# STEP 5: Hyperparameter Tuning with GridSearchCV
# =============================================================================
print("\n" + "=" * 60)
print("STEP 5: Hyperparameter Tuning with GridSearchCV")
print("=" * 60)

# --- Tune Logistic Regression ---
lr_param_grid = {
    'classifier__C': [0.01, 0.1, 1, 10],
    'classifier__solver': ['lbfgs', 'liblinear'],
    'classifier__class_weight': [None, 'balanced']
}

lr_grid = GridSearchCV(
    lr_pipeline, lr_param_grid,
    cv=5, scoring='f1', n_jobs=-1, verbose=0
)
lr_grid.fit(X_train, y_train)
print(f"\nBest LR params : {lr_grid.best_params_}")
print(f"Best LR CV F1  : {lr_grid.best_score_:.4f}")

# --- Tune Random Forest ---
rf_param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5],
    'classifier__class_weight': [None, 'balanced']
}

rf_grid = GridSearchCV(
    rf_pipeline, rf_param_grid,
    cv=5, scoring='f1', n_jobs=-1, verbose=0
)
rf_grid.fit(X_train, y_train)
print(f"\nBest RF params : {rf_grid.best_params_}")
print(f"Best RF CV F1  : {rf_grid.best_score_:.4f}")


# =============================================================================
# STEP 6: Final Evaluation After Tuning
# =============================================================================
print("\n" + "=" * 60)
print("STEP 6: Final Evaluation After Tuning")
print("=" * 60)

best_lr = lr_grid.best_estimator_
best_rf = rf_grid.best_estimator_

lr_y_pred  = best_lr.predict(X_test)
lr_y_prob  = best_lr.predict_proba(X_test)[:, 1]
rf_y_pred  = best_rf.predict(X_test)
rf_y_prob  = best_rf.predict_proba(X_test)[:, 1]

results = {
    'Logistic Regression': {
        'Accuracy': accuracy_score(y_test, lr_y_pred),
        'F1-Score': f1_score(y_test, lr_y_pred),
        'ROC-AUC': roc_auc_score(y_test, lr_y_prob)
    },
    'Random Forest': {
        'Accuracy': accuracy_score(y_test, rf_y_pred),
        'F1-Score': f1_score(y_test, rf_y_pred),
        'ROC-AUC': roc_auc_score(y_test, rf_y_prob)
    }
}

print("\nFinal Tuned Results:")
results_df = pd.DataFrame(results).T
print(results_df.to_string())

# Pick the best model
best_model_name = results_df['F1-Score'].idxmax()
best_pipeline   = best_lr if best_model_name == 'Logistic Regression' else best_rf
print(f"\n🏆 Best model: {best_model_name}")


# =============================================================================
# STEP 7: Visualizations
# =============================================================================
print("\n" + "=" * 60)
print("STEP 7: Generating Visualizations")
print("=" * 60)

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle('Task 2: Customer Churn ML Pipeline – Results', fontsize=15, fontweight='bold')

# (A) Churn Distribution
ax = axes[0, 0]
churn_counts = y.value_counts()
ax.bar(['No Churn', 'Churn'], churn_counts.values,
       color=['#2ecc71', '#e74c3c'], edgecolor='white', linewidth=1.5)
for i, v in enumerate(churn_counts.values):
    ax.text(i, v + 30, f'{v}\n({v/len(y):.1%})', ha='center', fontsize=10, fontweight='bold')
ax.set_title('Churn Distribution', fontsize=12, fontweight='bold')
ax.set_ylabel('Count')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# (B) Confusion Matrix – Best Model
ax = axes[0, 1]
y_pred_best = best_pipeline.predict(X_test)
cm = confusion_matrix(y_test, y_pred_best)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['No Churn', 'Churn'],
            yticklabels=['No Churn', 'Churn'],
            linewidths=2)
ax.set_title(f'Confusion Matrix\n({best_model_name})', fontsize=12, fontweight='bold')
ax.set_ylabel('Actual')
ax.set_xlabel('Predicted')

# (C) ROC Curves
ax = axes[0, 2]
for name, prob, color in [
    ('Logistic Regression', lr_y_prob, '#3498db'),
    ('Random Forest', rf_y_prob, '#e74c3c')
]:
    fpr, tpr, _ = roc_curve(y_test, prob)
    auc = roc_auc_score(y_test, prob)
    ax.plot(fpr, tpr, color=color, lw=2, label=f'{name} (AUC={auc:.3f})')
ax.plot([0, 1], [0, 1], 'k--', lw=1)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves', fontsize=12, fontweight='bold')
ax.legend(loc='lower right', fontsize=9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# (D) Model Comparison Bar Chart
ax = axes[1, 0]
metrics = ['Accuracy', 'F1-Score', 'ROC-AUC']
x = np.arange(len(metrics))
width = 0.35
bars1 = ax.bar(x - width/2, [results['Logistic Regression'][m] for m in metrics],
               width, label='Logistic Regression', color='#3498db', alpha=0.85)
bars2 = ax.bar(x + width/2, [results['Random Forest'][m] for m in metrics],
               width, label='Random Forest', color='#e74c3c', alpha=0.85)
for bar in bars1 + bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_ylim(0, 1.05)
ax.set_title('Model Comparison', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# (E) Feature Importances (RF)
ax = axes[1, 1]
rf_clf = best_rf.named_steps['classifier']
ohe_cols = best_rf.named_steps['preprocessor']\
              .named_transformers_['cat']\
              .named_steps['onehot']\
              .get_feature_names_out(categorical_features)
all_features = numeric_features + list(ohe_cols)
importances = rf_clf.feature_importances_
top_idx = np.argsort(importances)[-15:]
ax.barh(np.array(all_features)[top_idx], importances[top_idx],
        color='#9b59b6', alpha=0.85, edgecolor='white')
ax.set_title('Top 15 Feature Importances\n(Random Forest)', fontsize=12, fontweight='bold')
ax.set_xlabel('Importance')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# (F) Monthly Charges Distribution by Churn
ax = axes[1, 2]
df_plot = df.copy()
df_plot['Churn_label'] = df_plot['Churn'].map({0: 'No Churn', 1: 'Churn'})
for label, color in [('No Churn', '#2ecc71'), ('Churn', '#e74c3c')]:
    vals = df_plot.loc[df_plot['Churn_label'] == label, 'MonthlyCharges']
    ax.hist(vals, bins=30, alpha=0.6, color=color, label=label, edgecolor='white')
ax.set_xlabel('Monthly Charges ($)')
ax.set_ylabel('Count')
ax.set_title('Monthly Charges by Churn Status', fontsize=12, fontweight='bold')
ax.legend()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/task2_results.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Visualizations saved to task2_results.png")


# =============================================================================
# STEP 8: Export the Pipeline with joblib
# =============================================================================
print("\n" + "=" * 60)
print("STEP 8: Exporting the Pipeline with joblib")
print("=" * 60)

joblib.dump(best_pipeline, '/mnt/user-data/outputs/churn_pipeline.pkl')
print(f"✅ Best pipeline ({best_model_name}) exported → churn_pipeline.pkl")

# --- Verify by reloading ---
loaded_pipeline = joblib.load('/mnt/user-data/outputs/churn_pipeline.pkl')
loaded_preds = loaded_pipeline.predict(X_test)
assert (loaded_preds == y_pred_best).all(), "❌ Reload mismatch!"
print("✅ Pipeline reload verified — predictions match.")


# =============================================================================
# STEP 9: Inference Demo (Production-style usage)
# =============================================================================
print("\n" + "=" * 60)
print("STEP 9: Inference Demo – Predicting on New Customers")
print("=" * 60)

new_customers = pd.DataFrame([
    {
        'gender': 'Female', 'SeniorCitizen': 0, 'Partner': 'Yes',
        'Dependents': 'No', 'tenure': 1, 'PhoneService': 'No',
        'MultipleLines': 'No phone service', 'InternetService': 'DSL',
        'OnlineSecurity': 'No', 'OnlineBackup': 'Yes',
        'DeviceProtection': 'No', 'TechSupport': 'No',
        'StreamingTV': 'No', 'StreamingMovies': 'No',
        'Contract': 'Month-to-month', 'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Electronic check',
        'MonthlyCharges': 29.85, 'TotalCharges': 29.85
    },
    {
        'gender': 'Male', 'SeniorCitizen': 0, 'Partner': 'Yes',
        'Dependents': 'Yes', 'tenure': 60, 'PhoneService': 'Yes',
        'MultipleLines': 'Yes', 'InternetService': 'Fiber optic',
        'OnlineSecurity': 'Yes', 'OnlineBackup': 'Yes',
        'DeviceProtection': 'Yes', 'TechSupport': 'Yes',
        'StreamingTV': 'Yes', 'StreamingMovies': 'Yes',
        'Contract': 'Two year', 'PaperlessBilling': 'No',
        'PaymentMethod': 'Bank transfer (automatic)',
        'MonthlyCharges': 100.35, 'TotalCharges': 6025.0
    }
])

preds  = loaded_pipeline.predict(new_customers)
probas = loaded_pipeline.predict_proba(new_customers)[:, 1]

for i, (pred, prob) in enumerate(zip(preds, probas), 1):
    label = "CHURN ⚠️" if pred == 1 else "STAY ✅"
    print(f"  Customer {i}: {label}  (churn probability: {prob:.2%})")


# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print(f"""
  Dataset       : Telco Customer Churn ({df.shape[0]} records, {df.shape[1]} features)
  Models Trained: Logistic Regression, Random Forest
  Tuning Method : GridSearchCV (5-fold CV, F1 scoring)

  Final Results (Test Set):
  {'─'*40}
  {'Model':<25} {'Accuracy':>10} {'F1-Score':>10} {'ROC-AUC':>10}
  {'─'*40}""")
for m, r in results.items():
    print(f"  {m:<25} {r['Accuracy']:>10.4f} {r['F1-Score']:>10.4f} {r['ROC-AUC']:>10.4f}")
print(f"""  {'─'*40}

  🏆 Best Model : {best_model_name}
  📦 Exported   : churn_pipeline.pkl (joblib)
  📊 Plots      : task2_results.png
""")
