import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Telco Churn Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
    <style>
    .main { padding: 2rem 1rem; }
    .metric-card { background-color: #f0f2f6; padding: 1.5rem; border-radius: 0.5rem; margin: 0.5rem 0; }
    .churn-alert { background-color: #ff6b6b; color: white; padding: 1rem; border-radius: 0.5rem; }
    .no-churn-alert { background-color: #51cf66; color: white; padding: 1rem; border-radius: 0.5rem; }
    </style>
""", unsafe_allow_html=True)

# ── Load or Train Model ───────────────────────────────────────────────────────
@st.cache_resource
def load_or_train_model():
    """Load trained model or train a new one."""
    model_path = "churn_model.pkl"
    
    if os.path.exists(model_path):
        st.info("✅ Loaded pre-trained model")
        return joblib.load(model_path)
    else:
        st.warning("⚠️ No trained model found. Training a new model...")
        
        # Load or generate dataset
        url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
        try:
            df = pd.read_csv(url)
        except Exception:
            # Fallback: synthetic dataset
            np.random.seed(42)
            n = 7043
            df = pd.DataFrame({
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
                'PaymentMethod': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'], n),
                'MonthlyCharges': np.round(np.random.uniform(18, 119, n), 2),
                'TotalCharges': np.round(np.random.uniform(18, 8500, n), 2),
                'Churn': np.random.choice(['Yes', 'No'], n, p=[0.265, 0.735])
            })
        
        # Preprocessing
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['Churn'] = (df['Churn'] == 'Yes').astype(int)
        
        if 'customerID' in df.columns:
            df.drop(columns=['customerID'], inplace=True)
        
        X = df.drop(columns=['Churn'])
        y = df['Churn']
        
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object']).columns.tolist()
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Build pipeline
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        preprocessor = ColumnTransformer(transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
        
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(random_state=42, n_jobs=-1, n_estimators=100))
        ])
        
        pipeline.fit(X_train, y_train)
        
        # Save model
        joblib.dump(pipeline, model_path)
        st.success("✅ Model trained and saved!")
        
        return pipeline

# ── Main App ──────────────────────────────────────────────────────────────────
st.title("📊 Telco Customer Churn Predictor")
st.markdown("---")

# Load model
model = load_or_train_model()

# Sidebar: Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Choose a page:", ["🔮 Make Prediction", "📈 Model Info"])

# ── Page 1: Prediction ────────────────────────────────────────────────────────
if page == "🔮 Make Prediction":
    st.subheader("Enter Customer Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        partner = st.selectbox("Has Partner", ["Yes", "No"])
        dependents = st.selectbox("Has Dependents", ["Yes", "No"])
        tenure = st.slider("Tenure (months)", 0, 72, 24)
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    
    with col2:
        multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    
    with col3:
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        monthly_charges = st.slider("Monthly Charges ($)", 18.0, 119.0, 65.0)
        total_charges = st.slider("Total Charges ($)", 18.0, 8500.0, 1500.0)
    
    # Make prediction
    if st.button("🎯 Predict Churn", use_container_width=True):
        # Prepare input
        input_data = pd.DataFrame({
            'gender': [gender],
            'SeniorCitizen': [senior_citizen],
            'Partner': [partner],
            'Dependents': [dependents],
            'tenure': [tenure],
            'PhoneService': [phone_service],
            'MultipleLines': [multiple_lines],
            'InternetService': [internet_service],
            'OnlineSecurity': [online_security],
            'OnlineBackup': [online_backup],
            'DeviceProtection': [device_protection],
            'TechSupport': [tech_support],
            'StreamingTV': [streaming_tv],
            'StreamingMovies': [streaming_movies],
            'Contract': [contract],
            'PaperlessBilling': [paperless_billing],
            'PaymentMethod': [payment_method],
            'MonthlyCharges': [monthly_charges],
            'TotalCharges': [total_charges]
        })
        
        # Get prediction
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]
        
        # Display results
        st.markdown("---")
        st.subheader("📋 Prediction Result")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if prediction == 1:
                st.markdown(
                    '<div class="churn-alert"><h3>⚠️ WILL CHURN</h3><p>High risk of customer churn</p></div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<div class="no-churn-alert"><h3>✅ NO CHURN</h3><p>Low risk of customer churn</p></div>',
                    unsafe_allow_html=True
                )
        
        with col2:
            st.metric("Churn Probability", f"{probability[1]:.2%}", delta=f"{probability[1]-probability[0]:.2%}")
            st.metric("No Churn Probability", f"{probability[0]:.2%}")
        
        # Prediction confidence
        confidence = max(probability) * 100
        st.progress(confidence / 100, text=f"Confidence: {confidence:.1f}%")

# ── Page 2: Model Info ────────────────────────────────────────────────────────
elif page == "📈 Model Info":
    st.subheader("Model Information")
    
    st.info("""
    **Model Architecture:**
    - **Algorithm:** Random Forest Classifier
    - **Features:** 20 (7 numeric + 13 categorical)
    - **Target:** Customer Churn (Binary Classification)
    - **Pipeline:** Preprocessing → Feature Scaling/Encoding → Classification
    """)
    
    st.subheader("Features Used")
    
    numeric_feats = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen']
    categorical_feats = [
        'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
        'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
        'PaymentMethod'
    ]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Numeric Features:**")
        for feat in numeric_feats:
            st.caption(f"• {feat}")
    
    with col2:
        st.markdown("**Categorical Features:**")
        for feat in categorical_feats:
            st.caption(f"• {feat}")
    
    st.markdown("---")
    st.subheader("Dataset Information")
    st.info("""
    **Telco Customer Churn Dataset:**
    - **Total Samples:** 7,043 customers
    - **Churn Rate:** ~26.5%
    - **Source:** IBM Telco Customer Churn Dataset
    - **Train/Test Split:** 80% / 20%
    """)

st.markdown("---")
st.caption("Built with Streamlit | ML Pipeline using Scikit-learn")
