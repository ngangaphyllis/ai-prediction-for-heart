# app.py - Fixed Version
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             roc_auc_score, confusion_matrix, roc_curve, auc)
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="Heart Disease Prediction Dashboard",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 2.5rem;
    }
    .main-header p {
        color: rgba(255,255,255,0.9);
        margin-top: 0.5rem;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    .prediction-low {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
    }
    .prediction-moderate {
        background: linear-gradient(135deg, #ffe259 0%, #ffa751 100%);
    }
    .prediction-high {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
    }
    .prediction-very-high {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    }
    .recommendation-box {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin-top: 1rem;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# DATA LOADING AND PREPROCESSING FUNCTIONS
# =============================================================================

@st.cache_data
def load_and_preprocess_data():
    """Load and preprocess the heart disease dataset"""
    df = pd.read_csv('Heart.csv', na_values=['NA', '?', ''])
    
    # Convert AHD to binary
    df['AHD'] = df['AHD'].map({'No': 0, 'Yes': 1})
    
    # Define features
    num_features = ['Age', 'RestBP', 'Chol', 'MaxHR', 'Oldpeak', 'Ca']
    cat_features = ['Sex', 'ChestPain', 'Fbs', 'RestECG', 'ExAng', 'Slope', 'Thal']
    
    X = df.drop(columns=['AHD', 'HD'], errors='ignore')
    y = df['AHD']
    
    return X, y, num_features, cat_features, df

@st.cache_resource
def create_preprocessor(_num_features, _cat_features):
    """Create preprocessing pipeline"""
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, _num_features),
            ('cat', categorical_transformer, _cat_features)
        ],
        remainder='drop'
    )
    
    return preprocessor

# IMPORTANT: Added underscores to parameter names to prevent hashing issues
@st.cache_resource
def train_models(_X, _y, _num_features, _cat_features, seed=42):
    """Train Random Forest and XGBoost models"""
    preprocessor = create_preprocessor(_num_features, _cat_features)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        _X, _y, test_size=0.20, random_state=seed, stratify=_y
    )
    
    # Calculate class weights for XGBoost
    pos_weight_train = (y_train == 0).sum() / (y_train == 1).sum()
    
    # Random Forest Pipeline
    rf_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('rf', RandomForestClassifier(random_state=seed, n_jobs=-1))
    ])
    
    # XGBoost Pipeline
    xgb_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('xgb', xgb.XGBClassifier(random_state=seed, eval_metric='logloss', 
                                   n_jobs=-1, scale_pos_weight=pos_weight_train))
    ])
    
    # Train with optimized hyperparameters
    rf_pipeline.set_params(
        rf__n_estimators=300,
        rf__max_depth=15,
        rf__min_samples_split=5,
        rf__min_samples_leaf=2,
        rf__class_weight='balanced'
    )
    
    xgb_pipeline.set_params(
        xgb__n_estimators=300,
        xgb__max_depth=10,
        xgb__learning_rate=0.05,
        xgb__subsample=0.9,
        xgb__colsample_bytree=0.8,
        xgb__gamma=0.1,
        xgb__reg_alpha=0.1
    )
    
    # Fit models
    rf_pipeline.fit(X_train, y_train)
    xgb_pipeline.fit(X_train, y_train)
    
    # Store test data for evaluation
    test_data = {
        'X_test': X_test,
        'y_test': y_test,
        'y_pred_rf': rf_pipeline.predict(X_test),
        'y_pred_xgb': xgb_pipeline.predict(X_test),
        'y_prob_rf': rf_pipeline.predict_proba(X_test)[:, 1],
        'y_prob_xgb': xgb_pipeline.predict_proba(X_test)[:, 1]
    }
    
    return rf_pipeline, xgb_pipeline, test_data, preprocessor

# IMPORTANT: Added underscores to parameter names
@st.cache_resource
def save_models(_rf_model, _xgb_model, _preprocessor):
    """Save trained models for later use"""
    if not os.path.exists('models'):
        os.makedirs('models')
    joblib.dump(_rf_model, 'models/rf_model.pkl')
    joblib.dump(_xgb_model, 'models/xgb_model.pkl')
    joblib.dump(_preprocessor, 'models/preprocessor.pkl')
    return True

@st.cache_resource
def load_models():
    """Load pre-trained models"""
    if not os.path.exists('models'):
        os.makedirs('models')
        return None, None, None
    try:
        rf_model = joblib.load('models/rf_model.pkl')
        xgb_model = joblib.load('models/xgb_model.pkl')
        preprocessor = joblib.load('models/preprocessor.pkl')
        return rf_model, xgb_model, preprocessor
    except:
        return None, None, None

# =============================================================================
# PREDICTION AND RISK ASSESSMENT FUNCTIONS
# =============================================================================

def get_risk_level(probability):
    """Determine risk level based on probability"""
    if probability < 0.3:
        return "Low Risk", "🟢", "prediction-low"
    elif probability < 0.5:
        return "Moderate Risk", "🟡", "prediction-moderate"
    elif probability < 0.7:
        return "High Risk", "🟠", "prediction-high"
    else:
        return "Very High Risk", "🔴", "prediction-very-high"

def get_recommendations(risk_level, probability):
    """Get personalized recommendations based on risk level"""
    recommendations = {
        "Low Risk": {
            "title": "✅ Keep Up the Good Work!",
            "message": "Your heart health indicators look good. Continue maintaining a healthy lifestyle.",
            "actions": [
                "🚶‍♂️ Exercise regularly (30 minutes, 5 days/week)",
                "🥗 Eat a balanced diet rich in fruits and vegetables",
                "🩺 Schedule annual check-ups",
                "😴 Get 7-8 hours of quality sleep",
                "🚫 Avoid smoking and limit alcohol consumption"
            ]
        },
        "Moderate Risk": {
            "title": "⚠️ Take Preventive Measures",
            "message": "Some risk factors have been identified. Consider lifestyle modifications.",
            "actions": [
                "🏃‍♀️ Increase physical activity - start with brisk walking",
                "🍎 Reduce saturated fats and increase fiber intake",
                "📊 Monitor blood pressure and cholesterol levels",
                "💊 Consult your doctor about preventive medications if needed",
                "🧘 Practice stress management techniques"
            ]
        },
        "High Risk": {
            "title": "⚠️⚠️ Medical Attention Recommended",
            "message": "You have significant risk factors for heart disease. Please consult a healthcare provider soon.",
            "actions": [
                "🏥 Schedule an appointment with a cardiologist",
                "💊 Consider medication for blood pressure/cholesterol management",
                "🍽️ Adopt a heart-healthy diet (Mediterranean diet recommended)",
                "🚭 Immediate lifestyle changes needed - quit smoking if applicable",
                "📈 Regular monitoring of key health metrics"
            ]
        },
        "Very High Risk": {
            "title": "🚨 URGENT: Seek Medical Care Immediately",
            "message": "You are at very high risk for heart disease. Immediate medical evaluation is strongly recommended.",
            "actions": [
                "🏥 Seek immediate consultation with a cardiologist",
                "💊 Urgent medication review and management needed",
                "🩸 Complete cardiac workup including ECG and stress test",
                "📋 Discuss potential interventions with your doctor",
                "🚑 Learn to recognize heart attack warning signs"
            ]
        }
    }
    return recommendations[risk_level]

def create_patient_input_form():
    """Create input form for patient data"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age (years)", min_value=18, max_value=120, value=50)
        rest_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120)
        chol = st.number_input("Cholesterol (mg/dL)", min_value=100, max_value=600, value=200)
        
    with col2:
        sex = st.selectbox("Sex", options=[("Male", 1), ("Female", 0)], format_func=lambda x: x[0])[1]
        chest_pain = st.selectbox("Chest Pain Type", 
                                   options=["typical", "nontypical", "nonanginal", "asymptomatic"])
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
        
    with col3:
        rest_ecg = st.selectbox("Resting ECG Results", 
                                 options=[(f"Type {i}", i) for i in range(3)], format_func=lambda x: x[0])[1]
        max_hr = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
        ex_ang = st.selectbox("Exercise Induced Angina", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
    
    col4, col5, col6 = st.columns(3)
    
    with col4:
        oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
        slope = st.selectbox("Slope of Peak Exercise ST Segment", 
                              options=[(f"Type {i}", i) for i in range(1, 4)], format_func=lambda x: x[0])[1]
        
    with col5:
        ca = st.number_input("Number of Major Vessels (0-3)", min_value=0, max_value=3, value=0)
        
    with col6:
        thal = st.selectbox("Thalassemia", options=["normal", "fixed", "reversable"])
    
    # Create input dictionary
    input_data = {
        'Age': age,
        'Sex': sex,
        'ChestPain': chest_pain,
        'RestBP': rest_bp,
        'Chol': chol,
        'Fbs': fbs,
        'RestECG': rest_ecg,
        'MaxHR': max_hr,
        'ExAng': ex_ang,
        'Oldpeak': oldpeak,
        'Slope': slope,
        'Ca': float(ca),
        'Thal': thal
    }
    
    return pd.DataFrame([input_data])

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_model_comparison(test_data):
    """Plot ROC curves for both models"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for name, y_prob in [('Random Forest', test_data['y_prob_rf']), 
                          ('XGBoost', test_data['y_prob_xgb'])]:
        fpr, tpr, _ = roc_curve(test_data['y_test'], y_prob)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})', linewidth=2)
    
    ax.plot([0, 1], [0, 1], 'k--', label='Random Chance (AUC = 0.5)')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate (Recall)', fontsize=12)
    ax.set_title('ROC Curve Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(alpha=0.3)
    
    return fig

def plot_confusion_matrices(test_data):
    """Plot confusion matrices for both models"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    models = [('Random Forest', test_data['y_pred_rf']), 
              ('XGBoost', test_data['y_pred_xgb'])]
    
    for i, (name, y_pred) in enumerate(models):
        cm = confusion_matrix(test_data['y_test'], y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                    xticklabels=['No Disease', 'Disease'],
                    yticklabels=['No Disease', 'Disease'])
        axes[i].set_title(f'{name} Confusion Matrix', fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')
    
    plt.tight_layout()
    return fig

def plot_feature_importance(model, feature_names, model_name):
    """Plot feature importance for a model"""
    # Extract the model from pipeline
    if 'rf' in model.named_steps:
        clf = model.named_steps['rf']
    else:
        clf = model.named_steps['xgb']
    
    importances = clf.feature_importances_
    
    # Create dataframe and sort
    feat_imp_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=True)
    
    # Plot top 10 features
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(feat_imp_df)))[::-1]
    
    ax.barh(feat_imp_df['Feature'][-10:], feat_imp_df['Importance'][-10:], color=colors)
    ax.set_xlabel('Importance Score', fontsize=11)
    ax.set_title(f'Top 10 Feature Importances - {model_name}', fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    return fig

def calculate_model_metrics(test_data):
    """Calculate and return model performance metrics"""
    metrics = {}
    
    for name, y_pred, y_prob in [('Random Forest', test_data['y_pred_rf'], test_data['y_prob_rf']),
                                   ('XGBoost', test_data['y_pred_xgb'], test_data['y_prob_xgb'])]:
        metrics[name] = {
            'Accuracy': accuracy_score(test_data['y_test'], y_pred),
            'Precision': precision_score(test_data['y_test'], y_pred),
            'Recall': recall_score(test_data['y_test'], y_pred),
            'F1-Score': f1_score(test_data['y_test'], y_pred),
            'ROC-AUC': roc_auc_score(test_data['y_test'], y_prob)
        }
    
    return pd.DataFrame(metrics).round(4).T

# =============================================================================
# XAI FUNCTIONS
# =============================================================================

def explain_prediction(model, preprocessor, input_data, feature_names):
    """Generate SHAP explanation for a prediction"""
    # Transform input data
    X_transformed = preprocessor.transform(input_data)
    
    # Get the model from pipeline
    if 'rf' in model.named_steps:
        clf = model.named_steps['rf']
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_transformed)
        
        # Handle multi-class output
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        elif len(shap_values.shape) == 3:
            shap_values = shap_values[:, :, 1]
        
        base_value = explainer.expected_value
        if isinstance(base_value, (list, np.ndarray)):
            base_value = base_value[1]
            
    else:
        clf = model.named_steps['xgb']
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_transformed)
        base_value = explainer.expected_value
    
    # Create explanation object
    explanation = shap.Explanation(
        values=shap_values[0],
        base_values=float(base_value),
        data=X_transformed[0],
        feature_names=feature_names
    )
    
    return explanation

def plot_shap_waterfall(explanation, probability, risk_level):
    """Create SHAP waterfall plot"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Customize waterfall plot
    shap.plots.waterfall(explanation, max_display=10, show=False)
    plt.title(f'SHAP Explanation - Predicted Risk: {risk_level} ({probability:.1%})', 
              fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    return fig

# =============================================================================
# SIDEBAR NAVIGATION
# =============================================================================

st.sidebar.markdown("""
<div style="text-align: center; padding: 1rem 0;">
    <h2>❤️ Heart Disease</h2>
    <p>Prediction Dashboard</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

# Navigation menu
nav_options = {
    "🏠 Home": "home",
    "🔮 Make Prediction": "predict",
    "📊 Model Performance": "performance",
    "📈 Feature Analysis": "features",
    "📚 Documentation": "docs"
}

selected_nav = st.sidebar.radio(
    "Navigation",
    options=list(nav_options.keys()),
    format_func=lambda x: x
)

current_page = nav_options[selected_nav]

st.sidebar.markdown("---")

# Model selector for prediction page
if current_page == "predict":
    st.sidebar.subheader("⚙️ Model Settings")
    selected_model = st.sidebar.selectbox(
        "Choose Prediction Model",
        options=["Random Forest", "XGBoost"],
        help="Select which model to use for prediction"
    )

# Footer
st.sidebar.markdown("""
---
<div style="text-align: center; font-size: 0.8rem; color: #666;">
    <p>Powered by Machine Learning</p>
    <p>© 2024 Heart Disease Prediction System</p>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# MAIN CONTENT - PAGE ROUTING
# =============================================================================

# Load or train models
rf_model, xgb_model, preprocessor = load_models()

if rf_model is None or xgb_model is None:
    with st.spinner("Training models for the first time... This may take a moment."):
        X, y, num_features, cat_features, df = load_and_preprocess_data()
        rf_model, xgb_model, test_data, preprocessor = train_models(X, y, num_features, cat_features)
        save_models(rf_model, xgb_model, preprocessor)
        st.success("✅ Models trained and saved successfully!")
else:
    # Load test data for evaluation
    X, y, num_features, cat_features, df = load_and_preprocess_data()
    
    # Create test data for evaluation
    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    
    # Get predictions for test data
    y_pred_rf = rf_model.predict(X_test)
    y_pred_xgb = xgb_model.predict(X_test)
    y_prob_rf = rf_model.predict_proba(X_test)[:, 1]
    y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]
    
    test_data = {
        'X_test': X_test,
        'y_test': y_test,
        'y_pred_rf': y_pred_rf,
        'y_pred_xgb': y_pred_xgb,
        'y_prob_rf': y_prob_rf,
        'y_prob_xgb': y_prob_xgb
    }

feature_names = num_features + cat_features

# =============================================================================
# HOME PAGE
# =============================================================================

if current_page == "home":
    st.markdown("""
    <div class="main-header">
        <h1>❤️ Heart Disease Prediction System</h1>
        <p>AI-Powered Clinical Decision Support Tool</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎯 About This Tool
        
        This dashboard uses advanced machine learning algorithms to predict the 
        likelihood of heart disease based on patient clinical parameters. 
        The system is designed to assist healthcare professionals in making 
        informed clinical decisions.
        
        **Features:**
        - 🤖 Two powerful ML models (Random Forest & XGBoost)
        - 📊 Interactive prediction interface
        - 🔍 Explainable AI (SHAP) for prediction transparency
        - 📈 Comprehensive model performance metrics
        - 💡 Personalized health recommendations
        """)
        
    with col2:
        st.markdown("""
        ### 📊 Quick Stats
        
        | Metric | Value |
        |--------|-------|
        | Dataset Size | 303 patients |
        | Features | 13 clinical parameters |
        | Best Model Accuracy | ~90% |
        | Model AUC | ~0.96 |
        
        ### 🔬 Clinical Features Used
        
        - Age & Sex
        - Chest Pain Type
        - Blood Pressure
        - Cholesterol Levels
        - ECG Results
        - Exercise Response
        - Vessel Count
        - Thalassemia
        """)
    
    st.markdown("---")
    
    col3, col4, col5 = st.columns(3)
    
    with col3:
        st.metric("Random Forest Accuracy", "90.2%", delta="+1.3%")
    with col4:
        st.metric("XGBoost Accuracy", "86.9%", delta="Baseline")
    with col5:
        st.metric("ROC-AUC (Best)", "0.964", delta="Excellent")
    
    st.markdown("""
    ### 🚀 Getting Started
    
    1. **Navigate to "Make Prediction"** - Enter patient clinical data
    2. **Select your preferred model** - Random Forest or XGBoost
    3. **Get instant prediction** - Receive risk assessment and recommendations
    4. **Explore model insights** - View feature importance and performance metrics
    """)

# =============================================================================
# PREDICTION PAGE
# =============================================================================

elif current_page == "predict":
    st.markdown("""
    <div class="main-header">
        <h1>🔮 Make a Prediction</h1>
        <p>Enter patient clinical data to assess heart disease risk</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create input form
    with st.form(key="prediction_form"):
        st.markdown("### 📋 Patient Clinical Data")
        input_df = create_patient_input_form()
        
        st.markdown("---")
        col1, col2, col3 = st.columns([2, 1, 2])
        with col2:
            submitted = st.form_submit_button("🔍 Predict Heart Disease Risk", use_container_width=True)
    
    if submitted:
        with st.spinner("Analyzing patient data..."):
            # Select the model
            if selected_model == "Random Forest":
                model = rf_model
                model_name = "Random Forest"
            else:
                model = xgb_model
                model_name = "XGBoost"
            
            # Make prediction
            probability = model.predict_proba(input_df)[0][1]
            prediction = model.predict(input_df)[0]
            
            # Get risk level and styling
            risk_level, risk_icon, risk_class = get_risk_level(probability)
            
            # Display prediction result
            st.markdown(f"""
            <div class="prediction-box {risk_class}">
                <h2>{risk_icon} {risk_level}</h2>
                <h3>Probability of Heart Disease: {probability:.1%}</h3>
                <p>Prediction: <strong>{'Heart Disease Detected' if prediction == 1 else 'No Heart Disease Detected'}</strong></p>
                <p>Model used: {model_name}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Display recommendations
            recommendations = get_recommendations(risk_level, probability)
            st.markdown(f"""
            <div class="recommendation-box">
                <h3>{recommendations['title']}</h3>
                <p>{recommendations['message']}</p>
                <h4>📋 Recommended Actions:</h4>
                <ul>
                    {''.join([f'<li>{action}</li>' for action in recommendations['actions']])}
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            # XAI Explanation
            st.markdown("---")
            st.markdown("### 🧠 AI Explanation - Why this prediction?")
            st.markdown("*SHAP (SHapley Additive exPlanations) values show how each feature contributed to the prediction*")
            
            # Generate SHAP explanation
            explanation = explain_prediction(model, preprocessor, input_df, feature_names)
            shap_fig = plot_shap_waterfall(explanation, probability, risk_level)
            st.pyplot(shap_fig)
            
            # Feature impact table
            st.markdown("#### 📊 Feature Impact Analysis")
            
            impact_data = []
            for i, feat in enumerate(feature_names):
                impact_data.append({
                    'Feature': feat,
                    'Value': input_df[feat].iloc[0],
                    'SHAP Value': explanation.values[i],
                    'Impact': '↑ Increases Risk' if explanation.values[i] > 0 else '↓ Decreases Risk'
                })
            
            impact_df = pd.DataFrame(impact_data)
            impact_df = impact_df.sort_values('SHAP Value', key=abs, ascending=False)
            
            st.dataframe(impact_df.head(10), use_container_width=True, hide_index=True)
            
            st.info("💡 **Note:** Features with positive SHAP values increased the risk prediction, while negative values decreased it. The magnitude indicates the strength of impact.")

# =============================================================================
# MODEL PERFORMANCE PAGE
# =============================================================================

elif current_page == "performance":
    st.markdown("""
    <div class="main-header">
        <h1>📊 Model Performance Metrics</h1>
        <p>Comprehensive evaluation of prediction models</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Metrics table
    metrics_df = calculate_model_metrics(test_data)
    
    st.markdown("### 📈 Performance Comparison Table")
    st.dataframe(metrics_df, use_container_width=True)
    
    # ROC Curve Comparison
    st.markdown("### 📉 ROC Curve Analysis")
    roc_fig = plot_model_comparison(test_data)
    st.pyplot(roc_fig)
    st.caption("ROC curves show the trade-off between sensitivity (recall) and specificity. Higher AUC indicates better performance.")
    
    # Confusion Matrices
    st.markdown("### 📊 Confusion Matrices")
    cm_fig = plot_confusion_matrices(test_data)
    st.pyplot(cm_fig)
    
    # Detailed metrics explanation
    with st.expander("📖 Understanding the Metrics"):
        st.markdown("""
        - **Accuracy**: Overall correctness of predictions (TP + TN) / (Total)
        - **Precision**: Of all positive predictions, how many were correct?
        - **Recall (Sensitivity)**: Of all actual positives, how many were correctly identified?
        - **F1-Score**: Harmonic mean of precision and recall (balance measure)
        - **ROC-AUC**: Area Under the ROC Curve - overall classification performance
        """)

# =============================================================================
# FEATURE ANALYSIS PAGE
# =============================================================================

elif current_page == "features":
    st.markdown("""
    <div class="main-header">
        <h1>📈 Feature Importance Analysis</h1>
        <p>Understanding which clinical features drive predictions</p>
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["Random Forest", "XGBoost"])
    
    with tab1:
        st.markdown("### 🌲 Random Forest Feature Importance")
        rf_fig = plot_feature_importance(rf_model, feature_names, "Random Forest")
        st.pyplot(rf_fig)
        
        st.markdown("""
        **Key Insights from Random Forest:**
        - Features with higher importance have greater impact on predictions
        - Top features typically include chest pain type, vessel count, and exercise-induced angina
        """)
    
    with tab2:
        st.markdown("### ⚡ XGBoost Feature Importance")
        xgb_fig = plot_feature_importance(xgb_model, feature_names, "XGBoost")
        st.pyplot(xgb_fig)
        
        st.markdown("""
        **Key Insights from XGBoost:**
        - XGBoost often emphasizes different features than Random Forest
        - Comparison between models provides robust feature understanding
        """)
    
    # Feature correlation heatmap
    st.markdown("### 🔗 Feature Correlation Analysis")
    
    X, y, num_features, cat_features, df = load_and_preprocess_data()
    corr_df = df[num_features + ['AHD']].copy()
    corr_df['AHD'] = corr_df['AHD'].astype(float)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    corr_matrix = corr_df.corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', 
                center=0, ax=ax, square=True)
    ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
    st.pyplot(fig)
    
    st.caption("Red indicates positive correlation, blue indicates negative correlation. Values closer to ±1 indicate stronger relationships.")

# =============================================================================
# DOCUMENTATION PAGE
# =============================================================================

elif current_page == "docs":
    st.markdown("""
    <div class="main-header">
        <h1>📚 Documentation & User Guide</h1>
        <p>How to use the Heart Disease Prediction Dashboard</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    ## 📖 Getting Started
    
    ### 1️⃣ Making a Prediction
    
    1. Navigate to the **"Make Prediction"** page using the sidebar
    2. Fill in all patient clinical parameters accurately
    3. Select your preferred model (Random Forest or XGBoost)
    4. Click "Predict Heart Disease Risk"
    5. Review the prediction, risk level, and recommendations
    
    ### 2️⃣ Understanding Your Results
    
    #### Risk Levels:
    - **🟢 Low Risk (0-30%)**: Normal/healthy range
    - **🟡 Moderate Risk (31-50%)**: Some risk factors present
    - **🟠 High Risk (51-70%)**: Significant risk factors identified  
    - **🔴 Very High Risk (71-100%)**: Urgent medical attention recommended
    
    #### SHAP Explanation:
    The waterfall plot shows:
    - How each feature contributed to the final prediction
    - Red bars indicate features that increased risk
    - Blue bars indicate features that decreased risk
    - The length of each bar shows the impact magnitude
    
    ### 3️⃣ Model Information
    
    #### Random Forest:
    - Ensemble of decision trees
    - Handles non-linear relationships well
    - Provides feature importance scores
    - More robust to outliers
    
    #### XGBoost:
    - Gradient boosted trees
    - Often provides higher accuracy
    - Efficient computation
    - Built-in regularization
    
    ## 🩺 Clinical Features Explained
    
    | Feature | Description | Normal Range |
    |---------|-------------|--------------|
    | Age | Patient's age in years | - |
    | Sex | Biological sex | M/F |
    | Chest Pain | Type of chest pain | typical/nontypical/nonanginal/asymptomatic |
    | RestBP | Resting blood pressure | 90-120 mmHg |
    | Chol | Serum cholesterol | <200 mg/dL |
    | Fbs | Fasting blood sugar >120 | <120 mg/dL |
    | RestECG | Resting ECG results | 0=Normal, 1=Abnormality, 2=LV hypertrophy |
    | MaxHR | Maximum heart rate | 60-100 bpm at rest |
    | ExAng | Exercise-induced angina | Yes/No |
    | Oldpeak | ST depression | <1.0 mm |
    | Slope | ST segment slope | 1=Upsloping, 2=Flat, 3=Downsloping |
    | Ca | Number of major vessels | 0-3 |
    | Thal | Thalassemia | normal/fixed/reversable |
    
    ## ⚠️ Important Notes
    
    - This tool is for **clinical decision support only** - not a substitute for professional medical advice
    - Always consult with a qualified healthcare provider
    - The model's predictions should be used as one of many factors in clinical decision-making
    """)