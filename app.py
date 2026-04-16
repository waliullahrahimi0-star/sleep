"""
COM572 – Machine Learning Coursework: Task 1
Streamlit Application: app.py

Project Title: Predicting Sleep Disorders Using Machine Learning

Description:
    A user-friendly web application that allows healthcare practitioners
    and individuals to receive a preliminary assessment of sleep disorder
    risk based on lifestyle and biometric inputs. The model is trained
    entirely within the app using st.cache_resource (no pickle file required).

Usage:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="Sleep Disorder Predictor",
    page_icon="🛌",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# =============================================================================
# CUSTOM STYLING
# =============================================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.0rem;
        font-weight: 700;
        color: #1a3a5c;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.0rem;
        color: #555;
        margin-bottom: 1.5rem;
    }
    .disclaimer-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 12px 16px;
        border-radius: 4px;
        font-size: 0.85rem;
        color: #856404;
        margin-top: 1rem;
    }
    .result-box-none {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 16px;
        border-radius: 6px;
        margin-top: 1rem;
    }
    .result-box-insomnia {
        background-color: #fff3cd;
        border-left: 4px solid #fd7e14;
        padding: 16px;
        border-radius: 6px;
        margin-top: 1rem;
    }
    .result-box-apnea {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 16px;
        border-radius: 6px;
        margin-top: 1rem;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# CACHED MODEL TRAINING
# =============================================================================

@st.cache_resource(show_spinner="Training the predictive model… please wait.")
def load_and_train():
    """
    Load the dataset, preprocess it, and train the tuned Random Forest model.
    Decorated with st.cache_resource so the model is trained only once per
    server session, ensuring fast subsequent interactions.
    """
    # ---- Load ----
    df = pd.read_csv("Sleep_health_and_lifestyle_dataset.csv")

    # ---- Clean ----
    df = df.drop(columns=["Person ID"])
    df["Sleep Disorder"] = df["Sleep Disorder"].fillna("None")

    # ---- Feature engineering: Blood Pressure ----
    bp = df["Blood Pressure"].str.split("/", expand=True)
    df["Systolic BP"]  = pd.to_numeric(bp[0], errors="coerce")
    df["Diastolic BP"] = pd.to_numeric(bp[1], errors="coerce")
    df = df.drop(columns=["Blood Pressure"])

    # ---- BMI ordinal encoding ----
    bmi_order = {"Normal": 0, "Normal Weight": 0, "Overweight": 1, "Obese": 2}
    df["BMI Category"] = df["BMI Category"].map(bmi_order)

    # ---- Split ----
    X = df.drop(columns=["Sleep Disorder"])
    y = df["Sleep Disorder"]

    num_cols = X.select_dtypes(exclude="object").columns.tolist()
    cat_cols = X.select_dtypes(include="object").columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ---- Preprocessor ----
    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler())
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    preprocessor = ColumnTransformer([
        ("num", numerical_pipeline, num_cols),
        ("cat", categorical_pipeline, cat_cols)
    ])

    # ---- Best model (from GridSearchCV in train.py) ----
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_split=5,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42
        ))
    ])
    model.fit(X_train, y_train)

    classes = model.classes_.tolist()
    return model, classes, cat_cols


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def render_result(prediction: str, proba: dict):
    """Render the prediction result with colour-coded box and probability bars."""
    emoji_map  = {"None": "✅", "Insomnia": "⚠️", "Sleep Apnea": "🚨"}
    box_map    = {"None": "result-box-none", "Insomnia": "result-box-insomnia",
                  "Sleep Apnea": "result-box-apnea"}
    advice_map = {
        "None": (
            "No sleep disorder is detected based on the provided information. "
            "Continue maintaining healthy sleep habits — consistent bedtimes, "
            "limiting screen exposure before sleep, and regular physical activity "
            "all contribute to sustained sleep quality."
        ),
        "Insomnia": (
            "The model indicates a risk of <strong>Insomnia</strong> — "
            "difficulty falling or staying asleep. Common contributing factors "
            "include elevated stress levels, irregular schedules, and high "
            "caffeine intake. Please consult a qualified healthcare professional "
            "for a formal assessment."
        ),
        "Sleep Apnea": (
            "The model indicates a risk of <strong>Sleep Apnoea</strong> — "
            "repeated interruptions to breathing during sleep. This condition "
            "is associated with elevated BMI, high blood pressure, and reduced "
            "sleep quality. Early diagnosis is important; please seek medical advice."
        )
    }

    box_class = box_map[prediction]
    advice    = advice_map[prediction]
    emoji     = emoji_map[prediction]

    st.markdown(f"""
    <div class="{box_class}">
        <strong>{emoji} Predicted Sleep Disorder: {prediction}</strong><br><br>
        <span style="font-size:0.92rem;">{advice}</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### Prediction Probabilities")
    colour_map = {"None": "#28a745", "Insomnia": "#fd7e14", "Sleep Apnea": "#dc3545"}
    for label, prob in sorted(proba.items(), key=lambda x: -x[1]):
        bar_colour = colour_map.get(label, "#6c757d")
        st.markdown(f"""
        <div style="margin-bottom: 6px;">
            <span class="metric-label">{label}</span>
            <div style="background:#e9ecef; border-radius:4px; height:22px; margin-top:3px;">
                <div style="width:{prob*100:.1f}%; background:{bar_colour}; height:22px;
                            border-radius:4px; display:flex; align-items:center;
                            padding-left:8px; color:white; font-size:0.8rem; font-weight:600;">
                    {prob*100:.1f}%
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# APP LAYOUT
# =============================================================================

def main():
    # ---- Header ----
    st.markdown('<div class="main-header">🛌 Sleep Disorder Predictor</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">An AI-assisted tool to assess your risk of a sleep disorder '
        'based on your lifestyle and health profile.</div>',
        unsafe_allow_html=True
    )

    # ---- Load model ----
    model, classes, cat_cols = load_and_train()

    # ---- Disclaimer (always visible) ----
    st.markdown("""
    <div class="disclaimer-box">
        ⚠️ <strong>Medical Disclaimer:</strong> This tool is intended for educational and
        screening purposes only. It does <em>not</em> constitute medical advice, diagnosis,
        or treatment. Always consult a qualified healthcare professional if you have concerns
        about your sleep health.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ===========================================================
    # INPUT SECTION
    # ===========================================================
    st.markdown("### 👤 Personal Information")
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        age    = st.slider("Age (years)", min_value=18, max_value=80, value=35)
    with col2:
        occupation = st.selectbox("Occupation", [
            "Accountant", "Doctor", "Engineer", "Lawyer", "Manager",
            "Nurse", "Sales Representative", "Salesperson",
            "Scientist", "Software Engineer", "Teacher"
        ])
        bmi_label = st.selectbox("BMI Category", ["Normal", "Overweight", "Obese"])

    st.markdown("### 😴 Sleep Information")
    col3, col4 = st.columns(2)
    with col3:
        sleep_duration  = st.slider("Sleep Duration (hours/night)", 4.0, 10.0, 7.0, 0.1)
        quality_of_sleep = st.slider("Quality of Sleep (1–10)", 1, 10, 7)
    with col4:
        stress_level     = st.slider("Stress Level (1–10)", 1, 10, 5)
        physical_activity = st.slider("Physical Activity (mins/day)", 0, 120, 40)

    st.markdown("### 🫀 Health Metrics")
    col5, col6 = st.columns(2)
    with col5:
        systolic  = st.slider("Systolic Blood Pressure (mmHg)", 90, 180, 120)
        diastolic = st.slider("Diastolic Blood Pressure (mmHg)", 50, 120, 80)
    with col6:
        heart_rate   = st.slider("Resting Heart Rate (bpm)", 50, 110, 70)
        daily_steps  = st.number_input("Daily Steps", min_value=0, max_value=20000,
                                       value=7000, step=500)

    # ===========================================================
    # PREDICTION
    # ===========================================================
    st.markdown("---")
    predict_btn = st.button("🔍 Predict Sleep Disorder", use_container_width=True, type="primary")

    if predict_btn:
        # Map BMI label to ordinal value (matching train.py encoding)
        bmi_map = {"Normal": 0, "Overweight": 1, "Obese": 2}

        input_df = pd.DataFrame([{
            "Gender"                  : gender,
            "Age"                     : age,
            "Occupation"              : occupation,
            "Sleep Duration"          : sleep_duration,
            "Quality of Sleep"        : quality_of_sleep,
            "Physical Activity Level" : physical_activity,
            "Stress Level"            : stress_level,
            "BMI Category"            : bmi_map[bmi_label],
            "Heart Rate"              : heart_rate,
            "Daily Steps"             : daily_steps,
            "Systolic BP"             : systolic,
            "Diastolic BP"            : diastolic
        }])

        prediction = model.predict(input_df)[0]
        proba_arr  = model.predict_proba(input_df)[0]
        proba_dict = dict(zip(classes, proba_arr))

        st.markdown("## 📊 Prediction Result")
        render_result(prediction, proba_dict)

        # Feature importance note
        with st.expander("ℹ️ About this model"):
            st.markdown("""
            This application uses a **tuned Random Forest classifier** trained on the
            *Sleep Health and Lifestyle* dataset (n = 374). Random Forest combines
            multiple decision trees to reduce overfitting and improve generalisation.

            **Test-set performance:**
            | Metric | Score |
            |---|---|
            | Accuracy | 96.0% |
            | Weighted F1-Score | 0.960 |
            | Weighted Precision | 0.961 |
            | Weighted Recall | 0.960 |

            The most influential features are typically **Quality of Sleep**,
            **Stress Level**, **BMI Category**, and **Blood Pressure**.
            """)

    # Footer
    st.markdown("---")
    st.markdown(
        "<small style='color:#999;'>COM572 Machine Learning Coursework · "
        "Sleep Health & Lifestyle Dataset · Random Forest Classifier</small>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
