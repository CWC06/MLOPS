"""
Integrated Health Risk Prediction Web Application.

Supports two models:
  - Heart Disease Risk (binary: No Disease / Disease)
  - Lung Cancer Risk   (multiclass: Low / Medium / High)

Run from project root:
    streamlit run src/webapp/app.py
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Make sure project root is on the path when launched from any directory
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ------------------------------------------------------------------ #
# Page configuration                                                   #
# ------------------------------------------------------------------ #
st.set_page_config(
    page_title="Health Risk Prediction App",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------ #
# CSS                                                                  #
# ------------------------------------------------------------------ #
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.4rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        background-color: #f7f9fc;
        border: 1px solid #dce3ea;
        margin-top: 1rem;
    }
    .risk-high   { color: #d62728; font-weight: bold; font-size: 1.4rem; }
    .risk-medium { color: #ff7f0e; font-weight: bold; font-size: 1.4rem; }
    .risk-low    { color: #2ca02c; font-weight: bold; font-size: 1.4rem; }
    .risk-positive { color: #d62728; font-weight: bold; font-size: 1.4rem; }
    .risk-negative { color: #2ca02c; font-weight: bold; font-size: 1.4rem; }
    .metric-card {
        background: #fff;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin: 0.3rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------ #
# Header                                                               #
# ------------------------------------------------------------------ #
st.markdown('<h1 class="main-header">🏥 Health Risk Prediction Application</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">MLOps Assignment — Heart Disease & Lung Cancer Risk Models</p>',
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------ #
# Sidebar — model selection                                            #
# ------------------------------------------------------------------ #
st.sidebar.title("⚙️ Model Selection")
model_choice = st.sidebar.radio(
    "Choose a prediction model:",
    ["❤️ Heart Disease Risk", "🔬 Lung Cancer Risk"],
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **About the models**

    | Model | Type | Target |
    |---|---|---|
    | Heart Disease | Binary | 0 / 1 |
    | Lung Cancer   | Multiclass | Low / Medium / High |

    Both models are trained with **PyCaret** and tracked via **MLflow**.
    """,
)

# ------------------------------------------------------------------ #
# Helper: render probability table                                     #
# ------------------------------------------------------------------ #

def _render_probabilities(probabilities: dict) -> None:
    if not probabilities:
        return
    st.subheader("Probability / Confidence Scores")
    st.markdown(
        """
        <div style='background:#f0f2f6;padding:0.8rem;border-radius:5px;margin-bottom:0.8rem;'>
        <small>The table shows the model's confidence for each outcome class.
        Higher values indicate stronger confidence.</small>
        </div>
        """,
        unsafe_allow_html=True,
    )
    prob_df = pd.DataFrame(list(probabilities.items()), columns=["Class / Metric", "Value"])
    st.dataframe(prob_df, use_container_width=True, hide_index=True)


# ================================================================== #
# HEART DISEASE                                                        #
# ================================================================== #
if model_choice == "❤️ Heart Disease Risk":
    st.header("❤️ Heart Disease Risk Prediction")
    st.markdown("Predict whether a patient is at risk of heart disease based on clinical features.")

    tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction"])

    # ---------------------------------------------------------------- #
    # Single prediction                                                 #
    # ---------------------------------------------------------------- #
    with tab1:
        st.subheader("Patient Clinical Information")
        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.number_input("Age (years)", min_value=1, max_value=120, value=54)
            sex = st.selectbox("Sex", options=["M", "F"], format_func=lambda x: "Male" if x == "M" else "Female")
            chest_pain = st.selectbox(
                "Chest Pain Type",
                options=["ATA", "NAP", "ASY", "TA"],
                help="ATA=Atypical Angina, NAP=Non-Anginal Pain, ASY=Asymptomatic, TA=Typical Angina",
            )
            resting_bp = st.number_input("Resting Blood Pressure (mmHg)", min_value=50, max_value=250, value=130)

        with col2:
            cholesterol = st.number_input("Cholesterol (mg/dL)", min_value=0, max_value=600, value=250)
            fasting_bs = st.selectbox(
                "Fasting Blood Sugar > 120 mg/dL",
                options=[0, 1],
                format_func=lambda x: "No" if x == 0 else "Yes",
            )
            resting_ecg = st.selectbox(
                "Resting ECG",
                options=["Normal", "ST", "LVH"],
                help="ST=ST-T wave abnormality, LVH=Left Ventricular Hypertrophy",
            )
            max_hr = st.number_input("Maximum Heart Rate Achieved", min_value=50, max_value=250, value=147)

        with col3:
            exercise_angina = st.selectbox(
                "Exercise-Induced Angina",
                options=["N", "Y"],
                format_func=lambda x: "No" if x == "N" else "Yes",
            )
            oldpeak = st.number_input(
                "Oldpeak (ST depression)", min_value=-5.0, max_value=10.0, value=1.4, step=0.1
            )
            st_slope = st.selectbox(
                "ST Slope",
                options=["Up", "Flat", "Down"],
                help="Slope of the peak exercise ST segment",
            )

        if st.button("🔮 Predict Heart Disease Risk", type="primary", use_container_width=True):
            try:
                from src.models.predict_heart import predict as heart_predict

                input_data = {
                    "Age": age,
                    "Sex": sex,
                    "ChestPainType": chest_pain,
                    "RestingBP": resting_bp,
                    "Cholesterol": cholesterol,
                    "FastingBS": fasting_bs,
                    "RestingECG": resting_ecg,
                    "MaxHR": max_hr,
                    "ExerciseAngina": exercise_angina,
                    "Oldpeak": oldpeak,
                    "ST_Slope": st_slope,
                }

                with st.spinner("Running prediction..."):
                    result = heart_predict(input_data, return_proba=True)

                st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                st.subheader("Prediction Result")

                pred = result.get("prediction", "Unknown")
                if str(pred) == "1":
                    st.markdown(
                        '<p class="risk-positive">⚠️ Heart Disease DETECTED (Class 1)</p>',
                        unsafe_allow_html=True,
                    )
                    st.warning("This patient shows signs consistent with heart disease. Please consult a cardiologist.")
                else:
                    st.markdown(
                        '<p class="risk-negative">✅ No Heart Disease Detected (Class 0)</p>',
                        unsafe_allow_html=True,
                    )
                    st.success("The model does not detect significant heart disease risk for this patient.")

                _render_probabilities(result.get("probabilities", {}))
                st.markdown("</div>", unsafe_allow_html=True)

            except FileNotFoundError as e:
                st.error(str(e))
                st.info("Run the training notebook `notebooks/train_heart.ipynb` to train the heart disease model.")
            except Exception as e:
                st.error(f"Prediction error: {e}")

    # ---------------------------------------------------------------- #
    # Batch prediction                                                  #
    # ---------------------------------------------------------------- #
    with tab2:
        st.subheader("Batch Prediction")
        st.markdown(
            """
            Upload a CSV file containing patient records. Required columns:
            `Age`, `Sex`, `ChestPainType`, `RestingBP`, `Cholesterol`,
            `FastingBS`, `RestingECG`, `MaxHR`, `ExerciseAngina`, `Oldpeak`, `ST_Slope`
            """
        )

        uploaded = st.file_uploader("Upload patient CSV", type="csv", key="heart_batch")
        if uploaded is not None:
            try:
                df_upload = pd.read_csv(uploaded)
                st.write(f"Uploaded {len(df_upload)} records. Preview:")
                st.dataframe(df_upload.head(), use_container_width=True)

                if st.button("🔮 Predict All", type="primary", key="heart_batch_predict"):
                    from src.models.predict_heart import predict_batch as heart_batch

                    with st.spinner("Processing..."):
                        results_df = heart_batch(df_upload)

                    st.success("Batch prediction complete!")
                    st.dataframe(results_df, use_container_width=True)

                    csv_out = results_df.to_csv(index=False)
                    st.download_button(
                        "📥 Download Predictions",
                        data=csv_out,
                        file_name="heart_disease_predictions.csv",
                        mime="text/csv",
                    )
            except FileNotFoundError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"Error: {e}")


# ================================================================== #
# LUNG CANCER                                                          #
# ================================================================== #
else:
    st.header("🔬 Lung Cancer Risk Prediction")
    st.markdown("Predict the lung cancer risk level (Low / Medium / High) based on patient risk factors.")

    tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction"])

    # ---------------------------------------------------------------- #
    # Single prediction                                                 #
    # ---------------------------------------------------------------- #
    with tab1:
        st.subheader("Patient Risk Factor Information")

        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.number_input("Age", min_value=0, max_value=120, value=45, step=1, key="lc_age")
            gender = st.selectbox(
                "Gender",
                options=[1, 2],
                format_func=lambda x: "Male" if x == 1 else "Female",
                key="lc_gender",
            )
            air_pollution = st.slider("Air Pollution Exposure", min_value=1, max_value=9, value=5, key="lc_air")
            alcohol_use = st.slider("Alcohol Use", min_value=1, max_value=9, value=3, key="lc_alc")
            dust_allergy = st.slider("Dust Allergy", min_value=1, max_value=9, value=4, key="lc_dust")
            occupational_hazards = st.slider("Occupational Hazards", min_value=1, max_value=9, value=4, key="lc_occ")
            genetic_risk = st.slider("Genetic Risk", min_value=1, max_value=9, value=3, key="lc_gen")

        with col2:
            chronic_lung_disease = st.slider("Chronic Lung Disease", min_value=1, max_value=9, value=2, key="lc_cld")
            balanced_diet = st.slider("Balanced Diet", min_value=1, max_value=9, value=3, key="lc_diet")
            obesity = st.slider("Obesity", min_value=1, max_value=9, value=4, key="lc_ob")
            smoking = st.slider("Smoking", min_value=1, max_value=9, value=5, key="lc_smk")
            passive_smoker = st.slider("Passive Smoker", min_value=1, max_value=9, value=3, key="lc_ps")
            chest_pain_lc = st.slider("Chest Pain", min_value=1, max_value=9, value=4, key="lc_cp")
            coughing_blood = st.slider("Coughing of Blood", min_value=1, max_value=9, value=2, key="lc_cob")

        with col3:
            fatigue = st.slider("Fatigue", min_value=1, max_value=9, value=3, key="lc_fat")
            weight_loss = st.slider("Weight Loss", min_value=1, max_value=9, value=2, key="lc_wl")
            shortness_breath = st.slider("Shortness of Breath", min_value=1, max_value=9, value=3, key="lc_sob")
            wheezing = st.slider("Wheezing", min_value=1, max_value=9, value=2, key="lc_wh")
            swallowing_difficulty = st.slider("Swallowing Difficulty", min_value=1, max_value=9, value=2, key="lc_sd")
            clubbing_nails = st.slider("Clubbing of Finger Nails", min_value=1, max_value=9, value=1, key="lc_cn")
            frequent_cold = st.slider("Frequent Cold", min_value=1, max_value=9, value=2, key="lc_fc")
            dry_cough = st.slider("Dry Cough", min_value=1, max_value=9, value=3, key="lc_dc")
            snoring = st.slider("Snoring", min_value=1, max_value=9, value=4, key="lc_sn")

        if st.button("🔮 Predict Lung Cancer Risk", type="primary", use_container_width=True):
            try:
                from src.models.predict_lung_cancer import predict as lc_predict

                input_data = {
                    "Age": age,
                    "Gender": gender,
                    "Air Pollution": air_pollution,
                    "Alcohol use": alcohol_use,
                    "Dust Allergy": dust_allergy,
                    "OccuPational Hazards": occupational_hazards,
                    "Genetic Risk": genetic_risk,
                    "chronic Lung Disease": chronic_lung_disease,
                    "Balanced Diet": balanced_diet,
                    "Obesity": obesity,
                    "Smoking": smoking,
                    "Passive Smoker": passive_smoker,
                    "Chest Pain": chest_pain_lc,
                    "Coughing of Blood": coughing_blood,
                    "Fatigue": fatigue,
                    "Weight Loss": weight_loss,
                    "Shortness of Breath": shortness_breath,
                    "Wheezing": wheezing,
                    "Swallowing Difficulty": swallowing_difficulty,
                    "Clubbing of Finger Nails": clubbing_nails,
                    "Frequent Cold": frequent_cold,
                    "Dry Cough": dry_cough,
                    "Snoring": snoring,
                }

                with st.spinner("Running prediction..."):
                    result = lc_predict(input_data, return_proba=True)

                st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                st.subheader("Prediction Result")

                pred_label = result.get("prediction", "Unknown")
                if "High" in str(pred_label):
                    st.markdown(
                        f'<p class="risk-high">⚠️ Risk Level: {pred_label}</p>',
                        unsafe_allow_html=True,
                    )
                    st.warning("High risk detected. Recommend immediate medical evaluation.")
                elif "Medium" in str(pred_label):
                    st.markdown(
                        f'<p class="risk-medium">⚡ Risk Level: {pred_label}</p>',
                        unsafe_allow_html=True,
                    )
                    st.info("Medium risk detected. Consider scheduling a medical check-up.")
                else:
                    st.markdown(
                        f'<p class="risk-low">✅ Risk Level: {pred_label}</p>',
                        unsafe_allow_html=True,
                    )
                    st.success("Low risk detected. Maintain healthy lifestyle habits.")

                _render_probabilities(result.get("probabilities", {}))
                st.markdown("</div>", unsafe_allow_html=True)

            except FileNotFoundError as e:
                st.error(str(e))
                st.info("Train the model first: `python src/models/train_lung_cancer.py`")
            except Exception as e:
                st.error(f"Prediction error: {e}")

    # ---------------------------------------------------------------- #
    # Batch prediction                                                  #
    # ---------------------------------------------------------------- #
    with tab2:
        st.subheader("Batch Prediction")
        st.markdown(
            """
            Upload a CSV file with patient records. Required columns match the
            23 features in `data/raw/lung_cancer.csv` (without the `Level` target column).
            """
        )

        uploaded_lc = st.file_uploader("Upload patient CSV", type="csv", key="lc_batch")
        if uploaded_lc is not None:
            try:
                df_upload = pd.read_csv(uploaded_lc)
                st.write(f"Uploaded {len(df_upload)} records. Preview:")
                st.dataframe(df_upload.head(), use_container_width=True)

                if st.button("🔮 Predict All", type="primary", key="lc_batch_predict"):
                    from src.models.predict_lung_cancer import predict_batch as lc_batch

                    with st.spinner("Processing..."):
                        results_df = lc_batch(df_upload)

                    st.success("Batch prediction complete!")
                    st.dataframe(results_df, use_container_width=True)

                    csv_out = results_df.to_csv(index=False)
                    st.download_button(
                        "📥 Download Predictions",
                        data=csv_out,
                        file_name="lung_cancer_predictions.csv",
                        mime="text/csv",
                    )
            except FileNotFoundError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"Error: {e}")


# ------------------------------------------------------------------ #
# Footer                                                               #
# ------------------------------------------------------------------ #
st.markdown("---")
st.markdown(
    """
    <div style='text-align:center;color:#888;font-size:0.85rem;'>
        MLOps Assignment — Health Risk Prediction System<br>
        Built with Streamlit · PyCaret · MLflow · Hydra
    </div>
    """,
    unsafe_allow_html=True,
)
