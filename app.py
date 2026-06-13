import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings("ignore")

# ════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="➕ Alpha Care",
    page_icon="➕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ════════════════════════════════════════════════════════════════
# CSS — Electric Blue + White + Red medical theme
# ════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

/* ── Root ── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #EEF4FF !important;
}
[data-testid="stAppViewContainer"] > .main {
    background: linear-gradient(160deg, #EEF4FF 0%, #F8FAFF 60%, #EEF4FF 100%);
    font-family: 'Inter', sans-serif;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #3B82F6 !important;
    border-right: none !important;
}
[data-testid="stSidebar"] * {
    color: #ffffff !important;
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #ffffff !important;
    font-family: 'DM Serif Display', serif !important;
    letter-spacing: 0.01em;
}
[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.25) !important;
}
[data-testid="stSidebar"] .stInfo,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #dbeafe !important;
}

/* ── Header ── */
.ac-hero {
    background: linear-gradient(135deg, #1d4ed8 0%, #3B82F6 55%, #60a5fa 100%);
    border-radius: 20px;
    padding: 36px 44px;
    margin-bottom: 32px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: 0 12px 40px rgba(59,130,246,0.28);
    position: relative;
    overflow: hidden;
}
.ac-hero::before {
    content: '';
    position: absolute;
    top: -40px; right: -40px;
    width: 200px; height: 200px;
    background: rgba(255,255,255,0.06);
    border-radius: 50%;
}
.ac-hero::after {
    content: '';
    position: absolute;
    bottom: -60px; right: 120px;
    width: 280px; height: 280px;
    background: rgba(255,255,255,0.04);
    border-radius: 50%;
}
.ac-logo {
    font-family: 'DM Serif Display', serif;
    font-size: 3rem;
    color: #ffffff;
    line-height: 1;
    letter-spacing: -0.02em;
}
.ac-logo span {
    color: #ef4444;
    font-style: italic;
}
.ac-tagline {
    font-family: 'Inter', sans-serif;
    font-size: 0.95rem;
    color: #bfdbfe;
    margin-top: 6px;
    font-weight: 400;
    letter-spacing: 0.04em;
}
.ac-badge {
    background: rgba(255,255,255,0.12);
    border: 1.5px solid rgba(255,255,255,0.25);
    border-radius: 50px;
    padding: 10px 22px;
    color: #ffffff;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    font-family: 'JetBrains Mono', monospace;
}

/* ── Stat tiles ── */
.ac-tile {
    background: #ffffff;
    border: 1.5px solid #BFDBFE;
    border-radius: 16px;
    padding: 22px 18px;
    text-align: center;
    box-shadow: 0 2px 16px rgba(59,130,246,0.08);
}
.ac-tile-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.9rem;
    font-weight: 600;
    color: #1d4ed8;
    line-height: 1;
}
.ac-tile-val.red { color: #ef4444; }
.ac-tile-label {
    font-size: 0.7rem;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    margin-top: 5px;
    font-weight: 600;
}

/* ── Cards ── */
.ac-card {
    background: #ffffff;
    border: 1.5px solid #BFDBFE;
    border-radius: 18px;
    padding: 28px;
    margin-bottom: 20px;
    box-shadow: 0 2px 20px rgba(59,130,246,0.07);
}
.ac-card-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.15rem;
    color: #1d4ed8;
    margin-bottom: 18px;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* ── Result box ── */
.ac-result-positive {
    background: linear-gradient(135deg, #fee2e2, #fef2f2);
    border: 2px solid #ef4444;
    border-radius: 16px;
    padding: 28px;
    text-align: center;
    box-shadow: 0 4px 20px rgba(239,68,68,0.15);
}
.ac-result-negative {
    background: linear-gradient(135deg, #eff6ff, #dbeafe);
    border: 2px solid #3B82F6;
    border-radius: 16px;
    padding: 28px;
    text-align: center;
    box-shadow: 0 4px 20px rgba(59,130,246,0.15);
}
.ac-result-label {
    font-family: 'DM Serif Display', serif;
    font-size: 2rem;
    font-weight: 400;
    line-height: 1.1;
    margin-bottom: 8px;
}
.ac-result-label.positive { color: #dc2626; }
.ac-result-label.negative { color: #1d4ed8; }
.ac-prob {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2.8rem;
    font-weight: 600;
    color: #1d4ed8;
    line-height: 1;
}
.ac-prob.red { color: #ef4444; }
.ac-prob-label {
    font-size: 0.72rem;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    margin-top: 4px;
    font-weight: 600;
}

/* ── Input sliders & fields ── */
.stSlider > div > div > div {
    background: #3B82F6 !important;
}
.stNumberInput input {
    border: 1.5px solid #BFDBFE !important;
    border-radius: 10px !important;
    font-family: 'JetBrains Mono', monospace !important;
    color: #1d4ed8 !important;
}

/* ── Predict button ── */
.stButton > button {
    font-family: 'Inter', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    background: linear-gradient(135deg, #1d4ed8, #3B82F6) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 14px 32px !important;
    box-shadow: 0 4px 18px rgba(59,130,246,0.35) !important;
    width: 100% !important;
    letter-spacing: 0.04em !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    box-shadow: 0 6px 28px rgba(59,130,246,0.5) !important;
    transform: translateY(-1px) !important;
}

/* ── Section divider ── */
.ac-divider {
    height: 2px;
    background: linear-gradient(90deg, transparent, #3B82F6, transparent);
    border: none;
    margin: 28px 0;
}

/* ── Risk bar ── */
.risk-bar-wrap {
    background: #EEF4FF;
    border-radius: 999px;
    height: 10px;
    margin: 10px 0 4px 0;
    overflow: hidden;
}
.risk-bar-fill {
    height: 100%;
    border-radius: 999px;
    transition: width 0.5s ease;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab"] {
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    color: #6b7280 !important;
    font-size: 0.9rem !important;
}
.stTabs [aria-selected="true"] {
    color: #1d4ed8 !important;
    border-bottom-color: #3B82F6 !important;
}

/* ── Uploader ── */
[data-testid="stFileUploader"] {
    border: 2px dashed #BFDBFE !important;
    border-radius: 14px !important;
    background: #F8FAFF !important;
}

/* ── Footer ── */
.ac-footer {
    text-align: center;
    padding: 18px;
    color: #9ca3af;
    font-size: 0.78rem;
    border-top: 1.5px solid #BFDBFE;
    margin-top: 40px;
    font-family: 'Inter', sans-serif;
    letter-spacing: 0.05em;
}
.ac-footer span { color: #3B82F6; font-weight: 700; }
.ac-footer .red { color: #ef4444; }

/* ── Metrics ── */
[data-testid="stMetricValue"] { color: #1d4ed8 !important; font-family: 'JetBrains Mono', monospace !important; }
[data-testid="stMetricLabel"] { color: #6b7280 !important; font-size: 0.72rem !important; }

.main .block-container { padding-bottom: 80px !important; }
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# ORIGINAL CODE — UNTOUCHED (model training pipeline)
# ════════════════════════════════════════════════════════════════
@st.cache_resource
def train_pipeline():
    df = pd.read_csv('diabetes.csv')

    cols_with_zeros = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    df[cols_with_zeros] = df[cols_with_zeros].replace(0, np.nan)
    df.fillna(df.median(), inplace=True)

    X = df.drop('Outcome', axis=1)
    y = df['Outcome']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.8, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm     = confusion_matrix(y_test, y_pred)

    return model, scaler, df, acc, report, cm, X_test, y_test, y_pred

try:
    model, scaler, df, acc, report, cm, X_test, y_test, y_pred = train_pipeline()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False

# ════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='padding:24px 8px 16px 8px;'>
        <div style='font-family:"DM Serif Display",serif;font-size:1.6rem;
                    color:#ffffff;line-height:1;'>
            ➕ Alpha <span style='color:#fca5a5;font-style:italic;'>Care</span>
        </div>
        <div style='font-size:0.75rem;color:#bfdbfe;margin-top:6px;
                    font-family:"Inter",sans-serif;letter-spacing:0.08em;'>
            DIABETES RISK ASSESSMENT
        </div>
    </div>
    <hr>
    """, unsafe_allow_html=True)

    st.markdown("### 📋 Navigation")
    page = st.radio(
        "Go to",
        ["🔬 Risk Assessment", "📊 Model Performance", "📈 Data Insights"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown(
        "Alpha Care uses **Logistic Regression** trained on the Pima Indians "
        "Diabetes Dataset to assess diabetes risk from clinical measurements."
    )

    st.markdown("---")
    st.markdown("### 📁 Dataset")
    uploaded_csv = st.file_uploader("Upload diabetes.csv", type=["csv"])
    if uploaded_csv:
        st.session_state["uploaded_csv"] = uploaded_csv
        st.success("✅ File uploaded!")

    st.markdown("---")
    if model_loaded:
        st.success(f"✅ Model ready — {acc*100:.1f}% accuracy")
    else:
        st.error("❌ diabetes.csv not found. Upload it above.")

# ════════════════════════════════════════════════════════════════
# HERO HEADER
# ════════════════════════════════════════════════════════════════
st.markdown("""
<div class='ac-hero'>
    <div>
        <div class='ac-logo'>➕ Alpha <span>Care</span></div>
        <div class='ac-tagline'>Intelligent Diabetes Risk Assessment · Powered by Machine Learning</div>
    </div>
    <div class='ac-badge'>🩺 Clinical AI Tool</div>
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# STAT TILES
# ════════════════════════════════════════════════════════════════
if model_loaded:
    t1, t2, t3, t4 = st.columns(4)
    precision_1 = report.get("1", {}).get("precision", 0)
    recall_1    = report.get("1", {}).get("recall", 0)

    with t1:
        st.markdown(f"""<div class='ac-tile'>
            <div class='ac-tile-val'>{acc*100:.1f}%</div>
            <div class='ac-tile-label'>Model Accuracy</div>
        </div>""", unsafe_allow_html=True)
    with t2:
        st.markdown(f"""<div class='ac-tile'>
            <div class='ac-tile-val'>{precision_1*100:.1f}%</div>
            <div class='ac-tile-label'>Precision (Diabetic)</div>
        </div>""", unsafe_allow_html=True)
    with t3:
        st.markdown(f"""<div class='ac-tile'>
            <div class='ac-tile-val red'>{recall_1*100:.1f}%</div>
            <div class='ac-tile-label'>Recall (Diabetic)</div>
        </div>""", unsafe_allow_html=True)
    with t4:
        n_diabetic = int(df["Outcome"].sum())
        st.markdown(f"""<div class='ac-tile'>
            <div class='ac-tile-val'>{len(df):,}</div>
            <div class='ac-tile-label'>Training Records</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# PAGE: RISK ASSESSMENT
# ════════════════════════════════════════════════════════════════
if "Risk Assessment" in page:
    st.markdown("<div class='ac-card'><div class='ac-card-title'>🩺 Patient Clinical Parameters</div>", unsafe_allow_html=True)

    if not model_loaded:
        st.error("Upload **diabetes.csv** in the sidebar to activate the model.")
    else:
        col1, col2 = st.columns(2, gap="large")

        with col1:
            st.markdown("**Metabolic Indicators**")
            pregnancies    = st.number_input("Pregnancies", min_value=0, max_value=20, value=1, step=1)
            glucose        = st.slider("Glucose (mg/dL)", 44, 199, 120)
            blood_pressure = st.slider("Blood Pressure (mmHg)", 24, 122, 70)
            skin_thickness = st.slider("Skin Thickness (mm)", 7, 99, 23)

        with col2:
            st.markdown("**Body & Genetic Indicators**")
            insulin    = st.slider("Insulin (μU/mL)", 14, 846, 80)
            bmi        = st.slider("BMI", 18.2, 67.1, 32.0, step=0.1)
            dpf        = st.slider("Diabetes Pedigree Function", 0.078, 2.42, 0.47, step=0.001)
            age        = st.slider("Age (years)", 21, 81, 33)

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("🔬 Assess Diabetes Risk")

        if predict_btn:
            input_data = np.array([[pregnancies, glucose, blood_pressure,
                                    skin_thickness, insulin, bmi, dpf, age]])
            input_scaled = scaler.transform(input_data)
            prediction   = model.predict(input_scaled)[0]
            probability  = model.predict_proba(input_scaled)[0]
            risk_pct     = probability[1] * 100

            if prediction == 1:
                st.markdown(f"""
                <div class='ac-result-positive'>
                    <div class='ac-result-label positive'>⚠️ Diabetic Risk Detected</div>
                    <div style='color:#6b7280;font-size:0.85rem;margin-bottom:14px;'>
                        The model predicts a <b>positive</b> diabetes outcome based on the provided parameters.
                    </div>
                    <div class='ac-prob red'>{risk_pct:.1f}%</div>
                    <div class='ac-prob-label'>Probability of Diabetes</div>
                    <div class='risk-bar-wrap' style='margin-top:16px;'>
                        <div class='risk-bar-fill' style='width:{risk_pct}%;
                             background:linear-gradient(90deg,#f87171,#ef4444);'></div>
                    </div>
                    <div style='font-size:0.78rem;color:#9ca3af;margin-top:6px;'>
                        Please consult a qualified healthcare professional for diagnosis.
                    </div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='ac-result-negative'>
                    <div class='ac-result-label negative'>✅ Low Diabetes Risk</div>
                    <div style='color:#6b7280;font-size:0.85rem;margin-bottom:14px;'>
                        The model predicts a <b>negative</b> diabetes outcome based on the provided parameters.
                    </div>
                    <div class='ac-prob'>{probability[0]*100:.1f}%</div>
                    <div class='ac-prob-label'>Probability of No Diabetes</div>
                    <div class='risk-bar-wrap' style='margin-top:16px;'>
                        <div class='risk-bar-fill' style='width:{probability[0]*100}%;
                             background:linear-gradient(90deg,#60a5fa,#3B82F6);'></div>
                    </div>
                    <div style='font-size:0.78rem;color:#9ca3af;margin-top:6px;'>
                        Maintain a healthy lifestyle. Regular checkups are recommended.
                    </div>
                </div>""", unsafe_allow_html=True)

            # Feature importance
            st.markdown("<hr class='ac-divider'>", unsafe_allow_html=True)
            st.markdown("<div class='ac-card'><div class='ac-card-title'>📊 Feature Contribution to Prediction</div>", unsafe_allow_html=True)

            feature_names = ["Pregnancies", "Glucose", "BloodPressure",
                             "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
            coefs = model.coef_[0]
            contrib_df = pd.DataFrame({
                "Feature": feature_names,
                "Coefficient": coefs
            }).sort_values("Coefficient", key=abs, ascending=True)

            fig, ax = plt.subplots(figsize=(7, 4))
            colors = ["#ef4444" if c > 0 else "#3B82F6" for c in contrib_df["Coefficient"]]
            ax.barh(contrib_df["Feature"], contrib_df["Coefficient"], color=colors)
            ax.axvline(0, color="#1d4ed8", linewidth=1, linestyle="--", alpha=0.5)
            ax.set_xlabel("Coefficient Weight", fontsize=9, color="#6b7280")
            ax.tick_params(colors="#374151", labelsize=9)
            for spine in ax.spines.values():
                spine.set_color("#BFDBFE")
            fig.patch.set_facecolor("#F8FAFF")
            ax.set_facecolor("#F8FAFF")
            ax.set_title("Red = Increases Risk   |   Blue = Decreases Risk",
                         fontsize=8.5, color="#6b7280", pad=10)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            st.markdown("</div>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# PAGE: MODEL PERFORMANCE
# ════════════════════════════════════════════════════════════════
elif "Model Performance" in page:
    if not model_loaded:
        st.error("Upload **diabetes.csv** in the sidebar to view performance.")
    else:
        st.markdown("<div class='ac-card'><div class='ac-card-title'>🧮 Confusion Matrix</div>", unsafe_allow_html=True)

        # ── Original confusion matrix code (untouched) ────────────
        fig, ax = plt.subplots(figsize=(5, 4))
        cm_plot = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm_plot, annot=True, fmt="d", cmap="Blues",
                    xticklabels=[0, 1], yticklabels=[0, 1], ax=ax)
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("Actual", fontsize=10)
        ax.set_title("Confusion Matrix", fontsize=12)
        fig.patch.set_facecolor("#F8FAFF")
        ax.set_facecolor("#F8FAFF")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.markdown("</div>", unsafe_allow_html=True)

        # Classification report as table
        st.markdown("<div class='ac-card'><div class='ac-card-title'>📋 Classification Report</div>", unsafe_allow_html=True)
        report_df = pd.DataFrame(report).transpose().round(3)
        st.dataframe(
            report_df.style
            .background_gradient(cmap="Blues", subset=["precision", "recall", "f1-score"])
            .format(precision=3),
            use_container_width=True
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(f"""
        <div class='ac-tile' style='text-align:left;padding:20px 28px;'>
            <span style='font-family:"JetBrains Mono",monospace;color:#6b7280;
                         font-size:0.8rem;'>OVERALL ACCURACY</span>
            <div class='ac-tile-val' style='font-size:2.5rem;margin-top:4px;'>
                {acc*100:.2f}%
            </div>
        </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# PAGE: DATA INSIGHTS
# ════════════════════════════════════════════════════════════════
elif "Data Insights" in page:
    if not model_loaded:
        st.error("Upload **diabetes.csv** in the sidebar to view insights.")
    else:
        st.markdown("<div class='ac-card'><div class='ac-card-title'>📦 Insulin Distribution (Outlier Check)</div>", unsafe_allow_html=True)

        # ── Original boxplot code (untouched) ─────────────────────
        fig, ax = plt.subplots(figsize=(8, 3))
        sns.boxplot(x=df["Insulin"], ax=ax, color="#60a5fa")
        ax.set_title("Insulin Boxplot", fontsize=11)
        ax.tick_params(colors="#374151")
        for spine in ax.spines.values():
            spine.set_color("#BFDBFE")
        fig.patch.set_facecolor("#F8FAFF")
        ax.set_facecolor("#F8FAFF")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='ac-card'><div class='ac-card-title'>📊 Feature Distributions (All Variables)</div>", unsafe_allow_html=True)

        # ── Original multi-boxplot code (untouched) ───────────────
        cols_with_zeros = ["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]
        df[cols_with_zeros] = df[cols_with_zeros].replace(0, np.nan)
        df.fillna(df.median(), inplace=True)

        features = df.drop("Outcome", axis=1).columns

        fig = plt.figure(figsize=(15, 10))
        for i, col in enumerate(features, 1):
            plt.subplot(3, 3, i)
            sns.boxplot(y=df[col], color="skyblue")
            plt.title(col)
        plt.tight_layout()
        fig.patch.set_facecolor("#F8FAFF")
        st.pyplot(fig)
        plt.close()
        st.markdown("</div>", unsafe_allow_html=True)

        # Dataset preview
        st.markdown("<div class='ac-card'><div class='ac-card-title'>🗃️ Dataset Preview</div>", unsafe_allow_html=True)
        st.dataframe(
            df.head(20).style.background_gradient(cmap="Blues", subset=["Glucose", "BMI", "Insulin"]),
            use_container_width=True
        )
        st.markdown(f"""
        <div style='font-size:0.8rem;color:#6b7280;margin-top:8px;'>
            Showing 20 of <b>{len(df):,}</b> records · 
            <b>{df['Outcome'].sum()}</b> diabetic · 
            <b>{len(df)-df['Outcome'].sum()}</b> non-diabetic
        </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# FOOTER
# ════════════════════════════════════════════════════════════════
st.markdown("""
<div class='ac-footer'>
    ➕ <span>Alpha Care</span> · Diabetes Risk Assessment Tool ·
    <span class='red'>⚕</span> Not a substitute for professional medical advice ·
    Powered by <span>Logistic Regression + Streamlit</span>
</div>
""", unsafe_allow_html=True)