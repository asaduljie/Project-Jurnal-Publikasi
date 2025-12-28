import streamlit as st
import pandas as pd
import pickle

# =====================
# PAGE CONFIG
# =====================
st.set_page_config(
    page_title="AI Interior Design Recommendation",
    page_icon="🏠",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# =====================
# CUSTOM CSS (MODERN UI)
# =====================
st.markdown("""
<style>
    body {
        background-color: #0f1117;
        color: #ffffff;
    }
    .main {
        background-color: #0f1117;
    }
    .card {
        background: linear-gradient(145deg, #161a22, #0e1117);
        padding: 25px;
        border-radius: 16px;
        margin-bottom: 25px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.35);
    }
    .title {
        font-size: 38px;
        font-weight: 800;
        margin-bottom: 10px;
    }
    .subtitle {
        color: #9aa4b2;
        font-size: 16px;
        margin-bottom: 30px;
    }
    .label {
        font-size: 15px;
        font-weight: 600;
        margin-bottom: 6px;
    }
    .result {
        background-color: #173b2d;
        padding: 16px;
        border-radius: 12px;
        font-size: 20px;
        font-weight: 700;
        color: #6ef2b2;
        text-align: center;
    }
    .prompt-box {
        background-color: #111827;
        border-radius: 12px;
        padding: 18px;
        font-family: monospace;
        font-size: 14px;
        color: #e5e7eb;
    }
</style>
""", unsafe_allow_html=True)

# =====================
# LOAD MODEL
# =====================
model = pickle.load(open("model.pkl", "rb"))
encoders = pickle.load(open("encoders.pkl", "rb"))

# =====================
# LOAD DATA
# =====================
df = pd.read_csv("metadata.csv")
room_types = sorted(df["room_type"].unique())

# =====================
# HEADER
# =====================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="title">🏠 AI Interior Design Recommendation</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Machine Learning–based interior style recommendation with Generative AI prompt output</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# =====================
# INPUT SECTION
# =====================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="label">Select Room Type</div>', unsafe_allow_html=True)
room_type = st.selectbox("", room_types)
generate = st.button("✨ Generate Recommendation")
st.markdown('</div>', unsafe_allow_html=True)

# =====================
# MODEL FUNCTION
# =====================
def recommend_design(room_type):
    room_enc = encoders["room_type"].transform([room_type])[0]
    sample = pd.DataFrame([[room_enc]], columns=["room_type"])
    pred = model.predict(sample)[0]
    return encoders["style"].inverse_transform([pred])[0]

# =====================
# OUTPUT
# =====================
if generate:
    style = recommend_design(room_type)

    prompt = f"""
{style} {room_type} interior design,
modern minimalist style,
clean layout,
soft lighting,
high quality interior rendering,
ultra realistic
""".strip()

    # RESULT CARD
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Recommended Interior Style")
    st.markdown(f'<div class="result">{style}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # PROMPT CARD
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Generative AI Prompt (Stable Diffusion)")
    st.markdown(f'<div class="prompt-box">{prompt}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
