import streamlit as st
import pandas as pd
import pickle

# =====================
# PAGE CONFIG
# =====================
st.set_page_config(
    page_title="AI Interior Design",
    page_icon="🏠",
    layout="centered"
)

# =====================
# CUSTOM CSS
# =====================
st.markdown("""
<style>
html, body {
    background: radial-gradient(circle at top, #0f2027, #0b0f14 60%);
    color: #f5f7fa;
    font-family: 'Inter', sans-serif;
}

.block-container {
    padding-top: 3rem;
    max-width: 900px;
}

.card {
    background: rgba(255,255,255,0.06);
    backdrop-filter: blur(14px);
    border-radius: 18px;
    padding: 28px;
    margin-bottom: 28px;
    border: 1px solid rgba(255,255,255,0.08);
}

.title {
    font-size: 3rem;
    font-weight: 800;
    letter-spacing: -1px;
}

.subtitle {
    color: #aab1c3;
    margin-bottom: 2rem;
}

.success-box {
    background: linear-gradient(135deg, #1f8f5f, #26c281);
    color: #fff;
    padding: 18px;
    border-radius: 14px;
    font-size: 1.2rem;
    font-weight: 600;
}

textarea {
    background-color: #0f172a !important;
    color: #e5e7eb !important;
    border-radius: 14px !important;
}

button[kind="primary"] {
    background: linear-gradient(135deg, #2563eb, #4f46e5);
    border-radius: 12px;
    font-weight: 600;
    padding: 0.6rem 1.4rem;
}
</style>
""", unsafe_allow_html=True)

# =====================
# LOAD MODEL
# =====================
model = pickle.load(open("model.pkl", "rb"))
encoders = pickle.load(open("encoders.pkl", "rb"))

df = pd.read_csv("metadata.csv")

# =====================
# HEADER
# =====================
st.markdown('<div class="title">🏠 AI Interior Design</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Smart interior style recommendation powered by Machine Learning & Generative AI</div>', unsafe_allow_html=True)

# =====================
# INPUT CARD
# =====================
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)

    room_type = st.selectbox(
        "Select Room Type",
        sorted(df["room_type"].unique())
    )

    generate = st.button("✨ Generate Recommendation", type="primary")

    st.markdown('</div>', unsafe_allow_html=True)

# =====================
# LOGIC
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

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🎯 Recommended Interior Style")
    st.markdown(f'<div class="success-box">{style.upper()}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    prompt = f"""
{style} {room_type} interior design,
modern minimalist style,
clean layout,
soft lighting,
high quality interior rendering,
ultra realistic
"""

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🎨 Generative AI Prompt (Stable Diffusion)")
    st.code(prompt.strip())
    st.markdown('</div>', unsafe_allow_html=True)
