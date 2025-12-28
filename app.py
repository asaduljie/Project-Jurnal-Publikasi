import streamlit as st
import pandas as pd
import pickle

# =====================
# PAGE CONFIG
# =====================
st.set_page_config(
    page_title="AI Interior Design",
    page_icon="🏠",
    layout="wide",
)

# =====================
# CUSTOM CSS (🔥 WAJIB)
# =====================
st.markdown("""
<style>

/* ---------- GLOBAL ---------- */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background: radial-gradient(circle at top left, #111827, #020617);
    color: #E5E7EB;
}

/* Remove default padding */
.block-container {
    padding-top: 3rem;
    padding-bottom: 3rem;
    max-width: 1100px;
}

/* ---------- TITLE ---------- */
.title {
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(90deg, #22c55e, #06b6d4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
}

.subtitle {
    color: #9CA3AF;
    font-size: 1.1rem;
    margin-bottom: 2.5rem;
}

/* ---------- CARD ---------- */
.card {
    background: rgba(255,255,255,0.04);
    backdrop-filter: blur(12px);
    border-radius: 18px;
    padding: 28px;
    margin-bottom: 24px;
    border: 1px solid rgba(255,255,255,0.08);
}

/* ---------- BUTTON ---------- */
.stButton>button {
    width: 100%;
    border-radius: 14px;
    padding: 14px;
    font-weight: 600;
    background: linear-gradient(90deg, #22c55e, #06b6d4);
    color: black;
    border: none;
    transition: all 0.25s ease;
}

.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 30px rgba(34,197,94,0.25);
}

/* ---------- SELECTBOX ---------- */
div[data-baseweb="select"] {
    background-color: #020617 !important;
    border-radius: 12px;
}

/* ---------- SUCCESS ---------- */
.success-box {
    background: linear-gradient(90deg, rgba(34,197,94,0.15), rgba(6,182,212,0.15));
    border: 1px solid rgba(34,197,94,0.4);
    border-radius: 14px;
    padding: 18px;
    font-size: 1.2rem;
    font-weight: 600;
    color: #22c55e;
}

/* ---------- PROMPT BOX ---------- */
.prompt-box {
    background: #020617;
    border-radius: 14px;
    padding: 20px;
    border: 1px solid rgba(255,255,255,0.08);
    font-family: monospace;
    line-height: 1.6;
    color: #E5E7EB;
}

/* ---------- FOOTER ---------- */
.footer {
    text-align: center;
    color: #6B7280;
    font-size: 0.9rem;
    margin-top: 60px;
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
st.markdown('<div class="title">🏠 AI Interior Design</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Smart interior style recommendation powered by Machine Learning & Generative AI</div>',
    unsafe_allow_html=True
)

# =====================
# INPUT CARD
# =====================
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)

    room_type = st.selectbox("Select Room Type", room_types)

    st.markdown("<br>", unsafe_allow_html=True)
    generate = st.button("✨ Generate Recommendation")

    st.markdown('</div>', unsafe_allow_html=True)

# =====================
# PREDICTION LOGIC
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

    # Style Output
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Recommended Interior Style")
    st.markdown(f'<div class="success-box">{style}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Prompt Output
    prompt = f"""
{style} {room_type} interior design,
modern minimalist style,
clean layout,
soft lighting,
high quality interior rendering,
ultra realistic
""".strip()

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Generative AI Prompt (Stable Diffusion)")
    st.markdown(f'<div class="prompt-box">{prompt}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# =====================
# FOOTER
# =====================
st.markdown(
    '<div class="footer">© 2025 AI Interior Design · Research Prototype</div>',
    unsafe_allow_html=True
)
