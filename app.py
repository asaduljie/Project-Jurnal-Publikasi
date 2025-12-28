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
# GLOBAL CSS (WAJIB DI ATAS)
# =====================
st.markdown("""
<style>

/* ===== BACKGROUND ===== */
html, body, [data-testid="stApp"] {
    background: radial-gradient(circle at top, #111827, #020617);
    color: #E5E7EB;
    font-family: 'Inter', sans-serif;
}

/* ===== MAIN CONTAINER ===== */
.block-container {
    padding-top: 3rem;
    max-width: 860px;
}

/* ===== TITLES ===== */
h1 {
    font-size: 3rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.04em;
}

h2, h3 {
    font-weight: 700;
}

/* ===== SELECTBOX ===== */
div[data-baseweb="select"] > div {
    background: rgba(255,255,255,0.05);
    border-radius: 14px;
    border: 1px solid rgba(255,255,255,0.08);
}

/* ===== BUTTON ===== */
button[kind="primary"] {
    background: linear-gradient(135deg, #22c55e, #16a34a);
    color: #022c22;
    border-radius: 14px;
    height: 3.2rem;
    font-weight: 700;
    font-size: 1rem;
    border: none;
}

button[kind="primary"]:hover {
    transform: scale(1.02);
    box-shadow: 0 0 30px rgba(34,197,94,0.35);
}

/* ===== GLASS CARD ===== */
.card {
    background: rgba(255,255,255,0.04);
    backdrop-filter: blur(16px);
    border-radius: 20px;
    padding: 1.6rem;
    border: 1px solid rgba(255,255,255,0.08);
    margin-top: 1.5rem;
}

/* ===== STYLE BADGE ===== */
.style-badge {
    background: linear-gradient(135deg, #16a34a, #22c55e);
    color: #022c22;
    padding: 0.8rem 1.2rem;
    border-radius: 999px;
    font-weight: 800;
    font-size: 1.1rem;
    display: inline-block;
}

/* ===== CODE PROMPT ===== */
pre {
    background: rgba(0,0,0,0.5) !important;
    border-radius: 18px !important;
    padding: 1.2rem !important;
    font-size: 0.95rem !important;
    border: 1px solid rgba(255,255,255,0.08);
}

/* ===== REMOVE STREAMLIT FOOTER ===== */
footer, header {visibility: hidden;}

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
st.markdown("""
<h1>🏠 AI Interior Design<br/>Recommendation</h1>
<p style="opacity:0.7; font-size:1.1rem;">
Machine Learning × Generative AI for personalized interior concepts
</p>
""", unsafe_allow_html=True)

# =====================
# INPUT CARD
# =====================
st.markdown('<div class="card">', unsafe_allow_html=True)

room_type = st.selectbox(
    "Select Room Type",
    sorted(df["room_type"].unique())
)

generate = st.button("Generate Recommendation", type="primary")

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
    st.markdown("### Recommended Interior Style")
    st.markdown(f'<span class="style-badge">{style}</span>', unsafe_allow_html=True)
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
    st.markdown("### Generative AI Prompt (Stable Diffusion)")
    st.code(prompt)
    st.markdown('</div>', unsafe_allow_html=True)
