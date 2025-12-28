import streamlit as st
import pandas as pd
import pickle

# =====================
# PAGE CONFIG
# =====================
st.set_page_config(
    page_title="AI Interior Design",
    page_icon="🏡",
    layout="centered",
)

# =====================
# CUSTOM CSS (STARTUP LOOK)
# =====================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

body {
    background: radial-gradient(circle at top, #0f172a, #020617);
}

/* Remove Streamlit header/footer */
#MainMenu, footer, header {visibility: hidden;}

/* Hero Title */
.hero-title {
    font-size: 3rem;
    font-weight: 800;
    color: white;
    line-height: 1.1;
}

.hero-subtitle {
    font-size: 1.1rem;
    color: #94a3b8;
    margin-top: 0.5rem;
}

/* Glass Card */
.glass {
    background: rgba(255,255,255,0.06);
    backdrop-filter: blur(14px);
    border-radius: 18px;
    padding: 28px;
    border: 1px solid rgba(255,255,255,0.08);
    margin-top: 24px;
}

/* Button */
.stButton>button {
    width: 100%;
    background: linear-gradient(135deg, #22c55e, #16a34a);
    color: black;
    border: none;
    border-radius: 12px;
    padding: 14px;
    font-size: 1rem;
    font-weight: 700;
    transition: 0.2s;
}

.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 30px rgba(34,197,94,0.35);
}

/* Success badge */
.badge {
    background: linear-gradient(135deg, #16a34a, #22c55e);
    color: #022c22;
    padding: 16px;
    border-radius: 14px;
    font-size: 1.3rem;
    font-weight: 700;
    text-align: center;
}

/* Prompt box */
.prompt-box {
    background: #020617;
    border: 1px solid #1e293b;
    border-radius: 14px;
    padding: 18px;
    font-family: monospace;
    color: #e5e7eb;
    font-size: 0.95rem;
}

/* Section title */
.section-title {
    color: white;
    font-weight: 700;
    margin-bottom: 8px;
    margin-top: 28px;
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
# HERO SECTION
# =====================
st.markdown("""
<div>
  <div class="hero-title">🏡 AI Interior Design</div>
  <div class="hero-subtitle">
    Machine Learning powered interior style recommendation<br>
    with Generative AI prompt generation
  </div>
</div>
""", unsafe_allow_html=True)

# =====================
# INPUT CARD
# =====================
st.markdown('<div class="glass">', unsafe_allow_html=True)

room_types = sorted(df["room_type"].unique())
room_type = st.selectbox("Room Type", room_types)

st.markdown('</div>', unsafe_allow_html=True)

# =====================
# RECOMMENDATION LOGIC
# =====================
def recommend_design(room_type):
    room_enc = encoders["room_type"].transform([room_type])[0]
    sample = pd.DataFrame([[room_enc]], columns=["room_type"])
    pred = model.predict(sample)[0]
    return encoders["style"].inverse_transform([pred])[0]

if st.button("✨ Generate AI Recommendation"):
    style = recommend_design(room_type)

    # =====================
    # RESULT CARD
    # =====================
    st.markdown('<div class="glass">', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Recommended Interior Style</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="badge">{style.upper()}</div>', unsafe_allow_html=True)

    prompt = f"""
{style} {room_type} interior design,
modern minimalist style,
clean layout,
soft lighting,
high quality interior rendering,
ultra realistic
""".strip()

    st.markdown('<div class="section-title">Generative AI Prompt (Stable Diffusion)</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="prompt-box">{prompt}</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# =====================
# FOOTNOTE
# =====================
st.markdown("""
<div style="text-align:center; color:#64748b; font-size:0.85rem; margin-top:40px;">
Built with Machine Learning & Generative AI • Research Prototype
</div>
""", unsafe_allow_html=True)
