import streamlit as st
import pandas as pd
import pickle

# =====================
# PAGE CONFIG
# =====================
st.set_page_config(
    page_title="AI Interior Design",
    page_icon="🏠",
    layout="wide"
)

# =====================
# CUSTOM CSS (CLEAN + STARTUP)
# =====================
st.markdown("""
<style>
body {
    background: radial-gradient(circle at top, #0f172a, #020617);
    color: white;
}
.block-container {
    padding-top: 3rem;
    max-width: 1100px;
}
h1, h2, h3 {
    font-weight: 700;
}
.card {
    background: rgba(255,255,255,0.05);
    border-radius: 18px;
    padding: 24px;
    margin-bottom: 24px;
}
.badge {
    display: inline-block;
    padding: 10px 18px;
    border-radius: 999px;
    background: linear-gradient(90deg, #22c55e, #16a34a);
    font-weight: 600;
    color: #022c22;
}
button[kind="primary"] {
    background: linear-gradient(90deg, #22c55e, #16a34a);
    border-radius: 12px;
    font-weight: 600;
}
textarea {
    background-color: #020617 !important;
    color: #e5e7eb !important;
    border-radius: 12px !important;
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
st.markdown("""
<h1>🏠 AI Interior Design Recommendation</h1>
<p style="color:#94a3b8">
Machine Learning × Generative AI for personalized interior concepts
</p>
""", unsafe_allow_html=True)

# =====================
# INPUT CARD
# =====================
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)

    room_type = st.selectbox(
        "Select Room Type",
        sorted(df["room_type"].unique())
    )

    generate = st.button("Generate Recommendation", type="primary")

    st.markdown('</div>', unsafe_allow_html=True)

# =====================
# RECOMMENDATION LOGIC
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

    st.markdown("## Recommended Interior Style")
    st.markdown(f'<span class="badge">{style}</span>', unsafe_allow_html=True)

    prompt = f"""
{style} {room_type} interior design,
modern minimalist style,
clean layout,
soft lighting,
high quality interior rendering,
ultra realistic
""".strip()

    st.markdown("## Generative AI Prompt (Stable Diffusion / Midjourney)")
    st.text_area("", prompt, height=180)
