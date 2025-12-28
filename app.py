import streamlit as st
import pandas as pd
import pickle
import torch

from diffusers import StableDiffusionPipeline

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(
    page_title="AI Interior Design Generator",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 AI Interior Design Recommendation & Generation")
st.write(
    "This application recommends an interior design style using Machine Learning "
    "and generates an interior image using Generative AI."
)

# =============================
# LOAD MODEL & ENCODER
# =============================
@st.cache_resource
def load_ml_assets():
    model = pickle.load(open("model.pkl", "rb"))
    encoders = pickle.load(open("encoders.pkl", "rb"))
    return model, encoders

model, encoders = load_ml_assets()

# =============================
# LOAD DATA (OPTIONAL)
# =============================
df = pd.read_csv("metadata.csv")

room_types = sorted(df["room_type"].unique())

# =============================
# SIDEBAR INPUT
# =============================
st.sidebar.header("🔧 User Preference")

room_type = st.sidebar.selectbox(
    "Select Room Type",
    room_types
)

# =============================
# ML RECOMMENDATION FUNCTION
# =============================
def recommend_design(room_type):
    room_enc = encoders["room_type"].transform([room_type])[0]
    sample = pd.DataFrame([[room_enc]], columns=["room_type"])
    style_pred = model.predict(sample)[0]
    return encoders["style"].inverse_transform([style_pred])[0]

# =============================
# PROMPT GENERATOR
# =============================
def generate_prompt(style, room_type):
    return f"""
    {style} {room_type} interior design,
    modern minimalist style,
    clean layout,
    soft lighting,
    high quality interior rendering,
    ultra realistic,
    professional interior photography
    """

# =============================
# LOAD GENERATIVE AI MODEL
# =============================
@st.cache_resource
def load_diffusion():
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16
    )
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    return pipe

pipe = load_diffusion()

# =============================
# MAIN PROCESS
# =============================
st.subheader("📌 AI Recommendation Result")

if st.button("🚀 Generate Interior Design"):
    with st.spinner("Running Machine Learning model..."):
        style = recommend_design(room_type)

    st.success(f"Recommended Design Style: **{style}**")

    prompt = generate_prompt(style, room_type)

    st.markdown("### 🎨 Generative AI Prompt")
    st.code(prompt, language="text")

    with st.spinner("Generating interior image using Generative AI..."):
        image = pipe(prompt, guidance_scale=7.5).images[0]

    st.markdown("### 🖼️ Generated Interior Design")
    st.image(image, caption=f"{style} {room_type} Interior", use_container_width=True)

# =============================
# FOOTER
# =============================
st.markdown("---")
st.caption(
    "AI Interior Design System | Machine Learning + Generative AI | Academic Prototype"
)
