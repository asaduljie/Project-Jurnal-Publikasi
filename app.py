import streamlit as st
import pandas as pd
import pickle

# =====================
# LOAD MODEL & ENCODER
# =====================
model = pickle.load(open("model.pkl", "rb"))
encoders = pickle.load(open("encoders.pkl", "rb"))

# =====================
# UI CONFIG
# =====================
st.set_page_config(page_title="AI Interior Design Recommendation")
st.title("🏠 AI Interior Design Recommendation")

# =====================
# LOAD DATASET
# =====================
df = pd.read_csv("metadata.csv")
room_types = sorted(df["room_type"].unique())

room_type = st.selectbox("Select Room Type", room_types)

# =====================
# RECOMMENDATION LOGIC
# =====================
def recommend_design(room_type):
    room_enc = encoders["room_type"].transform([room_type])[0]

    # ⚠️ MODEL DILATIH HANYA DENGAN room_type
    X_input = pd.DataFrame([[room_enc]], columns=["room_type"])
    pred = model.predict(X_input)[0]

    return encoders["style"].inverse_transform([pred])[0]

# =====================
# BUTTON
# =====================
if st.button("Generate Recommendation"):
    style = recommend_design(room_type)

    st.subheader("Recommended Interior Style")
    st.success(style)

    prompt = f"""
    {style} {room_type} interior design,
    modern minimalist style,
    clean layout,
    soft lighting,
    high quality interior rendering,
    ultra realistic
    """

    st.subheader("Generative AI Prompt (for Stable Diffusion)")
    st.code(prompt)
