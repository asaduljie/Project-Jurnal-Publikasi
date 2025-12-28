import streamlit as st
import pandas as pd
import pickle

# =====================
# LOAD MODEL
# =====================
model = pickle.load(open("model.pkl", "rb"))
encoders = pickle.load(open("encoders.pkl", "rb"))

# =====================
# UI
# =====================
st.set_page_config(page_title="AI Interior Design Recommendation")
st.title("🏠 AI Interior Design Recommendation")

# Load dataset (only for room options)
df = pd.read_csv("metadata.csv")

room_types = sorted(df["room_type"].unique())
room_type = st.selectbox("Select Room Type", room_types)

# =====================
# RECOMMENDATION
# =====================
def recommend_design(room_type):
    room_enc = encoders["room_type"].transform([room_type])[0]
    sample = pd.DataFrame([[room_enc]], columns=["room_type"])
    pred = model.predict(sample)[0]
    return encoders["style"].inverse_transform([pred])[0]

if st.button("Generate Recommendation"):
    style = recommend_design(room_type)

    st.subheader("Recommended Style")
    st.success(style)

    # Prompt only (SAFE)
    prompt = f"""
    {style} {room_type} interior design,
    modern minimalist,
    clean layout,
    soft lighting,
    high quality interior rendering
    """

    st.subheader("Generative AI Prompt")
    st.code(prompt)
