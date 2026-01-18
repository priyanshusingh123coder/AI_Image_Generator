import streamlit as st
import torch
from diffusers import StableDiffusionPipeline
import os

st.set_page_config(page_title="AI Image Generator")

st.title("AI Image Generator")
prompt = st.text_input("Enter your image prompt")

@st.cache_resource
def load_model():
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    return pipe

pipe = load_model()

if st.button("Generate Image"):
    if prompt.strip() == "":
        st.warning("Please enter a prompt")
    else:
        with st.spinner("Generating image... Please wait ⏳"):
            image = pipe(
                prompt,
                guidance_scale=7.5,
                num_inference_steps=30
            ).images[0]

            os.makedirs("outputs", exist_ok=True)
            image.save("outputs/result.png")
            st.image(image, caption="Generated Image")
            st.success("Image generated and saved successfully!")
