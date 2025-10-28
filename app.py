# ==============================================================================
# Scene-Sync AI - Main Application (Google Gemini Version - Final)
#
# Author: Your Name
# Date: 2025-10-28
#
# Description: This script uses the latest 'gemini-2.5-flash-preview-09-2025' model
#              which is the most current and widely available free model.
# ==============================================================================

import os
import torch
import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from dotenv import load_dotenv
import google.generativeai as genai  # Import Google's API library

# --- PAGE CONFIGURATION ---
# Configure the browser tab's title, icon, and the layout of the page.
st.set_page_config(page_title="Scene-Sync AI", page_icon="🎵", layout="centered")

# --- MODEL LOADING ---
# Use Streamlit's caching decorator to load the models only once.
@st.cache_resource
def load_models():
    """
    Loads and caches the BLIP processor and model for image captioning.
    """
    print("[INFO] Caching models: Loading BLIP processor and model for the first time.")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
    print("[INFO] Models successfully cached.")
    return processor, model

# --- CORE LOGIC FUNCTIONS ---
def generate_caption(image_data, _processor, _model):
    """
    Generates a caption from uploaded image data using the BLIP model.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _model.to(device)
    
    raw_image = Image.open(image_data).convert('RGB')
    inputs = _processor(raw_image, return_tensors="pt").to(device)
    out = _model.generate(**inputs, max_new_tokens=50)
    caption = _processor.decode(out[0], skip_special_tokens=True)
    return caption, raw_image

# --- THIS IS THE CORRECTED FUNCTION ---
def get_music_recommendations(scene_description):
    """
    Generates music recommendations using the Google Gemini Pro API.
    This function now uses the 'gemini-2.5-flash-preview-09-2025' model name.
    """
    print("[INFO] Contacting Google Gemini API for music recommendations...")
    
    try:
        # Load environment variables from the .env file
        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            return "ERROR: Google API Key not found in .env file."
        
        # Configure the Google AI client library
        genai.configure(api_key=api_key)

        # Initialize the Gemini Pro model WITH THE CORRECT, LATEST NAME
        # This is the FINAL FIX: Changed to 'gemini-2.5-flash-preview-09-2025'
        model = genai.GenerativeModel('gemini-2.5-flash-preview-09-2025')
        
        # The prompt is engineered for quality output
        prompt = f"""
        As a professional Music Director for Sony's content creators, your task is to recommend background music for a video scene.
        Based on the following scene description, provide 3 diverse song recommendations. The output MUST be in a clean, structured Markdown format as specified below. Do not include any preamble or extra text.

        Scene Description: "{scene_description}"

        ---
        ### 🎵 Music Recommendation List

        **1. Song Title**
        * **Artist:** [Artist Name]
        * **Mood Tags:** [e.g., Uplifting, Serene, Epic]
        * **Use Case:** [e.g., Travel Vlog Intro, Timelapse]
        * **Reasoning:** [Briefly explain why this track fits]

        **2. Song Title**
        * **Artist:** [Artist Name]
        * **Mood Tags:** [e.g., Uplifting, Serene, Epic]
        * **Use Case:** [e.g., Travel Vlog Intro, Timelapse]
        * **Reasoning:** [Briefly explain why this track fits]

        **3. Song Title**
        * **Artist:** [Artist Name]
        * **Mood Tags:** [e.g., Uplifting, Serene, Epic]
        * **Use Case:** [e.g., Travel Vlog Intro, Timelapse]
        * **Reasoning:** [Briefly explain why this track fits]
        """

        # Generate the content
        response = model.generate_content(prompt)
        print("[SUCCESS] Recommendations received from Google Gemini.")
        
        # Return the generated text
        return response.text
    except Exception as e:
        # Provide a more informative error message for debugging
        print(f"[ERROR] An exception occurred: {e}")
        return f"ERROR: An error occurred while calling the Google Gemini API. Please check your API key and network connection. Details: {e}"

# --- STREAMLIT UI DEFINITION ---
st.title("🎵 Scene-Sync AI")
st.markdown("##### — Your Personal AI Music Director for Sony Creators —")
st.info("Upload an image that represents your video's scene, and let the AI find the perfect soundtrack for you.")

# Load the models using the cached function.
processor, model = load_models()

# The file uploader widget.
uploaded_file = st.file_uploader("Choose an image file...", type=["jpg", "jpeg", "png"])

# Main application logic flow.
if uploaded_file is not None:
    with st.spinner('Analyzing scene and curating music... This may take a moment.'):
        # 1. Generate image caption.
        caption, raw_image = generate_caption(uploaded_file, processor, model)
        
        # 2. Get music recommendations (now using corrected model name).
        recommendations = get_music_recommendations(caption)

    # --- Display Results ---
    st.subheader("🖼️ Your Uploaded Scene")
    st.image(raw_image, use_column_width=True)
    
    st.subheader("🤖 AI Scene Analysis")
    st.write(f"> {caption}")
    
    st.subheader("🎶 Your Custom Soundtrack")
    st.markdown(recommendations)

# --- SIDEBAR ---
st.sidebar.header("About Scene-Sync AI")
st.sidebar.info(
    "This prototype demonstrates a powerful workflow for content creators. "
    "It leverages a multimodal AI pipeline to bridge the gap between visual content and audio selection, "
    "dramatically speeding up the creative process for Sony Alpha users and Vloggers."
)
st.sidebar.success("Powered by: Google Gemini API")

