# ==============================================================================
# Scene-Sync AI - Main Application (Vlogger Edition)
#
# Author: Your Name
# Date: 2025-10-28
#
# Description: This version is generalized for all vloggers and content creators,
#              removing specific brand references from the UI.
#              This version's UI is in English.
# ==============================================================================

import os
import torch
import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from dotenv import load_dotenv
import google.generativeai as genai  # Import Google's API library

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Scene-Sync AI", page_icon="🎵", layout="centered")

# --- MODEL LOADING ---
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

def get_music_recommendations(scene_description):
    """
    Generates music recommendations using the Google Gemini Pro API.
    """
    print("[INFO] Contacting Google Gemini API for music recommendations...")
    
    try:
        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            return "ERROR: Google API Key not found in .env file."
        
        genai.configure(api_key=api_key)

        # Use the latest reliable model
        model = genai.GenerativeModel('gemini-2.5-flash-preview-09-2025')
        
        # --- UI TEXT CHANGE IN PROMPT (English) ---
        # The prompt has been updated to be more general.
        prompt = f"""
        As a professional Music Director for Vloggers and content creators, your task is to recommend background music for a video scene.
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

        response = model.generate_content(prompt)
        print("[SUCCESS] Recommendations received from Google Gemini.")
        
        return response.text
    except Exception as e:
        print(f"[ERROR] An exception occurred: {e}")
        return f"ERROR: An error occurred while calling the Google Gemini API. Please check your API key and network connection. Details: {e}"

# --- STREAMLIT UI DEFINITION ---
# --- UI TEXT CHANGES HERE (English) ---
st.title("🎵 Scene-Sync AI")
st.markdown("##### — Your AI Smart Music Assistant —") # <-- Text Updated
st.info("Upload an image that represents your video scene, and the AI will recommend the perfect background music.") # <-- Text Updated
# --- End of UI Text Changes ---

processor, model = load_models()

# --- UI TEXT CHANGES HERE (English) ---
uploaded_file = st.file_uploader("Choose an image file...", type=["jpg", "jpeg", "png"]) # <-- Text Updated

if uploaded_file is not None:
    with st.spinner('AI is analyzing the scene and finding music...') : # <-- Text Updated
        caption, raw_image = generate_caption(uploaded_file, processor, model)
        recommendations = get_music_recommendations(caption)

    st.subheader("🖼️ Your Uploaded Scene") # <-- Text Updated
    st.image(raw_image, use_column_width=True)
    
    st.subheader("🤖 AI Scene Analysis") # <-- Text Updated
    st.write(f"> {caption}")
    
    st.subheader("🎶 Your Custom Soundtrack") # <-- Text Updated
    st.markdown(recommendations)

# --- SIDEBAR ---
# --- UI TEXT CHANGES HERE (English) ---
st.sidebar.header("About Scene-Sync AI") # <-- Text Updated
st.sidebar.info(
    "This project aims to provide smart music suggestions for video creators and Vloggers. "
    "It uses AI image understanding to analyze visual content and then leverages the creative power of a Large Language Model (LLM) "
    "to generate professional music recommendation lists, greatly simplifying the content creation workflow."
) # <-- Text Updated
st.sidebar.success("Powered by: Google Gemini API")
# --- End of UI Text Changes ---

