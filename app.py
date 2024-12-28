import json
import os
import random
import re
import tempfile
from io import BytesIO
from pathlib import Path

import streamlit as st
from google import genai
from google.genai import types
from PIL import Image, ImageColor, ImageDraw, ImageFont


def call_llm(img, prompt):
    """
    Call Gemini Vision API to analyze the image.
    Returns None if API call fails.
    """
    system_prompt = """
    Return bounding boxes as a JSON array with labels. Never return masks or code fencing. Limit to 25 objects.
    If an object is present multiple times, name them according to their unique characteristic (colors, size, position, unique characteristics, etc..).
    Output a json list where each entry contains the 2D bounding box in "box_2d" and a text label in "label".
    """

    try:
        client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        response = client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=[prompt, img],
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.5,
                safety_settings=[
                    types.SafetySetting(
                        category="HARM_CATEGORY_DANGEROUS_CONTENT",
                        threshold="BLOCK_ONLY_HIGH",
                    ),
                ],
            ),
        )
        return response.text
    except Exception as e:
        st.error(f"Error calling Gemini API: {str(e)}")
        return None


def parse_json(json_input):
    """
    Parse JSON from LLM response, handling potential code fence markers.
    """
    try:
        # Try direct JSON parsing first
        return json.loads(json_input)
    except json.JSONDecodeError:
        # Fall back to regex parsing if direct parse fails
        match = re.search(r"```json\n(.*?)```", json_input, re.DOTALL)
        if not match:
            raise ValueError("No valid JSON found in response")
        return json.loads(match.group(1))


def plot_bounding_boxes(img, bounding_boxes):
    """
    Draw bounding boxes and labels on the image with larger text.
    """
    width, height = img.size
    colors = list(ImageColor.colormap.keys())  # Convert to list once
    draw = ImageDraw.Draw(img)

    # Create larger text size using default font
    font = ImageFont.load_default()
    # Scale up text by creating a new image with text and resizing it
    scale_factor = 2  # Increase this to make text larger
    
    try:
        boxes_data = parse_json(bounding_boxes)
    except (json.JSONDecodeError, ValueError) as e:
        st.error(f"Error parsing bounding box data: {str(e)}")
        return img

    for box in boxes_data:
        color = random.choice(colors)
        
        # Convert normalized coordinates to absolute
        y1, x1, y2, x2 = box["box_2d"]
        abs_y1 = int(y1 / 1000 * height)
        abs_x1 = int(x1 / 1000 * width)
        abs_y2 = int(y2 / 1000 * height)
        abs_x2 = int(x2 / 1000 * width)
        
        # Ensure correct coordinate order
        x1, x2 = min(abs_x1, abs_x2), max(abs_x1, abs_x2)
        y1, y2 = min(abs_y1, abs_y2), max(abs_y1, abs_y2)

        # Draw thicker rectangle
        draw.rectangle(((x1, y1), (x2, y2)), outline=color, width=6)
        
        # Draw label background for better visibility
        label = box["label"]
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_width = (text_bbox[2] - text_bbox[0]) * scale_factor
        text_height = (text_bbox[3] - text_bbox[1]) * scale_factor
        
        # Draw white background behind text for better visibility
        draw.rectangle(
            ((x1, y1 - text_height - 4), (x1 + text_width + 8, y1)),
            fill='white'
        )
        
        # Create separate image for text, scale it up, and paste it
        text_img = Image.new('RGBA', (text_width + 8, text_height + 4), (255, 255, 255, 0))
        text_draw = ImageDraw.Draw(text_img)
        text_draw.text((4, 2), label, font=font, fill=color)
        text_img = text_img.resize(
            (int(text_width * scale_factor), int(text_height * scale_factor)),
            Image.Resampling.LANCZOS
        )
        img.paste(text_img, (x1, y1 - text_height - 4), text_img)

    return img


def process_uploaded_image(uploaded_file):
    """
    Process uploaded image file and create temporary file.
    """
    suffix = f".{uploaded_file.type.split('/')[1]}"
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp_file.write(uploaded_file.getbuffer())
    
    img = Image.open(temp_file.name)
    width, height = img.size
    resized_image = img.resize(
        (1024, int(1024 * height / width)), 
        Image.Resampling.LANCZOS
    )
    
    return resized_image, Path(temp_file.name)


def main():
    st.set_page_config(
        page_title="ObjectSight AI",
        page_icon="👁️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.header("👁️ ObjectSight AI")
    
    # Add API key input in sidebar
    with st.sidebar:
        api_key = st.text_input("Enter your Google API Key", type="password")
        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key
            
        st.divider()  # Add a visual separator
    
    prompt = st.text_input(
        "Enter your prompt",
        placeholder="Example: Identify and locate all objects in this image"
    )
    run = st.button("Run!")

    with st.sidebar:
        uploaded_image = st.file_uploader(
            accept_multiple_files=False,
            label="Upload your photo here:",
            type=["jpg", "jpeg", "png"],
        )

        if uploaded_image:
            with st.expander("View the image"):
                st.image(uploaded_image)

    if uploaded_image and run and prompt:
        if not api_key:
            st.error("Please enter your Google API Key in the sidebar")
            return
            
        resized_image, temp_path = process_uploaded_image(uploaded_image)

        with st.spinner("Running..."):
            response = call_llm(resized_image, prompt)
            if response:
                plotted_image = plot_bounding_boxes(resized_image, response)
                st.image(plotted_image)
                
                # Add download button for the analyzed image
                buffered = BytesIO()
                plotted_image.save(buffered, format="PNG")
                st.download_button(
                    "📥 Download Analyzed Image",
                    buffered.getvalue(),
                    "analyzed_image.png",
                    "image/png"
                )
                
                st.balloons()
            
        # Clean up the temporary file
        try:
            os.unlink(temp_path)
        except Exception as e:
            print(f"Could not delete temporary file: {e}")  # Non-critical error


if __name__ == "__main__":
    main()