import streamlit as st
from PIL import Image
from groq import Groq
from dotenv import load_dotenv
import os
import base64
import io

# Load environment variables
load_dotenv()
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
client = Groq()

def encode_image_to_base64(image):
    buffered = io.BytesIO()
    # Convert the image to RGB if it's in a format like "RGBA" or "P" (e.g., GIFs with transparency)
    if image.mode in ("RGBA", "P"):
        image = image.convert("RGB")
    # Save image to buffer as JPEG or PNG depending on original format
    image.save(buffered, format="JPEG")  # JPEG used here as a default format
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

# OCR function to extract text from uploaded images using Groq Llama model
def extract_text_from_image(image):
    try:
        # Convert the PIL image to a base64-encoded string
        img_base64 = encode_image_to_base64(image)

        # Make the API call to Groq with base64 data embedded in a data URL
        completion = client.chat.completions.create(
            model="llama-3.2-11b-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Convert the content of this document image into GitHub Flavored Markdown format. Maintain the original structure and formatting as closely as possible. Include all text, tables, headings like and lists. Use appropriate Markdown syntax for headers, tables, and bullet points. Do not add any additional commentary or description outside of the document's content."},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}
                        },
                    ]
                }
            ],
            temperature=1,
            max_tokens=1024,
            top_p=1,
            stream=False,
            stop=None,
        )

        # Extract and return the text response from Groq
        text = completion.choices[0].message.content
        if not text.strip():
            raise ValueError("No text found in image.")
        return text

    except Exception as e:
        st.error(f"Error during OCR: {e}")
        return None

# Main UI and app logic
def main():
    st.title("Document OCR and Text Extraction")
    st.write("Upload an image with text, and we'll extract the text for you in formatted Markdown.")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Perform OCR and display extracted text in Markdown format
        st.subheader("Extracted Text in Markdown Format")
        text = extract_text_from_image(image)
        if text:
            # Display the extracted text using Markdown rendering
            st.markdown(text, unsafe_allow_html=True)  # Renders with Markdown formatting in Streamlit

# Run the Streamlit app
if __name__ == "__main__":
    main()
