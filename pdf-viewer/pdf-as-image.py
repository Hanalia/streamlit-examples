import streamlit as st
from pdf2image import convert_from_path
import tempfile
from typing import BinaryIO, List
from PIL import Image

def display_pdf_page(image: Image.Image, page_number: int, total_pages: int) -> None:
    """Function to display a specific page of the PDF as an image in the Streamlit app."""
    st.image(image, use_column_width=True, caption=f"Page {page_number}")
    st.write(f"Page {page_number} of {total_pages}")

def save_uploaded_file(uploaded_file: BinaryIO) -> str:
    """Save the uploaded PDF file to a temporary file and return the file path."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name
    return temp_file_path

def convert_pdf_to_images(pdf_path: str) -> List[Image.Image]:
    """Convert a PDF file to a list of images, one per page."""
    return convert_from_path(pdf_path)

def main():
    """Main function to handle the layout and file uploading."""
    st.set_page_config(layout="wide")

    # Title of the Streamlit app
    st.title("PDF Viewer")

    # Create two columns
    col1, col2 = st.columns([1, 1])

    with col2:
        # File uploader in the right column
        uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

        if uploaded_file is not None:
            # Save the uploaded file to a temporary file
            temp_file_path = save_uploaded_file(uploaded_file)

            # Convert PDF to list of images
            images = convert_pdf_to_images(temp_file_path)

            # Get the total number of pages
            total_pages = len(images)

            # Create a slider for page navigation
            page_number = st.slider("Select Page", min_value=1, max_value=total_pages, value=1)

    with col1:
        if uploaded_file is not None:
            # Display the selected page in the left column
            display_pdf_page(images[page_number - 1], page_number, total_pages)

if __name__ == "__main__":
    main()
