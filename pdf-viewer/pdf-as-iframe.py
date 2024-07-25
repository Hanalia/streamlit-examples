from typing import BinaryIO
import streamlit as st
import base64
from typing import BinaryIO
import streamlit as st
import base64

def display_pdf(file: BinaryIO) -> None:
    """Function to display a PDF file in the Streamlit app."""
    st.markdown("### PDF Preview")
    base64_pdf = base64.b64encode(file.read()).decode("utf-8")

    # Embedding PDF in HTML
    pdf_display = f"""<iframe src="data:application/pdf;base64,{base64_pdf}" width="400" height="100%" type="application/pdf"
                        style="height:100vh; width:100%"
                    >
                    </iframe>"""

    # Displaying File
    st.markdown(pdf_display, unsafe_allow_html=True)

def main():
    """Main function to handle the layout and file uploading."""
    st.set_page_config(layout="wide")
    # Create columns

    # Create two columns
    col1, col2 = st.columns([1, 1])

    with col1:
        # File uploader
        uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

    with col2:
      if uploaded_file is not None:
            # Display PDF
            display_pdf(uploaded_file)

if __name__ == "__main__":
    main()



