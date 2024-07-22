import streamlit as st
from PyPDF2 import PdfReader
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from dotenv import load_dotenv
import streamlit.components.v1 as v1
from langchain_core.runnables import Runnable
from typing import List

load_dotenv()
#Testbox

def get_pdf_text(pdf_docs: List[st.uploaded_file_manager.UploadedFile]) -> str:
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_summarizer_chain(prompt)-> Runnable:
    template = f"""Summarize the following text:

    {{context}}

    Summary based on the prompt: {prompt}"""

    custom_summary_prompt = PromptTemplate.from_template(template)
    model = ChatOpenAI(model="gpt-4o-mini")

    summarizer_chain = (
        custom_summary_prompt
        | model
        | StrOutputParser()
    )

    return summarizer_chain

def summarize_text(text:str, prompt:str)-> str:
    chain = get_summarizer_chain(prompt)
    response = chain.stream({"context": text})
    return response

def main():
    st.header("PDF Summarizer using ChatGPT💁")

    pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True, key='file_uploader')
    
    # Sample prompts for the user to choose from or edit
    prompt_options = [
        "Summarize the key points and conclusions.",
        "Provide a detailed summary with main arguments and evidence.",
        "Extract and summarize the most critical information."
    ]

    selected_prompt = st.selectbox("Choose a summary style or edit below:", prompt_options)

    # Allow editing the prompt
    custom_prompt = st.text_area("Edit your summary prompt:", value=selected_prompt, height=500)

    if st.button("Summarize PDF"):
        if pdf_docs:
            with st.spinner("Processing PDF..."):
                raw_text = get_pdf_text(pdf_docs)
            with st.spinner("Summarizing..."):
                summary = summarize_text(raw_text, custom_prompt)
                st.write_stream(summary)
        else:
            st.error("Please upload at least one PDF file.")

if __name__ == "__main__":
    main()
