## 최종업데이트 : 240722
from langchain_openai import ChatOpenAI
import streamlit as st
from PyPDF2 import PdfReader
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from dotenv import load_dotenv

load_dotenv()


### 만약 Stream 형태로 만들고 싶으면!
## chain.invoke 대신 chain.stream
## st.write 대신 st.write_stream

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_summarizer_chain():
    template = """Summarize the following text:
    
    {context}

    Summary:"""

    custom_summary_prompt = PromptTemplate.from_template(template)

    model = ChatOpenAI(model="gpt-4o-mini")

    summarizer_chain = (
        custom_summary_prompt
        | model
        | StrOutputParser()
    )

    return summarizer_chain

def summarize_text(text):
    chain = get_summarizer_chain()
    response = chain.stream({"context": text})
    return response

def main():
    st.set_page_config("Summarize PDF")
    st.header("PDF Summarizer using ChatGPT💁")

    pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True, key='file_uploader')

    if st.button("Summarize PDF"):
        if pdf_docs:
            with st.spinner("Processing PDF..."):
                raw_text = get_pdf_text(pdf_docs)
            with st.spinner("Summarizing..."):
                summary = summarize_text(raw_text)
                st.write_stream(summary)
        else:
            st.error("Please upload at least one PDF file.")
if __name__ == "__main__":
    main()