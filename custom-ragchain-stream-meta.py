## 최종업데이트 : 240724
## RAG Stream 버전
## context, question까지 chain 안에 assign 형태로 넣으면 stream 로직이 복잡해지기 때문에 다 분리해둠
from langchain_core.documents.base import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.runnables import Runnable
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import PyMuPDFLoader
from typing import List
from streamlit.runtime.uploaded_file_manager import UploadedFile
import os 
import fitz

load_dotenv()
def save_uploadedfile(uploadedfile: UploadedFile) -> str:
    # Create a temporary directory if it doesn't exist
    temp_dir = "temp_pdf_uploads"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    file_path = os.path.join(temp_dir, uploadedfile.name)
    with open(file_path, "wb") as f:
        f.write(uploadedfile.getbuffer())
    return file_path

## 약식버전

def pdf_to_documents(pdf_docs: List[UploadedFile]) -> List[Document]:
    documents = []
    for pdf in pdf_docs:
        file_path = save_uploadedfile(pdf)
        loader = PyMuPDFLoader(file_path)
        doc = loader.load()
        documents.extend(doc)
    return documents


## 커스텀 버전
# def pdf_to_documents(pdf_docs: List[UploadedFile]) -> List[Document]:
#     documents = []
    
#     for pdf in pdf_docs:
#         file_path = save_uploadedfile(pdf)
#         doc = fitz.open(file_path)
        
#         for page_num in range(len(doc)):
#             page = doc.load_page(page_num)
#             blocks = page.get_text("dict")["blocks"]
            
#             page_text = ""
#             for block in blocks:
#                 if "lines" not in block:
#                     continue
#                 for line in block["lines"]:
#                     for span in line["spans"]:
#                         page_text += span["text"] + " "
            
#             document = Document(page_content=page_text.strip(), metadata={"source": pdf.name, "page_number": page_num})
#             documents.append(document)
    
#     return documents



def chunk_documents(documents: List[Document]) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_chunks = []
    chunks = text_splitter.split_documents(documents)
    final_chunks.extend(chunks)
    return final_chunks

def get_vector_store(documents: List[Document]) -> None:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = FAISS.from_documents(documents, embedding=embeddings)
    vector_store.save_local("faiss_index")

## 추상화된 chain

## chain의 의미를 살려서 조금 더 고도화된 버전
## 범위를 한정해서 
def get_rag_chain() -> Runnable:
    template = """Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Use three sentences maximum and keep the answer as concise as possible.
    Always say "thanks for asking!" at the end of the answer.

    {context}

    Question: {question}

    Helpful Answer:"""

    custom_rag_prompt = PromptTemplate.from_template(template)

    model = ChatOpenAI(model="gpt-4o-mini")

    rag_chain_from_docs = (
        custom_rag_prompt
        | model
        | StrOutputParser()
    )
    
    return rag_chain_from_docs
    


def process_input(user_question):

    ## retriever 정의 후

    embeddings=OpenAIEmbeddings(model="text-embedding-3-small")
    new_db = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
    retriever = new_db.as_retriever(search_kwargs={"k": 2})

    retrieve_docs = retriever.invoke(user_question)


    chain = get_rag_chain()
    response = chain.stream({"question":user_question,"context" :retrieve_docs})


    st.write_stream(response)
    for doc in retrieve_docs:
        with st.expander("Evidence context"):
            st.write(doc.page_content)
            print(doc.metadata)
            if 'source' in doc.metadata and 'page' in doc.metadata:
                st.markdown(f"<p style='color:blue; font-weight:semi-bold;'>🔎 {doc.metadata['source']} pg.{doc.metadata['page']} </p>", unsafe_allow_html=True)


def main():
    st.set_page_config("Chat with PDF",layout="wide")
    st.header("Chat with PDF using ChatGPT💁")

    ## sidebar 정의
    with st.sidebar:
        st.title("Upload your PDF Files and Click on the Submit & Process Button")
        pdf_docs = st.file_uploader("PDF Uploader", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing Pdf"):
                pdf_documents = pdf_to_documents(pdf_docs)
            with st.spinner("Creating chunks for Pdf"):
                smaller_documents = chunk_documents(pdf_documents)
            with st.spinner("Vector DB creating"):
                get_vector_store(smaller_documents)
                st.success("Done")

    user_question = st.text_input("업로도된 PDF에 대해서 질문해 주세요",
                                 placeholder="이 문서를 요약해 주세요",)

    if user_question:
        process_input(user_question)




if __name__ == "__main__":
    main()