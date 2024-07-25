import streamlit as st
from langchain_core.documents.base import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.runnables import Runnable
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import PyMuPDFLoader
from typing import List, BinaryIO
import os
import tempfile
from pdf2image import convert_from_path
from PIL import Image

load_dotenv()

def save_uploadedfile(uploadedfile: BinaryIO) -> str:
    temp_dir = "temp_pdf_uploads"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    file_path = os.path.join(temp_dir, uploadedfile.name)
    with open(file_path, "wb") as f:
        f.write(uploadedfile.getbuffer())
    return file_path

def pdf_to_documents(pdf_docs: List[BinaryIO]) -> List[Document]:
    documents = []
    for pdf in pdf_docs:
        file_path = save_uploadedfile(pdf)
        loader = PyMuPDFLoader(file_path)
        doc = loader.load()
        for d in doc:
            d.metadata['file_path'] = file_path
        documents.extend(doc)
    return documents

def chunk_documents(documents: List[Document]) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(documents)

def get_vector_store(documents: List[Document]) -> None:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = FAISS.from_documents(documents, embedding=embeddings)
    vector_store.save_local("faiss_index")

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

    return custom_rag_prompt | model | StrOutputParser()

def display_pdf_page(image: Image.Image, page_number: int, total_pages: int) -> None:
    st.image(image, use_column_width=True, caption=f"Page {page_number}")
    # st.write(f"Page {page_number} of {total_pages}")
    # st.markdown(f"<div style='text-align: center;'>Page {page_number} of {total_pages}</div>", unsafe_allow_html=True)

def save_uploaded_file(uploaded_file: BinaryIO) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        return temp_file.name

def convert_pdf_to_images(pdf_path: str) -> List[Image.Image]:
    return convert_from_path(pdf_path)

@st.cache_data
def process_question(user_question):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    retriever = new_db.as_retriever(search_kwargs={"k": 2})

    retrieve_docs = retriever.invoke(user_question)
    chain = get_rag_chain()
    response = chain.invoke({"question": user_question, "context": retrieve_docs})

    return response, retrieve_docs



def main():
    st.set_page_config("Chat with PDF", layout="wide")
    st.header("Chat with PDF using ChatGPT💁")

    # Initialize session state
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []
    if "page_number" not in st.session_state:
        st.session_state.page_number = 1
    if "images" not in st.session_state:
        st.session_state.images = []
    if "response" not in st.session_state:
        st.session_state.response = None
    if "context" not in st.session_state:
        st.session_state.context = []

    # Create two columns
    left_column, right_column = st.columns([1, 1])

    with left_column:
        with st.sidebar:
            st.title("Upload your PDF Files and Click on the Submit & Process Button")
            pdf_docs = st.file_uploader("PDF Uploader", accept_multiple_files=True, type="pdf")
            if st.button("Submit & Process"):
                with st.spinner("Processing Pdf"):
                    pdf_documents = pdf_to_documents(pdf_docs)
                with st.spinner("Creating chunks for Pdf"):
                    smaller_documents = chunk_documents(pdf_documents)
                with st.spinner("Vector DB creating"):
                    get_vector_store(smaller_documents)
                    st.success("Done")
                st.session_state.uploaded_files = pdf_docs

        user_question = st.text_input("Ask a question about the uploaded PDF",
                                      placeholder="Summarize this document, please")

        # if user_question:
        #     process_input(user_question)

        if user_question:
            response, context = process_question(user_question)
            st.session_state.response = response
            st.session_state.context = context

        if st.session_state.response:
            st.write(st.session_state.response)
            for idx, doc in enumerate(st.session_state.context):
                with st.expander("Evidence context"):
                    st.write(doc.page_content.replace("$", r"\$"))
                    print(doc.metadata)
                    file_path = doc.metadata.get('source', '')
                    page_number = doc.metadata.get('page', 0) + 1
                    if file_path and page_number:
                        button_key = f"link_{file_path}_{page_number}_{idx}"  # Add idx to make the key unique
                        if st.button(f"🔎 {os.path.basename(file_path)} pg.{page_number}", key=button_key):
                            st.session_state.page = str(page_number)
                            st.session_state.file = file_path
                            st.rerun()

    with right_column:
        # Safely get query parameters
        file = st.session_state.get('file')
        page_number = st.session_state.get('page_number')

        if file and page_number:
            try:
                page_number = int(page_number)
                if st.session_state.images == [] or st.session_state.images[0][0] != file:
                    images = convert_pdf_to_images(file)
                    st.session_state.images = (file, images)
                else:
                    images = st.session_state.images[1]
                total_pages = len(images)
                display_pdf_page(images[page_number - 1], page_number, total_pages)

                # Add pagination buttons in a single row
                prev_col, _, next_col = st.columns([1, 5, 1])
                with prev_col:
                    if page_number > 1:
                        if st.button("Prev Page"):
                            st.session_state.page_number = page_number - 1
                            st.rerun()
                with next_col:
                    if page_number < total_pages:
                        if st.button("Next Page"):
                            st.session_state.page_number = page_number + 1
                            st.rerun()
            except (ValueError, IndexError, FileNotFoundError) as e:
                st.error(f"Error displaying PDF page: {str(e)}")

if __name__ == "__main__":
    main()