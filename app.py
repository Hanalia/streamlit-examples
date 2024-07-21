## 240721
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings=OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

## 추상화된 chain
def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatOpenAI(model="gpt-4o-mini")

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


def user_input(user_question):
    embeddings=OpenAIEmbeddings(model="text-embedding-3-small")

    new_db = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question,k=3)


    ## 단순한 방법
    chain = get_conversational_chain()
    response = chain(
        {"input_documents": docs, "question": user_question}
        , return_only_outputs=False)




    print(response)
    st.write("Reply: ", response["output_text"])
            # 증거자료 보여주기
    for doc in response["input_documents"]:
        with st.expander("Evidence context"):
            st.write(doc.page_content)            


def main():
    st.set_page_config("Chat with PDF")
    st.header("Chat with PDF using ChatGPT💁")

    user_question = st.text_input("업로도된 PDF에 대해서 질문해 주세요",
                                 placeholder="이 문서를 요약해 주세요",)

    if user_question:
        user_input(user_question)


    with st.sidebar:
        st.title("Upload your PDF Files and Click on the Submit & Process Button")
        pdf_docs = st.file_uploader("PDF Uploader", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing Pdf"):
                raw_text = get_pdf_text(pdf_docs)
            with st.spinner("Creating chunks for Pdf"):
                text_chunks = get_text_chunks(raw_text)
            with st.spinner("Vector DB creating"):
                get_vector_store(text_chunks)
                st.success("Done")


if __name__ == "__main__":
    main()
