## 최종업데이트 : 240721
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.runnables import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

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

## chain의 의미를 살려서 조금 더 고도화된 버전
## https://python.langchain.com/v0.2/docs/how_to/qa_sources/
def get_rag_chain():
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
        {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
        | custom_rag_prompt
        | model
        | StrOutputParser()
    )
    
    ## retriever 정의

    embeddings=OpenAIEmbeddings(model="text-embedding-3-small")
    new_db = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
    retriever = new_db.as_retriever()

    retrieve_docs = (lambda x: x["question"]) | retriever

    result_chain = RunnablePassthrough.assign(context=retrieve_docs).assign(
        answer=rag_chain_from_docs
    )

    return result_chain


def user_input(user_question):
    chain = get_rag_chain()
    response = chain.invoke({"question":user_question})


    print(response)
    st.write("Reply: ", response["answer"])
            # 증거자료 보여주기
    for doc in response["context"]:
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
