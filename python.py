import os 
import streamlit as st
import pickle
import time
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import FAISS

# Load environment variables
load_dotenv()

st.title("Research Tool 📈")
st.sidebar.title("Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_gemini.pkl"

main_placeholder = st.empty()

# Gemini API key
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("⚠️ GOOGLE_API_KEY not found. Please set it in your .env file.")
    st.stop()

# LLM (Gemini Pro)
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=api_key,
    temperature=0.9
)

# Embeddings (Gemini)
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=api_key
)

if process_url_clicked:
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started...✅")
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    main_placeholder.text("Text Splitter...Started...✅")
    docs = text_splitter.split_documents(data)

    vectorstore_gemini = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("Embedding Vector Started Building...✅")
    time.sleep(2)

    vectorstore_gemini.save_local("faiss_index_gemini")

query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        vectorstore = FAISS.load_local("faiss_index_gemini", embeddings, allow_dangerous_deserialization=True)

        chain = RetrievalQAWithSourcesChain.from_llm(
            llm=llm, 
            retriever=vectorstore.as_retriever()
        )
        result = chain({"question": query}, return_only_outputs=True)

        st.header("Answer")
        st.write(result["answer"])

        sources = result.get("sources", "")
        if sources:
            st.subheader("Sources:")
            sources_list = sources.split("\n")
            for source in sources_list:
                st.write(source)
