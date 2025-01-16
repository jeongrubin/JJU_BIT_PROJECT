import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_teddynote.messages import stream_response
import streamlit as st
import time

def process_pdf(pdf_filepath):
    """Processes the uploaded PDF file and splits its content into semantic chunks."""
    print("Processing PDF file...")
    try:
        loader = PyMuPDFLoader(pdf_filepath)
        pages = loader.load()
        print(f"Loaded {len(pages)} pages from the PDF.")

        text_splitter = SemanticChunker(
            OpenAIEmbeddings(api_key=os.environ.get("OPENAI_API_KEY")),
            breakpoint_threshold_type="standard_deviation",
            breakpoint_threshold_amount=1.25,
        )
        print("Initialized semantic chunker.")

        texts = []
        for page_number, page in enumerate(pages, start=1):
            chunks = text_splitter.split_text(page.page_content)
            texts.extend(chunks)
            print(f"Processed page {page_number} with {len(chunks)} chunks.")

        print("PDF processing complete.")
        return texts

    except Exception as e:
        print(f"Error during PDF processing: {e}")
        raise

def create_vector_database(texts):
    """Creates a vector database from the provided semantic chunks."""
    print("Creating vector database...")
    try:
        embeddings_model = OpenAIEmbeddings()
        print("Initialized OpenAI embeddings model.")

        db = Chroma.from_texts(
            texts,
            embeddings_model,
            collection_name='esg',
            persist_directory='./SWEngineer/db/chromadb',
            collection_metadata={'hnsw:space': 'cosine'},
        )
        print("Vector database created successfully.")
        return db

    except Exception as e:
        print(f"Error during vector database creation: {e}")
        raise

def query_database(db, query):
    """Queries the vector database using Max Marginal Relevance search."""
    print("Querying the vector database...")
    try:
        mmr_docs = db.max_marginal_relevance_search(query, k=5, fetch_k=20)
        print(f"Query returned {len(mmr_docs)} documents.")
        return mmr_docs

    except Exception as e:
        print(f"Error during querying: {e}")
        raise

def generate_response(query, mmr_docs):
    """Generates a response for the given query using the retrieved documents."""
    print("Generating response...")
    try:
        question = {
            "instruction": query,
            "mmr_docs": mmr_docs
        }

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "Analyze the following content and answer the question. Answer in Korean."),
                ("human", "{instruction}\n{mmr_docs}"),
            ]
        )
        print("Initialized ChatPromptTemplate.")

        llm = ChatOpenAI(
            temperature=0,
            model_name="gpt-4o",
        )
        print("Initialized ChatOpenAI model.")

        chain = prompt | llm
        answer = chain.stream(question)
        print("Response generated successfully.")

        st.markdown(f"<p style='font-size:20px;'>{query}</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size:18px;'>Response: {stream_response(answer, return_output=True)}</p>", unsafe_allow_html=True)

    except Exception as e:
        print(f"Error during response generation: {e}")
        raise

def main_streamlit():
    load_dotenv()

    st.set_page_config(page_title="Semantic Analysis", page_icon="🔬", layout="wide")

    st.sidebar.title("Options")
    st.sidebar.write("Upload a file and enter your query to analyze it.")

    uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])

    st.title("🔬 Semantic Analysis with LangChain")
    st.markdown("""
    This application allows you to upload a PDF, analyze its content, and retrieve information using natural language queries.
    
    **Features:**
    - Semantic chunking for better document understanding.
    - Vector database for fast and accurate search.
    - Responses in Korean with typing animation.
    """)

    if 'global_db' not in st.session_state or "texts" not in st.session_state:
        st.session_state.global_db = None
        st.session_state.texts = None

    if uploaded_file:
        temp_file_path = f"temp_{uploaded_file.name}"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.read())
        print(f"Uploaded file saved as {temp_file_path}.")

        if st.session_state.texts is None:
            st.session_state.texts = process_pdf(temp_file_path)

        if st.session_state.global_db is None:
            try:
                st.session_state.global_db = create_vector_database(st.session_state.texts)
            except Exception as e:
                print(f"An error occurred during file processing: {e}")

    query = st.sidebar.text_input("Enter your query:")
    query_button = st.sidebar.button("Send Query")

    if query_button:
        if st.session_state.global_db is None:
            print("Please upload a file to create the database first.")
        else:
            try:
                mmr_docs = query_database(st.session_state.global_db, query)
                generate_response(query, mmr_docs)
            except Exception as e:
                print(f"An error occurred during querying: {e}")

if __name__ == "__main__":
    main_streamlit()