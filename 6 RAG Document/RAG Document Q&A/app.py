import streamlit as st
import os
import time
from dotenv import load_dotenv

# LangChain + Groq (new modular structure)
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ✅ New chain imports (moved in v0.3+)
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain

from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

# 1️⃣ Load environment variables
# ---------------------------------------------------------------------



# ✅ Groq API key

# ---------------------------------------------------------------------
# 2️⃣ Initialize LLM
# ---------------------------------------------------------------------
llm = ChatGroq(groq_api_key="" model_name="llama-3.1-8b-instant")

# ---------------------------------------------------------------------
# 3️⃣ Define Prompt Template
# ---------------------------------------------------------------------
prompt = ChatPromptTemplate.from_template(
    """
    You are a research assistant. Use the provided context to answer the user's question.
    <context>
    {context}
    </context>

    Question: {input}
    """
)

# ---------------------------------------------------------------------
# 4️⃣ Create vector embedding (RAG setup)
# ---------------------------------------------------------------------
def create_vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = OllamaEmbeddings()
        st.session_state.loader = PyPDFDirectoryLoader("./research_papers")
        st.session_state.docs = st.session_state.loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_docs = text_splitter.split_documents(st.session_state.docs[:50])

        st.session_state.vectors = FAISS.from_documents(
            st.session_state.final_docs, st.session_state.embeddings
        )

# ---------------------------------------------------------------------
# 5️⃣ Streamlit UI
# ---------------------------------------------------------------------
st.title("📚 RAG with Research Papers")
user_prompt = st.text_input("Enter your query:")

if st.button("🔍 Create Vector Embedding"):
    create_vector_embedding()
    st.success("✅ Vector Database is ready!")

# ---------------------------------------------------------------------
# 6️⃣ Retrieval + Response
# ---------------------------------------------------------------------
if user_prompt:
    if "vectors" not in st.session_state:
        st.warning("Please click 'Create Vector Embedding' first.")
    else:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        start = time.process_time()
        response = retrieval_chain.invoke({"input": user_prompt})
        elapsed = time.process_time() - start

        st.write(f"⏱ Response time: {elapsed:.2f}s")
        st.subheader("💡 Answer:")
        st.write(response["answer"])

        with st.expander("📄 Document Similarity Search"):
            for i, doc in enumerate(response["context"]):
                st.markdown(f"**Document {i+1}:**")
                st.write(doc.page_content)
                st.markdown("---")
