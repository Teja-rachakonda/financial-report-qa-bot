import fitz
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma, FAISS
from langchain.chains import RetrievalQA 
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
import os
import tempfile

groq_model = ChatGroq(api_key="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", model_name="llama3-70b-8192")


embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

def load_pdf(pdf_file):
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(pdf_file.read())
        temp_file_path = tmp_file.name
    
    loader = PyMuPDFLoader(temp_file_path)
    return loader.load()


def chunk_pdf(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200
    )
    return text_splitter.split_documents(documents)

def create_vectore_store(documents):
    vectore_store = Chroma.from_documents(documents, embeddings)
    return vectore_store

def retrieve_relevant_documents(query, vectore_store):
    retriever = vectore_store.as_retriever(search_type="similarity", search_kwargs = {"k": 5})
    return retriever.get_relevant_documents(query)

from langchain_core.messages import HumanMessage

def query_with_rag(query, pdf_file):
    documents = load_pdf(pdf_file)
    document_chunks = chunk_pdf(documents)
    
    vectore_store = create_vectore_store(document_chunks)
    
    relevant_docs = retrieve_relevant_documents(query, vectore_store)
    
    context = "\n".join([doc.page_content for doc in relevant_docs])
    
    prompt = f"Answer the question based on the context below:\n\nContext: {context}\n\nQuestion: {query}\nAnswer:"
    
    response = groq_model.invoke([HumanMessage(content=prompt)])
    return response.content
