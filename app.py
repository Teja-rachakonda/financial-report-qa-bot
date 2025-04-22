import streamlit as st
from langchain_utils import query_with_rag

def main():
    st.write("Financial Report QQ&A Bot With RAG + GROQ")
    st.write("Upload an annual report (PDF) and ask any question related to the financial statement.")
    
    uploaded_file = st.file_uploader("Choose a financial report (PDF)", type = "pdf")
    
    if uploaded_file is not None:
        st.write("Processing your uploaded report...")
        
        question = st.text_input("Ask a question about the financial report:-")
        
        if question:
            answer = query_with_rag(question, uploaded_file)
            st.write(f"Answer: {answer}")
            
if __name__ == "__main__":
    main()
    