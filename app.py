import streamlit as st
from pypdf import PdfReader
import ollama

from utils.rag import create_vector_store


st.set_page_config(
    page_title="Research Paper Comparison Tool",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Research Paper Comparison Tool")
st.write("Upload two research papers and compare them using AI + RAG.")


# -----------------------------
# PDF TEXT EXTRACTION
# -----------------------------

def extract_text(pdf_file):
    reader = PdfReader(pdf_file)

    text = ""

    for page in reader.pages:
        text += page.extract_text() or ""

    return text


# -----------------------------
# PDF UPLOAD
# -----------------------------

paper1 = st.file_uploader(
    "Upload Research Paper 1",
    type=["pdf"]
)

paper2 = st.file_uploader(
    "Upload Research Paper 2",
    type=["pdf"]
)


# -----------------------------
# COMPARE PAPERS
# -----------------------------

if st.button("🔍 Compare Papers"):

    if paper1 is None or paper2 is None:

        st.warning("Please upload both research papers.")

    else:

        with st.spinner("Reading research papers..."):

            text1 = extract_text(paper1)
            text2 = extract_text(paper2)

        st.success("PDFs extracted successfully!")

        # Create vector databases
        with st.spinner("Creating embeddings and vector database..."):

            vector_store1 = create_vector_store(
                text1,
                "research_paper_1"
            )

            vector_store2 = create_vector_store(
                text2,
                "research_paper_2"
            )

        st.success("RAG database created successfully! ✅")

        # Retrieve important sections
        retriever1 = vector_store1.as_retriever(
            search_kwargs={"k": 5}
        )

        retriever2 = vector_store2.as_retriever(
            search_kwargs={"k": 5}
        )

        documents1 = retriever1.invoke(
            "objectives methodology findings limitations"
        )

        documents2 = retriever2.invoke(
            "objectives methodology findings limitations"
        )

        context1 = "\n\n".join(
            doc.page_content for doc in documents1
        )

        context2 = "\n\n".join(
            doc.page_content for doc in documents2
        )

        # AI prompt
        prompt = f"""
You are a research paper comparison assistant.

Compare these two research papers using the retrieved information.

RESEARCH PAPER 1:
{context1}

RESEARCH PAPER 2:
{context2}

Give the comparison using these headings:

1. Objectives
2. Methodology
3. Key Findings
4. Advantages
5. Limitations
6. Similarities
7. Differences
8. Overall Comparison

Give a clear and simple answer.
"""

        # Ask Ollama
        with st.spinner("AI is comparing the papers..."):

            response = ollama.chat(
                model="llama3.2",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

        st.subheader("📊 Comparison Result")

        st.write(
            response["message"]["content"]
        )