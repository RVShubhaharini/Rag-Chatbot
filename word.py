import streamlit as st
import fitz  # PyMuPDF
from docx import Document
import io
import zipfile
import re

def clean_text(text):
    """Remove NULL bytes and invalid XML characters."""
    # Remove NULL bytes and control chars except common whitespace
    return re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", "", text)

def convert_pdf_to_docx(pdf_bytes):
    """Convert PDF bytes to DOCX BytesIO object."""
    document = Document()
    with fitz.open(stream=pdf_bytes, filetype="pdf") as pdf:
        for page in pdf:
            text = page.get_text()
            if text.strip():
                safe_text = clean_text(text)
                document.add_paragraph(safe_text)
                document.add_page_break()

    docx_io = io.BytesIO()
    document.save(docx_io)
    docx_io.seek(0)
    return docx_io

st.title("Multiple PDF to DOCX Converter (ZIP Output)")

uploaded_pdfs = st.file_uploader(
    "Upload one or more PDF files",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_pdfs:
    st.success(f"{len(uploaded_pdfs)} PDF file(s) uploaded successfully.")

    output_zip_io = io.BytesIO()
    with zipfile.ZipFile(output_zip_io, 'w', zipfile.ZIP_DEFLATED) as output_zip:
        for pdf_file in uploaded_pdfs:
            pdf_bytes = pdf_file.read()
            docx_io = convert_pdf_to_docx(pdf_bytes)
            docx_filename = pdf_file.name.replace(".pdf", ".docx")
            output_zip.writestr(docx_filename, docx_io.getvalue())

    output_zip_io.seek(0)

    st.download_button(
        label="Download All Converted DOCX Files (ZIP)",
        data=output_zip_io,
        file_name="converted_docx_files.zip",
        mime="application/zip"
    )
