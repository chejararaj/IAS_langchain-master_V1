from langchain.document_loaders import PyPDFLoader
import os
import tempfile
from fastapi import UploadFile
from langchain.text_splitter import RecursiveCharacterTextSplitter


async def process_pdf_file(file: UploadFile):
    """Extracts content and metadata from an uploaded file"""

    # Create temporary file in order we can use langchain
    with tempfile.NamedTemporaryFile(delete=False, suffix=file.filename) as tmp_file:
        await file.seek(0)
        content = await file.read()
        tmp_file.write(content)
        tmp_file.flush()

        loader = PyPDFLoader(tmp_file.name)
        pages = loader.load()

    os.remove(tmp_file.name)
    chunk_size = 500
    chunk_overlap = 0

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    texts = text_splitter.split_documents(pages)
    page_content = [text.page_content for text in texts]
    metadata = [text.metadata for text in texts]
    return page_content, metadata
