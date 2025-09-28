from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from extract_texts import extract_texts_from_pdfs

import os
import openai


openai.api_key = os.getenv("OPENAI_API_KEY")  

#Load the texts
texts = extract_texts_from_pdfs("data")

# Split texts into chunks
all_chunks = []
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

for item in texts:
    chunks = splitter.split_text(item["text"])
    for chunk in chunks:
        all_chunks.append(
            Document(
                page_content=chunk,
                metadata={
                    "filename": item["filename"],
                    "category": item["category"]
                }
            )
        )

# Create embedding and index
embeddings = OpenAIEmbeddings()
vector_db = FAISS.from_documents(all_chunks, embeddings)

# Save the vector database locally
vector_db.save_local("legal_index_mz")

print("Vector index created and saved as 'legal_index_mz'")
