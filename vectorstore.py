from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import hashlib
import os

PERSIST_DIR = "db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 200
CHUNK_OVERLAP = 50

# Load text file
loader = TextLoader("demo.txt")
documents = loader.load()
file_text = documents[0].page_content
file_hash = hashlib.md5(file_text.encode()).hexdigest()
hash_marker = os.path.join(PERSIST_DIR, f"hash_{file_hash}.done")

if os.path.exists(hash_marker):
    print(" File already embedded. Skipping.")
    exit()

# Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
docs = splitter.split_documents(documents)

# Create embeddings and store
embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
vectorstore = Chroma.from_documents(docs, embedding, persist_directory=PERSIST_DIR)

with open(hash_marker, "w") as f:
    f.write("done")

print(" Vector store created and saved to ./db")