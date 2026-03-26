from dotenv import load_dotenv
import json
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings 
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
import os

load_dotenv()

######################################### READING PDF 

PDF_PATH = [
    "./pdfs/The_Housemaid.pdf",
    "./pdfs/HouseMaid_Wedding.pdf",
    "./pdfs/The_Housemaid_Is_Watching.pdf",
]

all_docs = []

for path in PDF_PATH:
    loader = PyPDFLoader(path)
    docs = loader.load()
    all_docs.extend(docs)

print("Total documents loaded:", len(all_docs))

#################################### CHUNKING

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

chunked_docs = text_splitter.split_documents(all_docs)
print('Chunks of data has been made, length is: ', len(chunked_docs))

############ CONVERT TO VECTORS USING VECTOR EMBEDDING MODEL 

####### INTITALISE EMBEDDING MODEL (CHANGED ONLY THIS)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

print('Embedding model has been initialised')

############# INITIALISE VECTOR DB PINECONE CLIENT
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
pinecone_index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))
print('Pinecone client has been initialised')

############### EMBED CHUNKS AND UPLOAD TO PINECONE

BATCH_SIZE = 20

vector_store = None

for i in range(0, len(chunked_docs), BATCH_SIZE):
    batch = chunked_docs[i:i + BATCH_SIZE]

    if vector_store is None:
        vector_store = PineconeVectorStore.from_documents(
            documents=batch,
            embedding=embeddings,
            index_name=os.getenv("PINECONE_INDEX_NAME"),
        )
    else:
        vector_store.add_documents(batch)

    print(f"Processed batch {i}-{i + len(batch)}")

print('Data is stored successfully in Pinecone vector DB')