from dotenv import load_dotenv
load_dotenv()

import os
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.llms import Ollama

# ---------------------------
# 1. EMBEDDINGS MODEL
# ---------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

# ---------------------------
# 2. PINECONE SETUP
# ---------------------------
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
pinecone_index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))

# ---------------------------
# 3. LLM (GEMINI)
# ---------------------------
# llm = ChatGoogleGenerativeAI(
#     model="models/gemini-2.0-flash",
#     google_api_key=os.getenv("GEMINI_API_KEY")
# )
llm = Ollama(model="llama3")

# ---------------------------
# 4. CHAT FUNCTION (RAG PIPELINE)
# ---------------------------
def chatting(question):

    # Step 1: Embed query
    query_vector = embeddings.embed_query(question)

    # Step 2: Search Pinecone
    results = pinecone_index.query(
        vector=query_vector,
        top_k=5,
        include_metadata=True
    )

    # Step 3: Build context safely (avoid empty crashes)
    context = "\n\n".join(
        match["metadata"].get("text", "")
        for match in results.get("matches", [])
        if match.get("metadata")
    )

    # Step 4: Prompt
    prompt = f"""
You are a helpful assistant.
Answer ONLY using the context below.

Context:
{context}

Question:
{question}
"""

    # Step 5: Get response
    response = llm.invoke(prompt)

    # IMPORTANT: Ollama returns STRING, not .content
    print("\n🤖 Answer:\n", response)


# ---------------------------
# 5. MAIN LOOP
# ---------------------------
def main():
    while True:
        user_question = input("\nHi I am Millie the housemaid ask me about my story --> ")
        chatting(user_question)


main()