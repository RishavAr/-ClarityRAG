from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import requests

# 1. Synthetic Confluence-like documents
documents = [
    "To configure OAuth in Confluence, go to Application Links and create a new link.",
    "Only administrators can create new spaces in Confluence.",
    "SAML Single Sign-On can be enabled from the Security settings.",
    "To integrate Jira, use the Jira plugin available under Marketplace.",
    "Permissions are managed on a per-space basis via the Space Settings menu."
]

# 2. Generate embeddings locally
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(documents, convert_to_numpy=True)

# 3. Store embeddings in FAISS index
embedding_dim = embeddings.shape[1]
faiss_index = faiss.IndexFlatL2(embedding_dim)
faiss_index.add(embeddings)
id_to_doc = {i: doc for i, doc in enumerate(documents)}

# 4. Embed user query
user_query = "How do I enable SSO for Confluence?"
query_vector = model.encode([user_query], convert_to_numpy=True)

# 5. Search vector DB
k = 2
_, indices = faiss_index.search(query_vector, k)
retrieved_chunks = [id_to_doc[i] for i in indices[0]]

# 6. Construct prompt
prompt = f"""
You are a helpful Confluence assistant. Use the context below to answer the question.

Context:
- {retrieved_chunks[0]}
- {retrieved_chunks[1]}

Question:
{user_query}
"""

# 7. Call Ollama (LLaMA3)
response = requests.post(
    "http://localhost:11434/api/generate",
    json={"model": "llama3", "prompt": prompt, "stream": False}
)

answer = response.json().get("response", "[No response from LLaMA]")

# 8. Final output
print("\nFinal Answer from LLaMA:")
print(answer)

