from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# -----------------------------
# 1. Knowledge base (30 facts)
# -----------------------------
docs = [
    "India is the seventh largest country by area.",
    "The capital of India is New Delhi.",
    "India has 28 states and 8 union territories.",
    "The Taj Mahal is located in Agra.",
    "Hindi is the most spoken language in India.",
    "India gained independence in 1947.",
    "The Indian flag has saffron, white and green colors.",
    "The Ganges is the holiest river in India.",
    "Mumbai is the financial capital of India.",
    "India is the largest democracy in the world.",
    "ISRO is India's space agency.",
    "India launched Chandrayaan missions to the moon.",
    "Cricket is the most popular sport in India.",
    "The Indian economy is one of the fastest growing.",
    "The Himalayas are located in northern India.",
    "India has a diverse culture and traditions.",
    "Bollywood is India's film industry.",
    "The currency of India is the Indian Rupee.",
    "India uses a parliamentary system of government.",
    "The Ashoka Chakra is in the Indian flag.",
    "Diwali is a major festival in India.",
    "Holi is known as the festival of colors.",
    "India has over 1.4 billion people.",
    "The Bengal tiger is India's national animal.",
    "The peacock is India's national bird.",
    "Delhi is one of the oldest cities in India.",
    "India has a tropical monsoon climate.",
    "The Brahmaputra is a major river in India.",
    "Indian cuisine is rich in spices and flavors.",
    "Yoga originated in ancient India."
]

# -----------------------------
# 2. Convert text → embeddings
# -----------------------------
vectorizer = TfidfVectorizer()
doc_vectors = vectorizer.fit_transform(docs)

# -----------------------------
# 3. Semantic search function
# -----------------------------
def search(query, top_k=3):
    # query embedding
    query_vec = vectorizer.transform([query])
    
    # cosine similarity with all docs
    scores = cosine_similarity(query_vec, doc_vectors)[0]
    
    # top-k results
    top_indices = np.argsort(scores)[::-1][:top_k]
    
    print("\nQuery:", query)
    print("\nTop Results:\n")
    
    for i in top_indices:
        print(f"Score: {scores[i]:.4f} | {docs[i]}")

# -----------------------------
# 4. Try it
# -----------------------------
while True:
    q = input("\nEnter query (or 'exit'): ")
    if q.lower() == "exit":
        break
    search(q)