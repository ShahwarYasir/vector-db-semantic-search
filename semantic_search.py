"""
Small semantic search system using Chroma (local vector database).

Setup:
    pip install chromadb
Run:
    python semantic_search.py

Chroma's default embedding model (all-MiniLM-L6-v2) is downloaded
automatically on first run, so no API key is needed.
"""

import chromadb

# 1. Create a local, persistent vector database
client = chromadb.PersistentClient(path="./chroma_db")

# Start fresh each run so results are reproducible
try:
    client.delete_collection("university_faqs")
except Exception:
    pass

collection = client.create_collection(
    name="university_faqs",
    metadata={"hnsw:space": "cosine"},  # use cosine distance
)

# 2. Sample information: 10 short fictional university FAQs
faqs = [
    ("When does the fall semester registration open?",
     "Registration for the fall semester opens on July 1 and closes on August 15.",
     "Admissions"),
    ("How do I apply for a scholarship?",
     "Submit the scholarship form and your latest transcript to the financial aid office before March 31.",
     "Financial Aid"),
    ("What are the library opening hours?",
     "The main library is open from 8 AM to 10 PM on weekdays and 10 AM to 6 PM on weekends.",
     "Campus Services"),
    ("How can I reset my student portal password?",
     "Click 'Forgot password' on the portal login page and follow the link sent to your university email.",
     "IT Support"),
    ("Does the university offer on-campus housing?",
     "Yes, dormitories are available for first-year students and can be requested during enrollment.",
     "Housing"),
    ("How do I drop a course after the semester starts?",
     "Submit a course withdrawal form to the registrar within the first four weeks of the semester.",
     "Registrar"),
    ("Is there a career counseling service for students?",
     "The career center offers CV reviews, mock interviews and job fairs every semester.",
     "Career Services"),
    ("What is the minimum GPA to stay in good academic standing?",
     "Students must maintain a cumulative GPA of at least 2.0 to remain in good standing.",
     "Academics"),
    ("How do I get a student ID card?",
     "Visit the student affairs desk with your admission letter and a passport-size photo.",
     "Campus Services"),
    ("Can I pay my tuition fee in installments?",
     "Yes, tuition can be paid in three installments if you apply through the finance office.",
     "Finance"),
]

# 3. Store documents, metadata and embeddings (Chroma embeds automatically)
collection.add(
    ids=[f"faq_{i}" for i in range(len(faqs))],
    documents=[f"{q} {a}" for q, a, _ in faqs],
    metadatas=[
        {"title": q, "category": cat, "source": "University FAQ Handbook"}
        for q, _, cat in faqs
    ],
)
print(f"Stored {collection.count()} records in the vector database.\n")

# 4. Similarity search with questions worded differently from the stored text
queries = [
    "I forgot my login for the online portal",
    "Where can I live if I'm a freshman?",
    "Is there help with paying school fees over time?",
    "What time can I study at the library on Sunday?",
]

for query in queries:
    results = collection.query(query_texts=[query], n_results=3)
    print("=" * 70)
    print(f"Query: {query}")
    for rank, (doc, meta, dist) in enumerate(
        zip(results["documents"][0], results["metadatas"][0], results["distances"][0]), 1
    ):
        print(f"  {rank}. [{meta['category']}] {meta['title']}")
        print(f"     distance: {dist:.3f}  (lower = more similar)")
print()

# 5. Bonus: combine semantic search with a metadata filter
query = "How do I pay?"
filtered = collection.query(
    query_texts=[query], n_results=2, where={"category": "Finance"}
)
print("=" * 70)
print(f"Filtered query (category = Finance): {query}")
for meta in filtered["metadatas"][0]:
    print(f"  - {meta['title']}")
