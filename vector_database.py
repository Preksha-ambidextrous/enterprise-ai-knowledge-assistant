
import os
from dotenv import load_dotenv
from google import genai
import chromadb

# Load environment variables
load_dotenv()

# Initialize Gemini
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("Gemini API key not found!")

client = genai.Client(api_key=api_key)

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(
    path="./chroma_db"
)

# Create a collection
collection = chroma_client.get_or_create_collection(
    name="company_knowledge"
)

# Sample company documents
documents = [
    "Employees receive 12 days of personal leave annually.",
    "Employees can work from home two days per week.",
    "The company provides health insurance to all full-time employees.",
    "Employees receive 20 days of annual vacation leave.",
    "The office working hours are 9 AM to 6 PM."
]

ids = ["doc1", "doc2", "doc3", "doc4", "doc5"]

# Generate embeddings for documents
response = client.models.embed_content(
    model="gemini-embedding-001",
    contents=documents
)

embeddings = [
    item.values for item in response.embeddings
]

# Store documents and embeddings in ChromaDB
collection.upsert(
    ids=ids,
    documents=documents,
    embeddings=embeddings
)

print("Documents successfully stored in ChromaDB!")

# Search for relevant information
question = "How many personal leave days can I take?"

query_response = client.models.embed_content(
    model="gemini-embedding-001",
    contents=question
)

query_embedding = query_response.embeddings[0].values

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=2
)

print("\nRelevant documents:")

for document in results["documents"][0]:
    print("-", document)
