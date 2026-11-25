
import os
from dotenv import load_dotenv
from google import genai
import chromadb

# 1. Load API key
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("Gemini API key not found!")

# 2. Initialize Gemini
client = genai.Client(api_key=api_key)

# 3. Connect to existing ChromaDB
chroma_client = chromadb.PersistentClient(
    path="./chroma_db"
)

# 4. Access the existing collection
collection = chroma_client.get_collection(
    name="employee_handbook"
)

print("Connected to employee handbook database!")
print("Total stored documents:", collection.count())


# 5. Define search function
def search_handbook(question):

    # Convert question into embedding
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=question
    )

    query_embedding = response.embeddings[0].values

    # Search ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
        include=["documents", "metadatas", "distances"]
    )

    # Display results
    print("\nQuestion:", question)
    print("\nRelevant handbook sections:\n")

    for i, document in enumerate(results["documents"][0]):

        metadata = results["metadatas"][0][i]
        distance = results["distances"][0][i]

        print(f"\n--- Result {i + 1} ---")
        print("Source:", metadata["source"])
        print("Page:", metadata["page"])
        print("Distance:", round(distance, 4))
        print("Content:", document)


# 6. Ask the user for a question
while True:

    question = input(
        "\nAsk a question about the handbook (type exit to quit): "
    )

    if question.lower().strip() == "exit":
        print("Goodbye!")
        break

    if not question.strip():
        print("Please enter a question.")
        continue

    search_handbook(question)
