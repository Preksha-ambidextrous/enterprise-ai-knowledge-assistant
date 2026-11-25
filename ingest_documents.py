
import os
import time 
from google.genai.errors import ClientError
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from pypdf import PdfReader
import chromadb

# 1. Load environment variables
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("Gemini API key not found!")

client = genai.Client(api_key=api_key)

# 2. Read the PDF
pdf_path = Path("documents/employee_handbook.pdf")

if not pdf_path.exists():
    raise FileNotFoundError(f"PDF not found: {pdf_path}")

reader = PdfReader(str(pdf_path))

print("Total PDF pages:", len(reader.pages))

# 3. Extract text page by page
pages = []

for page_number, page in enumerate(reader.pages, start=1):
    text = page.extract_text() or ""

    if text.strip():
        pages.append({
            "page": page_number,
            "text": text
        })

print("Pages containing extractable text:", len(pages))

# 4. Split text into chunks
chunk_size = 1000
overlap = 200

chunks = []
metadatas = []
ids = []

for page in pages:
    text = page["text"]
    start = 0
    chunk_number = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

            metadatas.append({
                "source": pdf_path.name,
                "page": page_number,
                "document_id": document_id
            })

            ids.append(
                f"page_{page['page']}_chunk_{chunk_number}"
            )

            chunk_number += 1

        start += chunk_size - overlap

print("Total chunks created:", len(chunks))

# 5. Create persistent ChromaDB database
chroma_client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = chroma_client.get_or_create_collection(
    name="employee_handbook"
)


# 6. Generate embeddings and store them safely

# Check which chunks are already stored
existing_records = collection.get(ids=ids)
existing_ids = set(existing_records["ids"])

# Only process chunks that are not already stored
pending = [
    (chunk, metadata, doc_id)
    for chunk, metadata, doc_id in zip(chunks, metadatas, ids)
    if doc_id not in existing_ids
]

print("Already stored:", len(existing_ids))
print("Remaining to process:", len(pending))

batch_size = 20

for start in range(0, len(pending), batch_size):

    batch = pending[start:start + batch_size]

    batch_chunks = [item[0] for item in batch]
    batch_metadata = [item[1] for item in batch]
    batch_ids = [item[2] for item in batch]

    # Retry if Gemini rate limit is reached
    for attempt in range(5):
        try:
            response = client.models.embed_content(
                model="gemini-embedding-001",
                contents=batch_chunks
            )
            break

        except ClientError as error:
            if error.code != 429 or attempt == 4:
                raise

            wait_time = 10 * (attempt + 1)

            print(
                f"Rate limit reached. "
                f"Waiting {wait_time} seconds..."
            )

            time.sleep(wait_time)

    embeddings = [
        item.values for item in response.embeddings
    ]

    collection.upsert(
        ids=batch_ids,
        documents=batch_chunks,
        embeddings=embeddings,
        metadatas=batch_metadata
    )

    completed = min(start + batch_size, len(pending))

    print(
        f"Newly stored: {completed}/{len(pending)}"
    )

    # Stay below the free-tier rate limit
    if completed < len(pending):
        time.sleep(15)

print("\nPDF ingestion completed successfully!")

print("Total records in ChromaDB:", collection.count())
