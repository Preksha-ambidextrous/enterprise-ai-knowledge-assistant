
import hashlib
import time
from pathlib import Path

from google.genai.errors import ClientError
from pypdf import PdfReader

from rag_service import client, collection

DOCUMENTS_DIR = Path("documents")
DOCUMENTS_DIR.mkdir(exist_ok=True)

MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
BATCH_SIZE = 20


def process_pdf(filename: str, file_bytes: bytes):

    if len(file_bytes) > MAX_FILE_SIZE:
        raise ValueError("PDF exceeds the 20 MB upload limit.")

    if not file_bytes.startswith(b"%PDF-"):
        raise ValueError("The uploaded file is not a valid PDF.")

    # Generate a unique document ID from file contents
    document_id = hashlib.sha256(file_bytes).hexdigest()

    safe_filename = Path(filename).name

    # Check for duplicates among newly indexed documents
    existing = collection.get(
        where={"document_id": document_id},
        limit=1
    )

    if existing["ids"]:
        return {
            "filename": safe_filename,
            "status": "already_exists",
            "new_chunks": 0,
            "message": "This document is already indexed."
        }

    # Check older legacy documents without document IDs
    legacy_records = collection.get(
        where={"source": safe_filename},
        include=["metadatas"]
    )

    legacy_file = DOCUMENTS_DIR / safe_filename

    if legacy_records["ids"] and legacy_file.exists():

        legacy_hash = hashlib.sha256(
            legacy_file.read_bytes()
        ).hexdigest()

        has_legacy_chunks = any(
            not metadata.get("document_id")
            for metadata in legacy_records["metadatas"]
        )

        if has_legacy_chunks and legacy_hash == document_id:
            return {
                "filename": safe_filename,
                "status": "already_exists",
                "new_chunks": 0,
                "message": "This document is already indexed."
            }

    # Save uploaded PDF
    
    file_path = DOCUMENTS_DIR / f"{document_id[:12]}_{safe_filename}"
    file_path.write_bytes(file_bytes)

    try:
        reader = PdfReader(str(file_path))

        chunks = []
        metadatas = []
        ids = []

        # Extract text and create chunks page by page
        for page_number, page in enumerate(reader.pages, start=1):

            page_text = page.extract_text() or ""

            if not page_text.strip():
                continue

            start = 0
            chunk_number = 0

            while start < len(page_text):

                end = start + CHUNK_SIZE
                chunk = page_text[start:end].strip()

                if chunk:
                    chunks.append(chunk)

                    metadatas.append({
                        "source": safe_filename,
                        "page": page_number,
                        "document_id": document_id
                    })

                    ids.append(
                        f"{document_id[:20]}_p{page_number}_c{chunk_number}"
                    )

                    chunk_number += 1

                start += CHUNK_SIZE - CHUNK_OVERLAP

        if not chunks:
            file_path.unlink(missing_ok=True)
            raise ValueError(
                "No extractable text found in this PDF."
            )

        # Generate embeddings and store in batches
        for start in range(0, len(chunks), BATCH_SIZE):

            end = start + BATCH_SIZE

            batch_chunks = chunks[start:end]
            batch_metadata = metadatas[start:end]
            batch_ids = ids[start:end]

            response = None

            for attempt in range(5):
                try:
                    response = client.models.embed_content(
                        model="gemini-embedding-001",
                        contents=batch_chunks
                    )
                    break

                except ClientError as error:

                    if error.code not in (429, 503) or attempt == 4:
                        raise

                    wait_time = min(10 * (attempt + 1), 40)

                    print(
                        f"Embedding service unavailable. "
                        f"Retrying in {wait_time} seconds..."
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

            if end < len(chunks):
                time.sleep(15)

        return {
            "filename": safe_filename,
            "status": "indexed",
            "pages": len(reader.pages),
            "new_chunks": len(chunks),
            "message": "PDF indexed successfully."
        }

    
    except Exception:
        # Remove any partially indexed chunks
        try:
            collection.delete(
                where={"document_id": document_id}
            )
        except Exception as cleanup_error:
            print(f"ChromaDB cleanup failed: {cleanup_error}")

        # Remove the uploaded PDF
        file_path.unlink(missing_ok=True)

        raise



# --------------------------------
# List indexed documents
# --------------------------------

def list_documents():

    result = collection.get(
        include=["metadatas"]
    )

    documents = {}

    for metadata in result["metadatas"]:

        document_id = metadata.get("document_id")
        filename = metadata.get("source", "Unknown")

        # Support older documents without document_id
        if document_id:
            key = document_id
        else:
            key = f"legacy:{filename}"

        if key not in documents:

            documents[key] = {
                "document_id": key,
                "filename": filename,
                "chunks": 0,
                "pages": set()
            }

        documents[key]["chunks"] += 1
        documents[key]["pages"].add(
            metadata.get("page")
        )

    document_list = []

    for document in documents.values():

        document["pages"] = len(
            document["pages"] - {None}
        )

        document_list.append(document)

    return document_list


# --------------------------------
# Delete indexed document
# --------------------------------

def delete_document(document_id):
    if document_id.startswith("legacy:"):
        filename = document_id[len("legacy:"):]
        safe_filename = Path(filename).name

        if not safe_filename or safe_filename != filename:
            raise ValueError("Invalid document filename.")

        legacy_records = collection.get(
            where={"source": safe_filename}
        )

        legacy_ids = [
            record_id
            for record_id in legacy_records["ids"]
            if record_id.startswith("page_")
        ]

        if not legacy_ids:
            raise ValueError("Legacy document not found.")

        collection.delete(ids=legacy_ids)

        old_file = DOCUMENTS_DIR / safe_filename
        old_file.unlink(missing_ok=True)

    else:
        existing = collection.get(
            where={"document_id": document_id},
            limit=1
        )

        if not existing["ids"]:
            raise ValueError("Document not found.")

        collection.delete(
            where={"document_id": document_id}
        )

        matching_files = list(
            DOCUMENTS_DIR.glob(f"{document_id[:12]}_*")
        )

        for file_path in matching_files:
            file_path.unlink(missing_ok=True)

    return {
        "document_id": document_id,
        "status": "deleted",
        "message": "Document and its indexed chunks deleted successfully."
    }