
from typing import Literal

from pydantic import BaseModel, Field
from rag_service import ask_handbook, collection

from fastapi import FastAPI, HTTPException, UploadFile, File
from document_service import process_pdf, list_documents, delete_document


app = FastAPI(
    title="Enterprise AI Knowledge Assistant",
    description="AI-powered company document question-answering API",
    version="2.0.0"
)


# A single conversation message
class ChatMessage(BaseModel):

    role: Literal["user", "assistant"]

    content: str


# Question and previous conversation
class QuestionRequest(BaseModel):

    question: str = Field(min_length=1)

    history: list[ChatMessage] = Field(
        default_factory=list
    )

    answer_mode: Literal["strict", "explain"] = "strict"

@app.get("/")
def home():

    return {
        "message": "Enterprise AI Knowledge Assistant API is running!"
    }


@app.get("/health")
def health():

    return {
        "status": "healthy",
        "knowledge_base_records": collection.count()
    }


@app.post("/ask")
def ask_question(request: QuestionRequest):

    try:

        history = [
            message.model_dump()
            for message in request.history
        ]

        result = ask_handbook(
            request.question,
            history,
            request.answer_mode
        )

        return {
            "question": request.question,
            "answer": result["answer"],
            "sources": result["sources"]
        }

    except ValueError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except Exception as error:

        raise HTTPException(
            status_code=503,
            detail=str(error)
        )


# --------------------------------
# List indexed documents
# --------------------------------

@app.get("/documents")
def get_documents():

    try:
        documents = list_documents()

        return {
            "total_documents": len(documents),
            "documents": documents
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


# --------------------------------
# Delete an indexed document
# --------------------------------

@app.delete("/documents/{document_id}")
def remove_document(document_id: str):

    try:
        return delete_document(document_id)

    except ValueError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error)
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()

        result = process_pdf(
            filename=file.filename,
            file_bytes=file_bytes
        )

        return result

    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))

    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@app.post("/documents/{document_id}/replace")
async def replace_document(
    document_id: str,
    file: UploadFile = File(...)
):
    # 1. Confirm the old document exists
    existing_documents = list_documents()

    if not any(
        doc["document_id"] == document_id
        for doc in existing_documents
    ):
        raise HTTPException(
            status_code=404,
            detail="Document to replace was not found."
        )

    # 2. Read the new PDF
    file_bytes = await file.read()

    try:
        # 3. Index the new PDF FIRST
        result = process_pdf(
            filename=file.filename,
            file_bytes=file_bytes
        )

        # Do not delete the old version if the new one
        # was already indexed or was not newly processed.
        if result.get("status") != "indexed":
            raise HTTPException(
                status_code=409,
                detail=(
                    "Replacement was not performed. "
                    "The uploaded PDF is already indexed."
                )
            )

        # 4. Delete the old document only after indexing succeeds
        try:
            delete_document(document_id)

        except Exception as error:
            raise HTTPException(
                status_code=500,
                detail=(
                    "New PDF was indexed successfully, "
                    "but the old PDF could not be deleted. "
                    "Both versions have been retained. "
                    f"Details: {error}"
                )
            )

        # 5. Return success
        return {
            "status": "replaced",
            "old_document_id": document_id,
            "new_document": result,
            "message": (
                "Document replaced successfully. "
                "The new version is indexed."
            )
        }

    except HTTPException:
        raise

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )

    
"""@app.get("/debug/document/{document_id}")
def debug_document(document_id: str):
    result = collection.get(
        where={"document_id": document_id},
        include=["metadatas"]
    )

    return {
        "searched_id": document_id,
        "matching_chunks": len(result["ids"]),
        "sample_ids": result["ids"][:5],
        "sample_metadata": result["metadatas"][:2]
    }
    

@app.delete("/debug/delete/{document_id}")
def debug_delete_document(document_id: str):
    try:
        collection.delete(
            where={"document_id": document_id}
        )

        remaining = collection.get(
            where={"document_id": document_id}
        )

        return {
            "document_id": document_id,
            "remaining_chunks": len(remaining["ids"]),
            "status": "success" if not remaining["ids"] else "failed"
        }

    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))"""