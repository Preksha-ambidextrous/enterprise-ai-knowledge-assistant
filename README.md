# Enterprise AI Knowledge Assistant

An AI-powered document question-answering system that allows users to upload PDF documents and ask questions using Retrieval-Augmented Generation (RAG).

The application uses Google Gemini, ChromaDB, FastAPI, and Streamlit to retrieve relevant information from uploaded documents and generate context-aware answers with source and page citations.

## Key Features

- **PDF Document Upload:** Upload and index multiple PDF documents.
- **RAG-Based Question Answering:** Retrieve relevant document chunks before generating answers.
- **Source Citations:** Display source filenames and page numbers.
- **Strict Document Mode:** Answers only using information available in uploaded documents.
- **Explain Mode:** Provides more detailed explanations based on retrieved document context.
- **Conversation History:** Supports contextual follow-up questions.
- **Document Management:** Upload, replace, and delete documents.
- **Duplicate Detection:** Prevents indexing identical documents multiple times.
- **Relevance Filtering:** Rejects unrelated questions when relevant document information is unavailable.

## Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| LLM | Google Gemini |
| Embeddings | Gemini Embedding API |
| Vector Database | ChromaDB |
| Backend | FastAPI |
| Frontend | Streamlit |
| PDF Processing | Python PDF processing libraries |
| Environment Management | Python Virtual Environment |

## Architecture

1. **Document Upload:** Users upload PDF documents through the Streamlit interface.
2. **PDF Processing:** Extract text, divide it into chunks, and prepare it for embedding.
3. **Embedding Generation:** Generate vector embeddings using Gemini.
4. **Vector Storage:** Store document chunks and embeddings in ChromaDB.
5. **Question Processing:** Convert user questions into embeddings.
6. **Similarity Search:** Retrieve the most relevant document chunks.
7. **Answer Generation:** Send retrieved context to Gemini to generate an answer.
8. **Source Attribution:** Display supporting document names and page numbers.

## Project Structure

```text
enterprise-ai-assistant/
│
├── api.py
├── app.py
├── frontend.py
├── document_service.py
├── pdf_processor.py
├── rag_chat.py
├── rag_service.py
├── vector_database.py
├── ingest_documents.py
├── search_documents.py
│
├── documents/
├── chroma_db/
│
├── .env
├── .env.example
├── .gitignore
├── requirements.txt
│
└── README.md
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone <your-github-repository-url>
cd enterprise-ai-assistant
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key
```

Get your API key from Google AI Studio.

### 5. Start the FastAPI backend

```bash
uvicorn api:app --reload
```

Backend:

```text
http://127.0.0.1:8000
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

### 6. Start the Streamlit frontend

Open another terminal, activate the virtual environment, and run:

```bash
streamlit run frontend.py
```

Open the local URL displayed in your terminal.

## Usage

1. Launch the application.
2. Upload one or more PDF documents.
3. Wait for document indexing to complete.
4. Select the preferred answer mode.
5. Ask questions about the uploaded documents.
6. Review answers with source and page citations.
7. Manage documents using the replace and delete options.

## Security

- API credentials are stored in `.env`.
- `.env` is excluded through `.gitignore`.
- Uploaded documents and the local vector database are excluded from Git.
- `.env.example` provides a configuration template without exposing credentials.

## Future Enhancements

- Support for Word, Excel, and other document formats.
- User authentication and role-based access control.
- Cloud-based vector database deployment.
- Advanced document analytics.
- Multi-user enterprise deployment.
- Improved monitoring and evaluation of RAG responses.

## Author

Developed as an Enterprise AI Knowledge Assistant project using Python, Generative AI, Retrieval-Augmented Generation, and vector database technologies.
