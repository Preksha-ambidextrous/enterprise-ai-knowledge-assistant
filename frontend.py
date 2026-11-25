import datetime
import requests
import streamlit as st

# ============================================================
# Configuration
# ============================================================

st.set_page_config(
    page_title="Enterprise AI Knowledge Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_URL = "http://127.0.0.1:8000"
ASK_URL = f"{BASE_URL}/ask"
HEALTH_URL = f"{BASE_URL}/health"
DOCUMENTS_URL = f"{BASE_URL}/documents"
UPLOAD_URL = f"{BASE_URL}/upload"

# ============================================================
# Visual style
# ============================================================

st.markdown(
    """
    <style>
    :root {
        --brand: #2563eb;
        --soft: #eff6ff;
        --muted: #64748b;
    }
    .block-container {
        max-width: 1120px;
        padding-top: 4.5rem;
        padding-bottom: 7rem;
    }
    .hero-kicker {
        color: #2563eb;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: .12em;
        text-transform: uppercase;
        margin-bottom: .4rem;
    }
    .hero-title {
        font-size: clamp(2rem, 4vw, 3rem);
        line-height: 1.12;
        font-weight: 750;
        letter-spacing: -0.045em;
        margin: 0 0 .65rem 0;
    }
    .hero-subtitle {
        color: #64748b;
        font-size: 1.05rem;
        max-width: 720px;
        line-height: 1.6;
    }
    .small-muted {
        color: #64748b;
        font-size: .88rem;
    }
    [data-testid="stChatMessage"] {
        border: 1px solid rgba(148, 163, 184, .18);
        border-radius: 16px;
        padding: .85rem 1rem;
    }
    [data-testid="stChatInput"] {
        border-radius: 18px;
    }
    section[data-testid="stSidebar"] {
        border-right: 1px solid rgba(148, 163, 184, .18);
    }
    div[data-testid="stMetric"] {
        background: rgba(148, 163, 184, .08);
        padding: .7rem .85rem;
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# Session state
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

if "delete_candidate" not in st.session_state:
    st.session_state.delete_candidate = None

# ============================================================
# Helpers
# ============================================================

def get_greeting():
    hour = datetime.datetime.now().hour
    if 5 <= hour < 12:
        return "Good morning"
    if 12 <= hour < 17:
        return "Good afternoon"
    if 17 <= hour < 22:
        return "Good evening"
    return "Hello"


def fetch_health():
    response = requests.get(HEALTH_URL, timeout=8)
    response.raise_for_status()
    return response.json()


def fetch_documents():
    response = requests.get(DOCUMENTS_URL, timeout=15)
    response.raise_for_status()
    return response.json().get("documents", [])


def render_sources(sources):
    if not sources:
        return
    with st.expander("📚 View document sources", expanded=False):
        for source in sources:
            st.markdown(
                f"**{source.get('document', 'Unknown document')}**  \n"
                f"Page {source.get('page', 'Unknown')}"
            )


def submit_question(question, answer_mode):
    question = (question or "").strip()
    if not question:
        return

    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    history = [
        {"role": item["role"], "content": item["content"]}
        for item in st.session_state.messages[:-1]
    ]

    try:
        with st.spinner("Searching your documents and preparing an answer..."):
            response = requests.post(
                ASK_URL,
                json={
                    "question": question,
                    "history": history,
                    "answer_mode": answer_mode,
                },
                timeout=(10, 300),
            )
            response.raise_for_status()
            result = response.json()

        answer = result.get("answer", "The assistant returned an empty answer.")
        sources = result.get("sources", [])
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": sources,
            }
        )
    except requests.exceptions.ConnectionError:
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": (
                    "I can't reach the FastAPI backend right now. "
                    "Please check that the backend is running, then try again."
                ),
                "sources": [],
            }
        )
    except requests.exceptions.Timeout:
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": "The request took too long. Please try again.",
                "sources": [],
            }
        )
    except requests.exceptions.HTTPError as error:
        try:
            detail = response.json().get("detail", str(error))
        except Exception:
            detail = str(error)
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": f"API error: {detail}",
                "sources": [],
            }
        )
    except Exception as error:
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": f"Something went wrong: {error}",
                "sources": [],
            }
        )


# ============================================================
# Sidebar: settings, knowledge base, upload and management
# ============================================================

with st.sidebar:
    st.markdown("## 🤖 Enterprise AI")
    st.caption("Your company knowledge, one conversation away.")
    st.divider()

    st.markdown("### Response style")
    answer_mode = st.selectbox(
        "Choose how answers are written",
        options=["strict", "explain"],
        format_func=lambda value: (
            "📄 Strict document mode" if value == "strict"
            else "💡 Explain mode"
        ),
        help=(
            "Strict mode answers only from indexed documents. "
            "Explain mode can add clearly labelled general explanations."
        ),
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("### Knowledge base")

    backend_ok = False
    try:
        health = fetch_health()
        backend_ok = True
        st.success("Backend connected")
        st.metric("Indexed chunks", health.get("knowledge_base_records", 0))
    except requests.exceptions.RequestException:
        st.error("Backend is offline")
        st.caption("Start FastAPI in its own terminal.")

    st.divider()
    
    st.subheader("Upload Documents")

    uploaded_files = st.file_uploader(
        "Choose PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
        key="document_uploader"
    )

    if st.button("Process Documents", use_container_width=True):

        if not uploaded_files:
            st.warning("Please select at least one PDF.")

        else:
            for uploaded_file in uploaded_files:

                with st.status(
                    f"Uploading {uploaded_file.name}...",
                    expanded=True
                ) as status:

                    try:
                        st.write("Validating PDF...")
                        st.write("Generating embeddings and indexing...")

                        response = requests.post(
                            "http://127.0.0.1:8000/upload",
                            files={
                                "file": (
                                    uploaded_file.name,
                                    uploaded_file.getvalue(),
                                    "application/pdf"
                                )
                            },
                            timeout=600
                        )

                        response.raise_for_status()
                        result = response.json()

                        if result["status"] == "already_exists":

                            status.update(
                                label=f"File already exists: {uploaded_file.name}",
                                state="complete",
                                expanded=True
                            )

                            st.warning(
                                f"**{uploaded_file.name}** is already in "
                                "your Knowledge Base. No new chunks were added."
                            )

                        elif result["status"] == "indexed":

                            status.update(
                                label=f"Upload successful: {uploaded_file.name}",
                                state="complete",
                                expanded=True
                            )

                            st.success(
                                f"**{uploaded_file.name}** indexed successfully. "
                                f"{result['new_chunks']} chunks added."
                            )

                        else:
                            status.update(
                                label=f"Unexpected response: {uploaded_file.name}",
                                state="error"
                            )
                            st.error(str(result))

                    except requests.exceptions.HTTPError:
                        try:
                            details = response.json().get(
                                "detail", response.text
                            )
                        except ValueError:
                            details = response.text

                        status.update(
                            label=f"Upload failed: {uploaded_file.name}",
                            state="error"
                        )

                        st.error(str(details))

                    except Exception as error:
                        status.update(
                            label=f"Upload failed: {uploaded_file.name}",
                            state="error"
                        )

                        st.error(str(error))

    st.divider()
    st.markdown("### Manage documents")

    if st.button("↻ Refresh document list", use_container_width=True):
        st.rerun()

    if backend_ok:
        try:
            documents = fetch_documents()
            if not documents:
                st.caption("No documents have been indexed yet.")
            else:
                st.caption(f"{len(documents)} indexed document(s)")
                for index, document in enumerate(documents):
                    document_id = document["document_id"]
                    filename = document.get("filename", "Unknown PDF")
                    with st.expander(f"📄 {filename}", expanded=False):
                        col_a, col_b = st.columns(2)
                        col_a.metric("Pages", document.get("pages", 0))
                        col_b.metric("Chunks", document.get("chunks", 0))

                        st.markdown("**Replace this PDF**")
                        replacement_file = st.file_uploader(
                            "Choose the updated PDF",
                            type=["pdf"],
                            key=f"replacement_file_{index}_{document_id}",
                            label_visibility="collapsed",
                        )

                        if st.button(
                            "Replace document",
                            key=f"replace_{index}_{document_id}",
                            use_container_width=True,
                            type="primary",
                            disabled=not backend_ok or replacement_file is None,
                        ):
                            try:
                                with st.spinner(
                                    f"Indexing the updated version of {filename}..."
                                ):
                                    replace_response = requests.post(
                                        f"{DOCUMENTS_URL}/{document_id}/replace",
                                        files={
                                            "file": (
                                                replacement_file.name,
                                                replacement_file.getvalue(),
                                                "application/pdf",
                                            )
                                        },
                                        timeout=600,
                                    )
                                    replace_response.raise_for_status()
                                    replace_result = replace_response.json()

                                st.success(
                                    replace_result.get(
                                        "message",
                                        f"{filename} was replaced successfully.",
                                    )
                                )
                                

                            except requests.exceptions.HTTPError as error:
                                try:
                                    detail = replace_response.json().get(
                                        "detail", str(error)
                                    )
                                except Exception:
                                    detail = str(error)
                                st.error(f"Replacement failed: {detail}")

                            except requests.exceptions.RequestException as error:
                                st.error(
                                    "Could not complete replacement. "
                                    "Check the backend connection. "
                                    f"Details: {error}"
                                )

                        st.divider()
                        confirm_delete = st.checkbox(
                            "I understand this permanently deletes the PDF and its indexed chunks",
                            key=f"confirm_delete_{index}_{document_id}",
                        )

                        if st.button(
                            "Delete document",
                            key=f"delete_{index}_{document_id}",
                            use_container_width=True,
                            disabled=not confirm_delete,
                        ):
                            try:
                                delete_response = requests.delete(
                                    f"{DOCUMENTS_URL}/{document_id}",
                                    timeout=60,
                                )
                                delete_response.raise_for_status()
                                st.success(f"{filename} was deleted.")
                                st.rerun()
                            except requests.exceptions.RequestException as error:
                                try:
                                    detail = delete_response.json().get("detail", str(error))
                                except Exception:
                                    detail = str(error)
                                st.error(f"Delete failed: {detail}")
        except requests.exceptions.RequestException:
            st.warning("Document list is unavailable. Check the backend.")

    st.divider()
    if st.button("＋ New conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.caption("Built with Gemini · FastAPI · ChromaDB")


# ============================================================
# Main chat area
# ============================================================

if not st.session_state.messages:
    st.markdown('<div class="hero-kicker">COMPANY KNOWLEDGE ASSISTANT</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="hero-title">{get_greeting()}!<br>How can I help you today?</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="hero-subtitle">'
        'Ask a question about your company documents. I can find relevant information, '
        'explain policies, and show you exactly which document and page support an answer.'
        '</div>',
        unsafe_allow_html=True,
    )

    st.write("")
    st.markdown("#### Try asking")
    suggestion_columns = st.columns(2)
    suggestions = [
        ("📘", "Summarize the key points in the employee handbook."),
        ("🧭", "What are the important company policies?"),
        ("🔎", "Find the information about leave and holidays."),
        ("💡", "Explain a policy with a practical example."),
    ]
    for index, (icon, suggestion) in enumerate(suggestions):
        with suggestion_columns[index % 2]:
            if st.button(
                f"{icon}  {suggestion}",
                key=f"suggestion_{index}",
                use_container_width=True,
            ):
                st.session_state.pending_question = suggestion
                st.rerun()
else:
    st.markdown("## Conversation")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                render_sources(message.get("sources", []))

# Chat composer
typed_question = st.chat_input(
    "Message your knowledge assistant...",
    key="chat_composer",
)

question_to_send = typed_question
if st.session_state.pending_question:
    question_to_send = st.session_state.pending_question
    st.session_state.pending_question = None

if question_to_send:
    submit_question(question_to_send, answer_mode)
    st.rerun()

st.markdown(
    '<div class="small-muted" style="text-align:center;">'
    'AI-generated answers can make mistakes. Check the cited document pages for important decisions.'
    '</div>',
    unsafe_allow_html=True,
)
