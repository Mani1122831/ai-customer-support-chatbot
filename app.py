import os
import time
import re

import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pypdf import PdfReader


# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="Smartify AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    st.error("Gemini API key not found. Check your .env file.")
    st.stop()

client = genai.Client(api_key=API_KEY)

MODEL = "gemini-3.8-flash"


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "document_chunks" not in st.session_state:
    st.session_state.document_chunks = []

if "document_embeddings" not in st.session_state:
    st.session_state.document_embeddings = []

if "document_names" not in st.session_state:
    st.session_state.document_names = []

if "quick_question" not in st.session_state:
    st.session_state.quick_question = None


# ============================================================
# PDF FUNCTIONS
# ============================================================

def extract_pdf_text(uploaded_files):
    documents = []

    for uploaded_file in uploaded_files:

        reader = PdfReader(uploaded_file)

        for page_number, page in enumerate(reader.pages, start=1):

            text = page.extract_text() or ""
            text = re.sub(r"\s+", " ", text).strip()

            if text:
                documents.append(
                    {
                        "file": uploaded_file.name,
                        "page": page_number,
                        "text": text,
                    }
                )

    return documents


def make_chunks(documents, words_per_chunk=180):

    chunks = []

    for document in documents:

        words = document["text"].split()

        for start in range(0, len(words), words_per_chunk):

            piece = " ".join(
                words[start:start + words_per_chunk]
            )

            if piece.strip():

                chunks.append(
                    {
                        "file": document["file"],
                        "page": document["page"],
                        "text": piece,
                    }
                )

    return chunks


def create_embeddings(client, chunks, model="gemini-embedding-2"):
    """Create one Gemini embedding per document chunk."""
    if not chunks:
        return []

    all_embeddings = []
    batch_size = 20

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]

        contents = [
            types.Content(
                parts=[
                    types.Part.from_text(text=chunk["text"])
                ]
            )
            for chunk in batch
        ]

        response = client.models.embed_content(
            model=model,
            contents=contents,
            config=types.EmbedContentConfig(
                output_dimensionality=768
            ),
        )

        if not response.embeddings:
            raise RuntimeError("No embeddings were returned by Gemini.")

        if len(response.embeddings) != len(batch):
            raise RuntimeError(
                "Gemini returned an unexpected number of embeddings."
            )

        for embedding in response.embeddings:
            all_embeddings.append(list(embedding.values))

    return all_embeddings


def semantic_search(
    client,
    question,
    chunks,
    document_embeddings,
    top_k=4,
    model="gemini-embedding-2",
):
    """Find the most relevant document chunks using cosine similarity."""
    if not chunks or not document_embeddings:
        return []

    query_contents = types.Content(
        parts=[
            types.Part.from_text(text=question)
        ]
    )

    response = client.models.embed_content(
        model=model,
        contents=[query_contents],
        config=types.EmbedContentConfig(
            output_dimensionality=768
        ),
    )

    if not response.embeddings:
        return []

    query = list(response.embeddings[0].values)

    def cosine_similarity(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(y * y for y in b) ** 0.5

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot / (norm_a * norm_b)

    scored = []

    for index, vector in enumerate(document_embeddings):
        if index >= len(chunks):
            break

        score = cosine_similarity(query, vector)
        scored.append((score, chunks[index]))

    scored.sort(
        key=lambda item: item[0],
        reverse=True
    )

    return [
        item[1]
        for item in scored[:top_k]
    ]


def find_relevant_chunks(question, chunks, top_k=4):

    if not chunks:
        return []

    question_words = set(
        word.lower()
        for word in re.findall(r"[a-zA-Z0-9]+", question)
        if len(word) > 2
    )

    scored = []

    for chunk in chunks:

        chunk_words = set(
            word.lower()
            for word in re.findall(
                r"[a-zA-Z0-9]+",
                chunk["text"]
            )
            if len(word) > 2
        )

        score = len(
            question_words.intersection(chunk_words)
        )

        scored.append(
            (score, chunk)
        )

    scored.sort(
        key=lambda item: item[0],
        reverse=True
    )

    return [
        item[1]
        for item in scored[:top_k]
        if item[0] > 0
    ]


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ================================
       MAIN PAGE
    ================================= */

    .stApp {
        background:
            radial-gradient(
                circle at 12% 5%,
                rgba(124, 58, 237, 0.22),
                transparent 28%
            ),
            radial-gradient(
                circle at 90% 5%,
                rgba(59, 130, 246, 0.14),
                transparent 25%
            ),
            linear-gradient(
                180deg,
                #070a14 0%,
                #0a1020 100%
            );
    }

    header {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    #MainMenu {
        visibility: hidden;
    }

    .block-container {
        max-width: 1400px;
        padding-top: 30px;
        padding-bottom: 150px;
    }


    /* ================================
       SIDEBAR
    ================================= */

    section[data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                #080d18 0%,
                #0b1120 100%
            );

        border-right:
            1px solid rgba(255,255,255,0.07);
    }

    section[data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }


    /* ================================
       BUTTONS
    ================================= */

    .stButton > button {
        width: 100%;

        min-height: 44px;

        border-radius: 12px;

        background: #10182a;

        color: #e2e8f0;

        border:
            1px solid
            rgba(139,92,246,0.18);

        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        background: #191738;

        color: white;

        border-color:
            rgba(139,92,246,0.45);

        transform: translateY(-1px);
    }


    /* ================================
       CONTAINERS
    ================================= */

    [data-testid="stVerticalBlockBorderWrapper"] {
        background:
            linear-gradient(
                145deg,
                rgba(18,28,48,0.94),
                rgba(10,17,31,0.94)
            );

        border:
            1px solid
            rgba(255,255,255,0.08);

        border-radius: 20px;

        padding: 3px;
    }


    /* ================================
       METRICS
    ================================= */

    [data-testid="stMetric"] {
        background:
            linear-gradient(
                145deg,
                #131d33,
                #0e1628
            );

        border:
            1px solid
            rgba(255,255,255,0.08);

        border-radius: 18px;

        padding: 18px;

        min-height: 120px;

        box-shadow:
            0 10px 30px rgba(0,0,0,0.18);
    }

    [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
    }

    [data-testid="stMetricValue"] {
        color: #c4b5fd !important;

        font-weight: 800 !important;
    }


    /* ================================
       HEADINGS
    ================================= */

    h1 {
        color: white !important;

        font-weight: 850 !important;
    }

    h2 {
        color: white !important;
    }

    h3 {
        color: #f8fafc !important;
    }


    /* ================================
       CHAT
    ================================= */

    [data-testid="stChatMessage"] {
        background: transparent !important;
    }

    [data-testid="stChatMessageContent"] {
        color: #e2e8f0;
    }


    /* ================================
       CHAT INPUT
    ================================= */

    .stChatInput > div {
        background: white !important;

        border:
            2px solid
            #8b5cf6 !important;

        border-radius:
            18px !important;

        box-shadow:
            0 10px 35px
            rgba(76,29,149,0.26) !important;
    }

    .stChatInput textarea {
        color: #111827 !important;

        background: white !important;

        font-size: 16px !important;
    }

    .stChatInput textarea::placeholder {
        color: #64748b !important;

        opacity: 1 !important;
    }

    .stChatInput button {
        background: #7c3aed !important;

        color: white !important;

        border-radius: 10px !important;
    }


    /* ================================
       FILE UPLOADER
    ================================= */

    [data-testid="stFileUploader"] {
        background: #101827;

        border:
            1px solid
            rgba(255,255,255,0.07);

        border-radius: 15px;

        padding: 7px;
    }


    /* ================================
       SOURCE
    ================================= */

    .source {
        color: #93c5fd;

        font-size: 12px;

        margin-top: 4px;
    }


    /* ================================
       DIVIDER
    ================================= */

    hr {
        border-color:
            rgba(255,255,255,0.07) !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🤖 Smartify AI")

    st.caption(
        "Customer Support Copilot"
    )

    st.success(
        "🟢 AI system online"
    )

    st.divider()

    st.subheader(
        "📚 Knowledge Base"
    )

    uploaded_files = st.file_uploader(
        "Upload company PDFs",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if uploaded_files:

        names = [
            file.name
            for file in uploaded_files
        ]

        if names != st.session_state.document_names:

            with st.spinner(
                "Reading and understanding documents..."
            ):

                documents = extract_pdf_text(
                    uploaded_files
                )

                chunks = make_chunks(documents)

                try:
                    embeddings = create_embeddings(
                        client,
                        chunks
                    )

                    st.session_state.document_chunks = chunks
                    st.session_state.document_embeddings = embeddings
                    st.session_state.document_names = names

                except Exception as error:

                    st.session_state.document_chunks = []
                    st.session_state.document_embeddings = []
                    st.session_state.document_names = []

                    st.error(
                        "Could not create document embeddings."
                    )

                    st.caption(str(error))

        st.success(
            f"{len(uploaded_files)} document(s) ready"
        )

        for file in uploaded_files:

            st.caption(
                f"📄 {file.name}"
            )

    else:

        st.caption(
            "Upload FAQs, manuals, policies, "
            "or product documents."
        )

    st.divider()

    st.subheader(
        "⚡ Quick Actions"
    )

    if st.button(
        "📦 Track my order",
        use_container_width=True
    ):

        st.session_state.quick_question = (
            "How can I track my order?"
        )

    if st.button(
        "↩️ Return policy",
        use_container_width=True
    ):

        st.session_state.quick_question = (
            "What is the return policy?"
        )

    if st.button(
        "💳 Payment help",
        use_container_width=True
    ):

        st.session_state.quick_question = (
            "I have a payment problem. What should I do?"
        )

    if st.button(
        "📞 Contact support",
        use_container_width=True
    ):

        st.session_state.quick_question = (
            "How can I contact customer support?"
        )

    st.divider()

    if st.button(
        "🗑️ Clear conversation",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()


# ============================================================
# HEADER
# ============================================================

st.caption(
    "SMARTIFY AI  •  CUSTOMER SUPPORT"
)

st.title(
    "AI Support Copilot"
)

st.write(
    "Give customers fast, helpful answers "
    "using your company knowledge."
)


# ============================================================
# HERO
# ============================================================

with st.container(border=True):

    st.subheader(
        "⚡ Intelligent customer support"
    )

    st.write(
        "Upload company documents and ask "
        "questions in natural language. "
        "Smartify finds relevant information "
        "before generating an answer."
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.info("⚡ Gemini AI")

    with c2:
        st.info("📄 Document grounded")

    with c3:
        st.info("🔒 Knowledge base")


st.write("")


# ============================================================
# METRICS
# ============================================================

m1, m2, m3, m4 = st.columns(4)

with m1:

    st.metric(
        "AI assistance",
        "Smart"
    )

with m2:

    st.metric(
        "Availability",
        "24/7"
    )

with m3:

    st.metric(
        "Knowledge sections",
        str(
            len(
                st.session_state.document_chunks
            )
        )
    )

with m4:

    st.metric(
        "AI engine",
        "Gemini"
    )


# ============================================================
# WELCOME
# ============================================================

st.write("")

with st.container(border=True):

    st.subheader(
        "👋 Welcome to Smartify"
    )

    if st.session_state.document_chunks:

        st.write(
            f"Your knowledge base contains "
            f"{len(st.session_state.document_chunks)} "
            "searchable sections. "
            "Ask a question to begin."
        )

    else:

        st.write(
            "Upload a company PDF from the sidebar "
            "to enable document-grounded answers."
        )


# ============================================================
# SUGGESTED QUESTIONS
# ============================================================

st.subheader(
    "✨ Suggested questions"
)

st.caption(
    "Choose a common customer request."
)

q1, q2, q3 = st.columns(3)

with q1:

    if st.button(
        "📦 Track my order",
        use_container_width=True
    ):

        st.session_state.quick_question = (
            "How can I track my order?"
        )

with q2:

    if st.button(
        "↩️ Return policy",
        use_container_width=True
    ):

        st.session_state.quick_question = (
            "What is the return policy?"
        )

with q3:

    if st.button(
        "💳 Payment help",
        use_container_width=True
    ):

        st.session_state.quick_question = (
            "I have a payment problem. What should I do?"
        )


# ============================================================
# CHAT
# ============================================================

st.subheader(
    "💬 Customer conversation"
)

st.caption(
    "Ask your AI support assistant."
)


for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

        if message.get("source"):

            st.caption(
                f"📄 Source: {message['source']}"
            )


# ============================================================
# INPUT
# ============================================================

prompt = st.chat_input(
    "Type your question..."
)

if (
    st.session_state.quick_question
    and not prompt
):

    prompt = (
        st.session_state.quick_question
    )

    st.session_state.quick_question = None


# ============================================================
# AI
# ============================================================

if prompt:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    with st.chat_message(
        "user"
    ):

        st.markdown(prompt)


    relevant = semantic_search(
        client,
        prompt,
        st.session_state.document_chunks,
        st.session_state.document_embeddings,
        top_k=4
    )


    if relevant:

        context = "\n\n".join(
            [
                (
                    f"Document: {item['file']}\n"
                    f"Page: {item['page']}\n"
                    f"{item['text']}"
                )
                for item in relevant
            ]
        )

        source = (
            f"{relevant[0]['file']} "
            f"· page {relevant[0]['page']}"
        )

    else:

        context = (
            "No relevant company document information "
            "was found."
        )

        source = None


    instructions = """
You are Smartify AI, a professional customer support assistant.

Use the company information provided in the prompt.

Never invent company-specific:
- prices
- policies
- refund rules
- delivery promises
- contact information
- product specifications

If the answer cannot be found in the company information,
say:

"I couldn't find that information in the uploaded
company documents."

Keep the answer concise, clear, polite, and helpful.
"""


    with st.chat_message(
        "assistant"
    ):

        placeholder = st.empty()

        answer_parts = []

        try:

            stream = client.models.generate_content_stream(
                model=MODEL,
                contents=(
                    "COMPANY INFORMATION:\n\n"
                    f"{context}\n\n"
                    "CUSTOMER QUESTION:\n\n"
                    f"{prompt}"
                ),
                config={
                    "system_instruction": instructions,
                    "max_output_tokens": 300,
                },
            )


            for chunk in stream:

                if chunk.text:

                    answer_parts.append(
                        chunk.text
                    )

                    placeholder.markdown(
                        "".join(answer_parts)
                        + "▌"
                    )


            answer = "".join(
                answer_parts
            ).strip()

            if not answer:

                answer = (
                    "I couldn't generate an answer. "
                    "Please try again."
                )


            placeholder.markdown(
                answer
            )


            if source:

                st.caption(
                    f"📄 Source: {source}"
                )


            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "source": source,
                }
            )


        except Exception as error:

            error_text = str(error)

            # Temporary overload / rate-limit handling
            if (
                "503" in error_text
                or "429" in error_text
                or "unavailable" in error_text.lower()
                or "resource_exhausted"
                in error_text.lower()
            ):

                time.sleep(2)

                try:

                    response = client.models.generate_content(
                        model=MODEL,
                        contents=(
                            "COMPANY INFORMATION:\n\n"
                            f"{context}\n\n"
                            "CUSTOMER QUESTION:\n\n"
                            f"{prompt}"
                        ),
                        config={
                            "system_instruction":
                                instructions,
                            "max_output_tokens": 250,
                        },
                    )

                    answer = response.text.strip()

                    placeholder.markdown(
                        answer
                    )

                    if source:

                        st.caption(
                            f"📄 Source: {source}"
                        )

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "source": source,
                        }
                    )

                except Exception:

                    st.warning(
                        "Gemini is temporarily busy. "
                        "Please try again in a few seconds."
                    )

            else:

                st.error(
                    "Something went wrong while "
                    "generating the response."
                )

                st.caption(
                    error_text
                )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Smartify AI • Gemini-powered customer support"
)