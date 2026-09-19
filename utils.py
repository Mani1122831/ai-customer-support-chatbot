import re
import numpy as np
from pypdf import PdfReader


def extract_pdf_text(uploaded_files):
    """
    Extract text from uploaded PDF files.
    Keeps file name and page number for citations.
    """

    documents = []

    for uploaded_file in uploaded_files:

        reader = PdfReader(uploaded_file)

        for page_number, page in enumerate(
            reader.pages,
            start=1
        ):

            text = page.extract_text() or ""

            text = re.sub(
                r"\s+",
                " ",
                text
            ).strip()

            if text:

                documents.append(
                    {
                        "file": uploaded_file.name,
                        "page": page_number,
                        "text": text,
                    }
                )

    return documents


def make_chunks(
    documents,
    words_per_chunk=180
):
    """
    Split PDF text into smaller searchable chunks.
    """

    chunks = []

    for document in documents:

        words = document["text"].split()

        for start in range(
            0,
            len(words),
            words_per_chunk
        ):

            piece = " ".join(
                words[
                    start:start + words_per_chunk
                ]
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


def create_embeddings(
    client,
    chunks,
    model="gemini-embedding-2"
):
    """
    Create Gemini embeddings for all document chunks.
    """

    if not chunks:
        return []

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    response = client.models.embed_content(
        model=model,
        contents=texts,
    )

    embeddings = []

    for embedding in response.embeddings:

        embeddings.append(
            np.array(
                embedding.values,
                dtype=np.float32
            )
        )

    return embeddings


def search_embeddings(
    client,
    question,
    chunks,
    document_embeddings,
    top_k=4,
    model="gemini-embedding-2"
):
    """
    Find the most semantically relevant document chunks.
    """

    if not chunks or not document_embeddings:

        return []

    query_response = client.models.embed_content(
        model=model,
        contents=question,
    )

    query_vector = np.array(
        query_response.embeddings[0].values,
        dtype=np.float32
    )

    query_norm = np.linalg.norm(
        query_vector
    )

    if query_norm == 0:

        return []

    scores = []

    for index, doc_vector in enumerate(
        document_embeddings
    ):

        doc_norm = np.linalg.norm(
            doc_vector
        )

        if doc_norm == 0:

            score = 0

        else:

            score = float(
                np.dot(
                    query_vector,
                    doc_vector
                )
                /
                (
                    query_norm *
                    doc_norm
                )
            )

        scores.append(
            (
                score,
                chunks[index]
            )
        )

    scores.sort(
        key=lambda item: item[0],
        reverse=True
    )

    return [
        item[1]
        for item in scores[:top_k]
    ]