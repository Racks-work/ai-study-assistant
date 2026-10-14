"""Document parsing, grounded retrieval, and offline study aids."""
from __future__ import annotations
import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

MAX_FILE_BYTES = 12 * 1024 * 1024

@dataclass(frozen=True)
class Chunk:
    source: str
    location: str
    text: str


def extract_document(filename: str, data: bytes) -> list[Chunk]:
    """Read PDF or UTF-8 TXT/MD, keeping page/line provenance."""
    if len(data) > MAX_FILE_BYTES:
        raise ValueError('File exceeds 12 MB limit.')
    suffix = Path(filename).suffix.lower()
    if suffix == '.pdf':
        try:
            reader = PdfReader(io.BytesIO(data))
            if reader.is_encrypted:
                raise ValueError('Password-protected PDFs are not supported.')
            pages = [(f'page {i}', p.extract_text() or '') for i, p in enumerate(reader.pages, 1)]
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(f'Could not read PDF: {exc}') from exc
    elif suffix in ('.txt', '.md'):
        try:
            raw = data.decode('utf-8-sig')
        except UnicodeDecodeError as exc:
            raise ValueError('Text documents must be UTF-8 encoded.') from exc
        pages = [('document', raw)]
    else:
        raise ValueError('Only PDF, TXT, and MD files are supported.')
    result = []
    for location, text in pages:
        result.extend(chunk_text(text, filename, location))
    return result


def chunk_text(text: str, source: str, location: str, size: int = 950, overlap: int = 130) -> list[Chunk]:
    if size < 1 or not (0 <= overlap < size):
        raise ValueError('Invalid chunk parameters.')
    cleaned = re.sub(r'\s+', ' ', text).strip()
    if not cleaned:
        return []
    chunks = []
    start = 0
    while start < len(cleaned):
        end = min(start + size, len(cleaned))
        if end < len(cleaned):
            boundary = cleaned.rfind(' ', start + size // 2, end)
            if boundary > start:
                end = boundary
        part = cleaned[start:end].strip()
        if part:
            chunks.append(Chunk(source, location, part))
        if end == len(cleaned):
            break
        start = max(start + 1, end - overlap)
    return chunks


def retrieve(query: str, chunks: list[Chunk], top_k: int = 4) -> list[tuple[Chunk, float]]:
    """TF-IDF cosine similarity; return no matches when there is no lexical overlap."""
    if not query.strip() or not chunks:
        return []
    texts = [c.text for c in chunks]
    try:
        vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
        matrix = vectorizer.fit_transform(texts + [query])
    except ValueError:
        return []
    scores = cosine_similarity(matrix[-1], matrix[:-1]).ravel()
    order = scores.argsort()[::-1]
    return [(chunks[int(i)], float(scores[i])) for i in order[:top_k] if scores[i] > 0.01]


def offline_answer(question: str, matches: list[tuple[Chunk, float]]) -> str:
    if not matches:
        return 'I could not find relevant information in the uploaded documents. Try a more specific question.'
    lines = ['**Relevant excerpts from your documents** (offline mode; not an AI-generated synthesis):']
    for i, (chunk, _) in enumerate(matches, 1):
        excerpt = chunk.text[:650] + ('…' if len(chunk.text) > 650 else '')
        lines.append(f'**[{i}] {chunk.source} — {chunk.location}:** {excerpt}')
    return '\n\n'.join(lines)


def make_quiz(chunks: Iterable[Chunk], count: int = 5) -> list[dict[str, str]]:
    """Deterministic cloze questions based on literal source sentences."""
    result = []
    seen = set()
    for chunk in chunks:
        sentences = re.split(r'(?<=[.!?])\s+', chunk.text)
        for sentence in sentences:
            words = re.findall(r"\b[A-Za-z][A-Za-z-]{5,}\b", sentence)
            if not (45 <= len(sentence) <= 240 and len(words) >= 3):
                continue
            candidates = [w for w in words if w.lower() not in {'because', 'during', 'through', 'between', 'another', 'however', 'therefore', 'important', 'example', 'different'}]
            if not candidates:
                continue
            answer = max(candidates, key=len)
            key = (sentence.lower(), answer.lower())
            if key in seen:
                continue
            seen.add(key)
            question = re.sub(r'\b' + re.escape(answer) + r'\b', '________', sentence, count=1)
            result.append({'question': question, 'answer': answer, 'source': f'{chunk.source} — {chunk.location}'})
            if len(result) >= count:
                return result
    return result
