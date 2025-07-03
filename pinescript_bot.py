# Advanced PineScriptBot with improved PDF analysis and context handling
import os
import re
import logging
from io import BytesIO
from datetime import datetime
from typing import List, Dict

import fitz  # PyMuPDF
import PyPDF2
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class PineScriptBot:
    """Pine Script assistant that relies on uploaded PDF references."""

    def __init__(self, telegram_token: str, openai_api_key: str, db_path: str = "./chroma_db"):
        self.telegram_token = telegram_token
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.pdf_collection = self.chroma_client.get_or_create_collection(
            name="pdf_content",
            metadata={"hnsw:space": "cosine"}
        )
        self.pdf_content = ""
        self.conversation_memory: Dict[str, List[Dict]] = {}
        self.max_memory_messages = 200
        self.analysis_depth = 30

    # ------------------------------------------------------------------
    # PDF utilities
    # ------------------------------------------------------------------
    def extract_pdf_text(self, pdf_bytes: bytes) -> str:
        """Extract text from PDF using PyMuPDF with PyPDF2 fallback."""
        text = ""
        try:
            pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
            for page_num in range(len(pdf_document)):
                page = pdf_document.load_page(page_num)
                text += page.get_text()
                text += "\n"
            pdf_document.close()
        except Exception as exc:
            logger.warning("PyMuPDF failed: %s", exc)
            try:
                pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))
                for page in pdf_reader.pages:
                    text += page.extract_text() or ""
                    text += "\n"
            except Exception as e2:
                logger.error("PyPDF2 failed: %s", e2)
                raise e2
        return text

    def smart_chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
        """Create overlapping text chunks that respect headings and code fences."""
        lines = text.splitlines()
        chunks: List[str] = []
        current: List[str] = []
        length = 0
        for line in lines:
            if re.match(r"^\s*#+", line) or line.startswith("```"):
                if current:
                    chunks.append("\n".join(current))
                    current = []
                    length = 0
            current.append(line)
            length += len(line)
            if length >= chunk_size:
                chunks.append("\n".join(current))
                current = []
                length = 0
        if current:
            chunks.append("\n".join(current))
        final_chunks = []
        for i, chunk in enumerate(chunks):
            start = max(i - 1, 0)
            merged = "\n".join([chunks[start], chunk]) if start != i else chunk
            final_chunks.append(merged)
        return final_chunks

    def store_pdf_content(self, content: str, filename: str) -> None:
        """Store the PDF content into the vector DB for later retrieval."""
        self.pdf_content = content
        chunks = self.smart_chunk_text(content)
        embeddings = self.embedding_model.encode(chunks).tolist()
        ids = [f"{filename}_{i}" for i in range(len(chunks))]
        metadatas = [{"source": filename, "index": i} for i in range(len(chunks))]
        self.pdf_collection.add(embeddings=embeddings, documents=chunks, metadatas=metadatas, ids=ids)
        logger.info("Stored %s chunks from %s", len(chunks), filename)

    # ------------------------------------------------------------------
    # Conversation memory helpers
    # ------------------------------------------------------------------
    def add_to_memory(self, user_id: str, role: str, content: str) -> None:
        history = self.conversation_memory.setdefault(user_id, [])
        history.append({"role": role, "content": content, "timestamp": datetime.utcnow().isoformat()})
        if len(history) > self.max_memory_messages:
            del history[0:len(history) - self.max_memory_messages]

    def get_history(self, user_id: str, last_n: int) -> List[Dict]:
        return self.conversation_memory.get(user_id, [])[-last_n:]

    # ------------------------------------------------------------------
    # Knowledge search
    # ------------------------------------------------------------------
    def search_knowledge(self, query: str, n_results: int = 5) -> List[str]:
        """Query the vector DB using the given text and a few synonyms."""
        # simple synonym expansion helps catch wording differences
        synonyms = {
            "rsi": ["relative strength index"],
            "ema": ["exponential moving average"],
            "sma": ["simple moving average"],
        }
        queries = [query]
        for key, words in synonyms.items():
            if key in query.lower():
                queries.extend(words)
        docs: List[str] = []
        for q in queries:
            emb = self.embedding_model.encode([q])[0].tolist()
            results = self.pdf_collection.query(query_embeddings=[emb], n_results=n_results)
            docs.extend(results.get('documents', [[]])[0])
        # remove duplicates while preserving order
        seen = set()
        unique_docs = []
        for d in docs:
            if d not in seen:
                unique_docs.append(d)
                seen.add(d)
            if len(unique_docs) >= n_results:
                break
        return unique_docs

    # ------------------------------------------------------------------
    # GPT interaction
    # ------------------------------------------------------------------
    def build_context(self, user_id: str, user_message: str) -> str:
        history = self.get_history(user_id, self.analysis_depth)
        summary = "\n".join([f"{m['role']}: {m['content'][:80]}" for m in history[-5:]])
        relevant = self.search_knowledge(user_message)
        return f"Conversation:\n{summary}\n\nPDF refs:\n" + "\n---\n".join(relevant)

    def generate_response(self, user_id: str, user_message: str) -> str:
        context = self.build_context(user_id, user_message)
        messages = [
            {"role": "system", "content": "You are a Pine Script expert. Always use the PDF provided by the user as the definitive reference."},
            {"role": "user", "content": context + "\n\n" + user_message}
        ]
        response = self.openai_client.chat.completions.create(
            model="gpt-4-turbo",
            messages=messages,
            max_tokens=800,
            temperature=0.3
        )
        answer = response.choices[0].message.content
        self.add_to_memory(user_id, "user", user_message)
        self.add_to_memory(user_id, "assistant", answer)
        return answer

