import logging
import re
import time
from typing import List, Dict, Any, Optional
import json
from sentence_transformers import SentenceTransformer
from groq import Groq
from ..core.config import settings
from ..core.database import db_manager
from ..services.cache import rag_cache
from ..services.hallucination_validator import hallucination_validator
from ..core.utils import text_processor

logger = logging.getLogger(__name__)


class LightweightLLMRAG:
    """RAG engine with local embeddings + Groq LLM"""

    def __init__(self):
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.groq_client = None
        self.initialized = False

    # ---------------- INITIALIZATION ---------------- #

    async def initialize(self):
        if self.initialized:
            return

        if not settings.groq_api_key:
            raise ValueError("Groq API key missing")

        self.groq_client = Groq(api_key=settings.groq_api_key)

        await self._load_documents_from_database()

        self.initialized = True
        logger.info("RAG initialized successfully")

    async def _load_documents_from_database(self):
        async with db_manager.get_connection() as conn:
            result = await conn.fetchrow("SELECT COUNT(*) as count FROM documents")
            logger.info(f"Documents: {result['count']}")

    # ---------------- EMBEDDINGS ---------------- #

    async def _generate_embedding(self, text: str) -> Optional[List[float]]:
        try:
            text = text[: settings.max_text_length]
            embedding = self.embedding_model.encode(text)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Embedding error: {e}")
            return None

    async def _store_embedding(self, doc_id: int, embedding: List[float]):
        async with db_manager.get_connection() as conn:
            emb_str = "[" + ",".join(map(str, embedding)) + "]"

            await conn.execute(
                "UPDATE documents SET embedding=$1::vector WHERE id=$2",
                emb_str,
                doc_id,
            )

    # ---------------- INGESTION ---------------- #

    async def add_documents_bulk(self, documents: List[Dict[str, Any]]):
        """
        Add multiple documents into DB with embeddings
        """
        try:
            async with db_manager.get_connection() as conn:

                inserted_ids = []

                for doc in documents:
                    content = doc.get("content")
                    if not content:
                        continue

                    title = doc.get("title", "Untitled")
                    source = doc.get("source", "unknown")

                    metadata = json.dumps(doc.get("metadata", {}))

                    # Insert document
                    result = await conn.fetchrow(
                        """
                        INSERT INTO documents (content, title, source, metadata, status)
                        VALUES ($1, $2, $3, $4, 'pending')
                        RETURNING id
                        """,
                        content,
                        title,
                        source,
                        metadata,
                    )

                    doc_id = result["id"]

                    # Generate embedding
                    embedding = await self._generate_embedding(content)

                    if embedding:
                        emb_str = "[" + ",".join(map(str, embedding)) + "]"

                        await conn.execute(
                            """
                            UPDATE documents
                            SET embedding=$1::vector, status='processed'
                            WHERE id=$2
                            """,
                            emb_str,
                            doc_id,
                        )

                    inserted_ids.append(str(doc_id))

                return inserted_ids

        except Exception as e:
            logger.error(f"Bulk ingestion failed: {e}")
            raise

    # ---------------- RETRIEVAL ---------------- #

    @staticmethod
    def _expand_short_query(query: str) -> str:
        """Dense retrieval often needs more than bare keywords."""
        q = query.strip()
        if len(q.split()) <= 2:
            return f"What is {q} in legal context?"
        return q

    @staticmethod
    def _parse_metadata_row(raw: Any) -> Dict[str, Any]:
        if raw is None:
            return {}
        if isinstance(raw, dict):
            return dict(raw)
        if isinstance(raw, str):
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return {}
        return {}

    def _document_from_row(
        self,
        row: Any,
        score: float,
        *,
        vector_score: Optional[float] = None,
    ) -> Dict[str, Any]:
        meta = self._parse_metadata_row(row["metadata"])
        title = row.get("title") or meta.get("title") or "Untitled"
        source = row.get("source") or meta.get("source") or "unknown"
        meta = {
            **meta,
            "title": title,
            "source": source,
            "similarity_score": float(score),
        }
        out: Dict[str, Any] = {
            "id": row["id"],
            "content": row["content"] or "",
            "score": float(score),
            "title": title,
            "source": source,
            "metadata": meta,
        }
        if vector_score is not None:
            out["vector_score"] = float(vector_score)
        return out

    async def _vector_similarity_search_raw(
        self, query_embedding: List[float], limit: int
    ) -> List[Dict[str, Any]]:
        async with db_manager.get_connection() as conn:
            q_emb = "[" + ",".join(map(str, query_embedding)) + "]"

            rows = await conn.fetch(
                """
                SELECT id, content, title, source, metadata,
                       1 - (embedding <=> $1::vector) AS score
                FROM documents
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> $1::vector
                LIMIT $2
                """,
                q_emb,
                limit,
            )

            return [self._document_from_row(r, r["score"]) for r in rows]

    async def _keyword_search(self, query: str, limit: int) -> List[Dict[str, Any]]:
        q = query.strip()
        if len(q) < 2:
            return []
        async with db_manager.get_connection() as conn:
            try:
                rows = await conn.fetch(
                    """
                    SELECT id, content, title, source, metadata,
                           ts_rank_cd(
                             to_tsvector('english', content),
                             websearch_to_tsquery('english', $1)
                           ) AS kw_score
                    FROM documents
                    WHERE to_tsvector('english', content)
                          @@ websearch_to_tsquery('english', $1)
                    ORDER BY kw_score DESC NULLS LAST
                    LIMIT $2
                    """,
                    q,
                    limit,
                )
            except Exception as e:
                logger.warning(f"Keyword search skipped: {e}")
                return []

        if not rows:
            return []
        max_kw = max((r["kw_score"] or 0.0) for r in rows) or 1.0
        out: List[Dict[str, Any]] = []
        for r in rows:
            kw = float(r["kw_score"] or 0.0)
            # Map into a 0–1 band comparable to cosine similarity for threshold logic
            norm = min(0.95, 0.25 + 0.7 * (kw / max_kw))
            out.append(self._document_from_row(r, norm, vector_score=None))
        return out

    @staticmethod
    def _rrf_merge(
        vec_docs: List[Dict[str, Any]],
        kw_docs: List[Dict[str, Any]],
        *,
        top_n: int,
        k: int = 60,
    ) -> List[Dict[str, Any]]:
        """
        Reciprocal rank fusion so keyword hits are not dropped when vector search
        already returned many rows (vec-then-kw merge starved FTS).
        """
        scores: Dict[int, float] = {}
        by_id: Dict[int, Dict[str, Any]] = {}
        for rank, d in enumerate(vec_docs):
            did = d["id"]
            if did not in by_id:
                by_id[did] = d
            scores[did] = scores.get(did, 0.0) + 1.0 / (k + rank + 1)
        for rank, d in enumerate(kw_docs):
            did = d["id"]
            if did not in by_id:
                by_id[did] = d
            scores[did] = scores.get(did, 0.0) + 1.0 / (k + rank + 1)
        ordered_ids = sorted(scores.keys(), key=lambda i: scores[i], reverse=True)
        return [by_id[i] for i in ordered_ids[:top_n]]

    @staticmethod
    def _extract_article_number(query: str) -> Optional[str]:
        m = re.search(r"(?i)\barticle\s+(\d{1,4})\b", query.strip())
        if not m:
            return None
        return m.group(1)

    async def _fetch_chunks_for_article_number(
        self, article_num: str, limit: int
    ) -> List[Dict[str, Any]]:
        if not re.fullmatch(r"\d{1,4}", article_num):
            return []
        n = int(article_num)
        # Avoid matching Article 10 when looking for Article 1 (POSIX regex, PG ~*)
        pattern = rf"\marticle\s+{n}([^0-9]|$)"
        async with db_manager.get_connection() as conn:
            try:
                rows = await conn.fetch(
                    """
                    SELECT id, content, title, source, metadata,
                           0.99::float AS score
                    FROM documents
                    WHERE content IS NOT NULL
                      AND content ~* $1::text
                    ORDER BY id ASC
                    LIMIT $2
                    """,
                    pattern,
                    limit,
                )
            except Exception as e:
                logger.warning(f"Article-number retrieval failed: {e}")
                return []
        return [self._document_from_row(r, float(r["score"])) for r in rows]

    @staticmethod
    def _prepend_unique(
        preferred: List[Dict[str, Any]], rest: List[Dict[str, Any]], max_len: int
    ) -> List[Dict[str, Any]]:
        seen = set()
        out: List[Dict[str, Any]] = []
        for d in preferred + rest:
            did = d["id"]
            if did in seen:
                continue
            seen.add(did)
            out.append(d)
            if len(out) >= max_len:
                break
        return out

    @staticmethod
    def _apply_threshold_with_fallback(
        docs: List[Dict[str, Any]], threshold: float, top_k: int
    ) -> List[Dict[str, Any]]:
        passing = [d for d in docs if d.get("score", 0) >= threshold]
        if len(passing) >= top_k:
            return passing[:top_k]
        rest = [d for d in docs if d not in passing]
        combined = passing + rest
        return combined[:top_k]

    async def retrieve_documents(
        self,
        query: str,
        top_k: int = 5,
        algorithm: str = "hybrid",
        similarity_threshold: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve ranked documents without calling the LLM.
        Used by adaptive RAG and can be tuned independently from answer generation.
        """
        vector_query = self._expand_short_query(query)
        query_emb = await self._generate_embedding(vector_query)
        if not query_emb:
            return []

        fetch_cap = min(max(top_k * 4, top_k), 48)
        article_n = self._extract_article_number(query)
        article_boost: List[Dict[str, Any]] = []
        if article_n:
            article_boost = await self._fetch_chunks_for_article_number(
                article_n, limit=min(8, fetch_cap)
            )

        if algorithm == "keyword_only":
            kw = await self._keyword_search(query.strip(), fetch_cap)
            merged = self._prepend_unique(article_boost, kw, max_len=fetch_cap)
            return self._apply_threshold_with_fallback(merged, similarity_threshold, top_k)

        vec_docs = await self._vector_similarity_search_raw(query_emb, fetch_cap)

        if algorithm == "vector_only":
            merged = self._prepend_unique(article_boost, vec_docs, max_len=fetch_cap)
        else:
            kw_docs = await self._keyword_search(query.strip(), fetch_cap)
            rrf = self._rrf_merge(
                vec_docs, kw_docs, top_n=max(fetch_cap, top_k * 2)
            )
            merged = self._prepend_unique(article_boost, rrf, max_len=fetch_cap)

        return self._apply_threshold_with_fallback(merged, similarity_threshold, top_k)

    async def _vector_similarity_search(self, query_embedding, top_k=5, threshold=0.3):
        """Backward-compatible name: threshold + cap applied."""
        docs = await self._vector_similarity_search_raw(query_embedding, top_k * 4)
        return self._apply_threshold_with_fallback(docs, threshold, top_k)

    # ---------------- QUERY ---------------- #

    async def query(
        self,
        query: str,
        top_k: int = 5,
        algorithm: str = "hybrid",
        similarity_threshold: float = 0.3,
        **kwargs,
    ):
        start = time.time()

        # cache
        cached = await rag_cache.get_rag_query(query, algorithm)
        if cached:
            return cached

        docs = await self.retrieve_documents(
            query=query,
            top_k=top_k,
            algorithm=algorithm,
            similarity_threshold=similarity_threshold,
        )

        if not docs:
            return {
                "response": "No strong match found in uploaded documents. Try a more specific legal query.",
                "sources": [],
            }

        # LLM response
        response = await self._generate_llm_response(query, docs)

        result = {
            "response": response,
            "sources": docs,
            "processing_time": time.time() - start,
        }

        await rag_cache.cache_rag_query(query, result, algorithm)

        return result

    def _build_llm_context(self, docs: List[Dict[str, Any]]) -> str:
        budget = max(2000, getattr(settings, "rag_max_total_context_chars", 12000))
        n = len(docs)
        per = max(800, budget // n)
        parts: List[str] = []
        for i, d in enumerate(docs, 1):
            body = (d.get("content") or "")[:per]
            label = d.get("title") or d.get("metadata", {}).get("title", f"Passage {i}")
            parts.append(f"[{label}]\n{body}")
        return "\n\n---\n\n".join(parts)

    async def _generate_llm_response(self, query: str, docs: List[Dict[str, Any]]) -> str:
        try:
            context = self._build_llm_context(docs)

            prompt = f"""
    You are a legal AI assistant.

    Context:
    {context}

    Question:
    {query}

    Give a clear and concise legal answer based on the context. If the context does not contain the answer, say so explicitly.
    """

            response = self.groq_client.chat.completions.create(
                model=settings.groq_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=800,
            )

            answer = response.choices[0].message.content.strip()

            reject, _ = hallucination_validator.should_reject_response(
                answer, context, query
            )

            if reject:
                return (
                    "I could not find exact information in the uploaded documents. "
                    "However, based on general legal knowledge:\n\n"
                    + text_processor.clean_text_comprehensive(answer)
                )

            return text_processor.clean_text_comprehensive(answer)

        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            return "Error generating response"


# Singleton instance
lightweight_llm_rag = LightweightLLMRAG()
