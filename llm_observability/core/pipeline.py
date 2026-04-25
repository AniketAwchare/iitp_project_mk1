"""
pipeline.py — FastAPI serving endpoint + LangChain RAG pipeline.
Designed to run on Google Colab / Kaggle with 4-bit quantized Mistral 7B.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .config import Config
from .logger import QueryLogger

logger = logging.getLogger(__name__)

# ── Request / Response schemas ────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str
    use_rag: bool = True
    metadata: Optional[Dict[str, Any]] = None


class QueryResponse(BaseModel):
    id: str
    query: str
    response: str
    retrieved_context: List[str] = []


# ── Pipeline ──────────────────────────────────────────────────────────

class LLMPipeline:
    """
    Wraps model loading, RAG retrieval, and generation in one object.
    Call load_model() and load_rag() before serving requests.
    """

    def __init__(self, config: Config):
        self.config  = config
        self.qlogger = QueryLogger(config.log_path)
        self.model      = None
        self.tokenizer  = None
        self.retriever  = None
        self._app: Optional[FastAPI] = None

    # ── Model ─────────────────────────────────────────────────────────

    def load_model(self) -> None:
        """Load 4-bit quantized LLM (Colab T4 / Kaggle P100 compatible)."""
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )

        logger.info("Loading tokenizer: %s", self.config.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name, trust_remote_code=True
        )

        quant_cfg = None
        if self.config.model_quantize_4bit:
            quant_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

        logger.info("Loading model weights …")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=quant_cfg,
            device_map=self.config.device,
            trust_remote_code=True,
        )
        self.model.eval()
        logger.info("Model ready.")

    # ── RAG ───────────────────────────────────────────────────────────

    def load_rag(self, documents: Optional[List[str]] = None) -> None:
        """Build or reload a FAISS vector store from financial documents."""
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_community.vectorstores import FAISS
        from langchain.schema import Document
        from langchain.text_splitter import RecursiveCharacterTextSplitter

        emb = HuggingFaceEmbeddings(model_name=self.config.embedding_model)
        idx_path = self.config.vector_store_path

        if os.path.exists(idx_path):
            logger.info("Loading existing FAISS index from %s", idx_path)
            vs = FAISS.load_local(idx_path, emb, allow_dangerous_deserialization=True)
        elif documents:
            logger.info("Building FAISS index from %d documents …", len(documents))
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
            )
            raw_docs = [Document(page_content=t) for t in documents]
            chunks   = splitter.split_documents(raw_docs)
            vs = FAISS.from_documents(chunks, emb)
            vs.save_local(idx_path)
            logger.info("FAISS index saved: %d chunks", len(chunks))
        else:
            logger.warning("No documents and no existing index — RAG disabled.")
            return

        self.retriever = vs.as_retriever(
            search_kwargs={"k": self.config.top_k_retrieval}
        )

    # ── Core inference ────────────────────────────────────────────────

    def retrieve(self, query: str) -> List[str]:
        if self.retriever is None:
            return []
        docs = self.retriever.get_relevant_documents(query)
        return [d.page_content for d in docs]

    def generate(self, query: str, context: Optional[List[str]] = None) -> str:
        if self.model is None:
            raise RuntimeError("Call load_model() first.")
        import torch

        ctx_block = ""
        if context:
            ctx_block = "\n\nRelevant context:\n" + "\n---\n".join(context)

        prompt = (
            "[INST] You are a precise financial services assistant. "
            "Use only the provided context when answering. "
            "If the context is insufficient, say so clearly."
            f"{ctx_block}\n\nQuestion: {query} [/INST]"
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                do_sample=self.config.temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(
            out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        ).strip()

    def query(self, query_text: str, use_rag: bool = True,
              metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Full pipeline: retrieve → generate → log → return."""
        context = self.retrieve(query_text) if use_rag else []
        response = self.generate(query_text, context)
        record   = self.qlogger.log(
            query=query_text,
            response=response,
            retrieved_context=context,
            metadata=metadata or {},
        )
        return {"id": record["id"], "query": query_text,
                "response": response, "retrieved_context": context}

    # ── FastAPI app ───────────────────────────────────────────────────

    def get_app(self) -> FastAPI:
        if self._app is not None:
            return self._app

        app = FastAPI(
            title="LLM Observability — Financial Services API",
            version="1.0.0",
            description="Mistral 7B + RAG endpoint with built-in observability logging.",
        )

        pipeline = self  # capture for route closures

        @app.get("/health")
        def health():
            return {"status": "ok", "model": pipeline.config.model_name}

        @app.post("/query", response_model=QueryResponse)
        def handle_query(req: QueryRequest):
            try:
                result = pipeline.query(req.query, req.use_rag, req.metadata)
                return QueryResponse(**result)
            except Exception as exc:
                logger.exception("Query failed")
                raise HTTPException(status_code=500, detail=str(exc))

        self._app = app
        return app
