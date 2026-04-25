# %% [markdown]
# # Notebook 01 — Infrastructure Setup
# **LLM Observability Framework · Financial Services**
# IIT Patna M.Tech AI & Data Science 2025-26
#
# **Runtime required:** T4 GPU (Google Colab) or P100 (Kaggle)
#
# This notebook:
# 1. Installs all dependencies
# 2. Loads FinanceBench + FiQA-2018 datasets
# 3. Builds a FAISS vector index from the financial corpus
# 4. Loads Mistral 7B (4-bit quantized)
# 5. Runs an end-to-end smoke test: query → RAG retrieve → generate → log

# %% [markdown]
# ## Cell 1 — Install Dependencies

# %%
# Check GPU availability first
import subprocess
result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
if result.returncode == 0:
    print("✅ GPU detected:")
    print(result.stdout[:500])
else:
    print("⚠️  No GPU detected — Mistral 7B will be slow. Switch runtime to T4.")

# %%
# Install all requirements (takes ~3-4 minutes on Colab)
# Run once per session
import sys

print("Installing dependencies...")
subprocess.run([
    sys.executable, "-m", "pip", "install", "-q",
    "transformers==4.40.0",
    "bitsandbytes==0.43.0",
    "accelerate==0.29.3",
    "sentence-transformers==2.7.0",
    "langchain==0.1.20",
    "langchain-community==0.0.38",
    "faiss-cpu==1.8.0",
    "datasets==2.19.0",
    "huggingface-hub==0.22.2",
    "evaluate==0.4.1",
    "rouge-score==0.1.2",
    "scipy==1.13.0",
    "scikit-learn==1.4.2",
    "mlflow==2.13.0",
    "fastapi==0.111.0",
    "uvicorn==0.29.0",
    "streamlit==1.33.0",
    "plotly==5.21.0",
    "loguru==0.7.2",
], check=True)

print("✅ Installation complete.")

# %% [markdown]
# ## Cell 2 — Clone Repository (if running from Colab directly)

# %%
import os

REPO_PATH = "/content/llm-observability-framework"

if not os.path.exists(REPO_PATH):
    # Replace with your GitHub URL after pushing
    GITHUB_URL = "https://github.com/<YOUR_USERNAME>/llm-observability-framework.git"
    print(f"Cloning from {GITHUB_URL} ...")
    subprocess.run(["git", "clone", GITHUB_URL, REPO_PATH], check=True)
    print("✅ Repo cloned.")
else:
    print("✅ Repo already present at", REPO_PATH)

# Add to Python path
if REPO_PATH not in sys.path:
    sys.path.insert(0, REPO_PATH)

os.chdir(REPO_PATH)
print("Working directory:", os.getcwd())

# %% [markdown]
# ## Cell 3 — Load Datasets (FinanceBench + FiQA-2018)

# %%
from llm_observability.data.loaders import load_financebench, load_fiqa

print("Loading FinanceBench...")
fb_docs, fb_qa = load_financebench(split="train", max_samples=300)
print(f"  FinanceBench: {len(fb_docs)} documents, {len(fb_qa)} QA pairs")

print("\nLoading FiQA-2018...")
fq_docs, fq_qa = load_fiqa(split="train", max_samples=300)
print(f"  FiQA: {len(fq_docs)} documents, {len(fq_qa)} QA pairs")

# Combine all documents for the RAG corpus
all_documents = fb_docs + fq_docs
print(f"\n✅ Total corpus: {len(all_documents)} documents")

# Preview a sample
print("\n--- Sample document (FinanceBench) ---")
print(fb_docs[0][:400])
print("\n--- Sample QA pair ---")
print(f"Q: {fb_qa[0]['question']}")
print(f"A: {fb_qa[0]['answer'][:200]}")

# %% [markdown]
# ## Cell 4 — Build FAISS Vector Index

# %%
from llm_observability.core.config import Config
from llm_observability.core.pipeline import LLMPipeline

config = Config(
    vector_store_path="./data/financial_corpus/faiss_index",
    embedding_model="all-MiniLM-L6-v2",
    chunk_size=512,
    chunk_overlap=64,
    top_k_retrieval=3,
    # Skip model loading for index build
    model_name="mistralai/Mistral-7B-Instruct-v0.2",
)

pipeline = LLMPipeline(config)

print("Building FAISS index from financial corpus...")
print(f"  Documents to index: {len(all_documents)}")
pipeline.load_rag(documents=all_documents)
print("✅ FAISS index built and saved to:", config.vector_store_path)

# Quick retrieval test (no LLM needed)
test_query = "What is Basel III Tier 1 capital requirement?"
chunks = pipeline.retrieve(test_query)
print(f"\n--- Retrieval test ---")
print(f"Query: {test_query}")
for i, chunk in enumerate(chunks):
    print(f"\nChunk {i+1}: {chunk[:200]}...")

# %% [markdown]
# ## Cell 5 — Load Mistral 7B (4-bit Quantized)
# ⚠️ Requires T4 GPU (15 GB VRAM). Takes ~3-4 minutes to load.

# %%
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# %%
print("Loading Mistral 7B (4-bit quantized)...")
print("This takes ~3-4 minutes on T4...")

pipeline.load_model()
print("✅ Model loaded successfully.")

# Memory check
if torch.cuda.is_available():
    used = torch.cuda.memory_allocated() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU memory used: {used:.1f} / {total:.1f} GB")

# %% [markdown]
# ## Cell 6 — End-to-End Smoke Test

# %%
from llm_observability.core.logger import QueryLogger

logger_obj = QueryLogger(log_path="./logs")

test_queries = [
    "What is the Basel III Tier 1 capital ratio requirement for banks?",
    "Explain the concept of dollar-cost averaging in equity investing.",
    "What are the key differences between term and whole life insurance?",
]

print("=" * 60)
print("END-TO-END SMOKE TEST")
print("=" * 60)

for query in test_queries:
    print(f"\n🔵 Query: {query}")

    # Retrieve context
    chunks = pipeline.retrieve(query)
    print(f"   Retrieved {len(chunks)} chunks from FAISS")

    # Generate response
    response = pipeline.generate(query, context=chunks)
    print(f"   Response: {response[:300]}...")

    # Log it
    record = logger_obj.log(
        query=query,
        response=response,
        retrieved_context=chunks,
        metadata={"source": "smoke_test", "model": config.model_name},
    )
    print(f"   Logged with ID: {record['id']}")

print("\n✅ Smoke test complete. All queries processed and logged.")

# Verify logs
logs = logger_obj.load_logs()
print(f"\nLog file contains {len(logs)} records.")
print(f"Latest log ID: {logs[-1]['id']}")

# %% [markdown]
# ## Cell 7 — Summary
#
# ✅ **Infrastructure setup complete:**
# - Dependencies installed
# - FinanceBench + FiQA-2018 datasets loaded
# - FAISS index built from {len(all_documents)} financial documents
# - Mistral 7B (4-bit) loaded on GPU
# - End-to-end pipeline tested
#
# **Next:** Open `colab_02_modules_demo.py` to demo all 4 observability modules.

# %%
print("Infrastructure setup complete!")
print(f"  Corpus size  : {len(all_documents)} documents")
print(f"  QA pairs     : {len(fb_qa) + len(fq_qa)}")
print(f"  Model        : {config.model_name} (4-bit)")
print(f"  Logs written : ./logs/")
print(f"  FAISS index  : {config.vector_store_path}")
print("\nRun colab_02_modules_demo.py next.")
