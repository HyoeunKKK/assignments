"""Shared configuration for the Spoken QA cascade.

All heavy scripts import this so paths / HF cache / split definitions live in one place.
"""
import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = "/raid/slpr/hekim/project02"
DATA_ROOT = os.path.join(
    PROJECT_ROOT, "SLP_project02_data", "SLP_project02_data"
)
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

# Keep every Hugging Face download on /raid (root fs only has ~14 GB free).
HF_HOME = os.path.join(PROJECT_ROOT, "hf_cache")
os.environ.setdefault("HF_HOME", HF_HOME)

for _d in (CACHE_DIR, OUTPUT_DIR, HF_HOME):
    os.makedirs(_d, exist_ok=True)

# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------
# dev has gold labels; release is the deliverable (no labels).
SPLITS = {
    "dev": {
        "dir": os.path.join(DATA_ROOT, "release_dev"),
        "gold": os.path.join(DATA_ROOT, "release_dev", "gold.jsonl"),
    },
    "release": {
        "dir": os.path.join(DATA_ROOT, "release"),
        "gold": None,
    },
}


def split_dir(split, kind):
    """kind in {'questions','documents'}."""
    return os.path.join(SPLITS[split]["dir"], kind)


# ---------------------------------------------------------------------------
# Models (total ~9.51B <= 10B budget)
# ---------------------------------------------------------------------------
ASR_MODEL = "large-v3"                      # faster-whisper, 1.55B
EMBED_MODEL = "BAAI/bge-large-en-v1.5"      # dense retriever, 0.34B
LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"      # 7.62B

# bge-* query instruction (documents are embedded without a prefix).
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# ---------------------------------------------------------------------------
# Retrieval / RAG hyper-parameters
# ---------------------------------------------------------------------------
# Hybrid = weighted sum of min-max-normalized dense + BM25 scores.
# Tuned on dev: dense=0.8 gives best Recall@4 (0.90) / Recall@5 (0.91).
DENSE_WEIGHT = 0.8
RETRIEVE_TOPN = 10  # candidates kept in retrieval output
RAG_TOPK = 4        # docs actually placed in the LLM prompt (== document_ids)
