# Project 2: Spoken Question Answering (ASR → Retriever → LLM)

A three-stage cascade that answers spoken questions over a corpus of spoken
documents. Each question is transcribed, relevant documents are retrieved from
the transcribed corpus, and a text LLM produces a short answer from the
retrieved context.

## Pipeline

```
wav ──ASR──► transcript ──Retriever(BM25+dense)──► top-k docs ──LLM(RAG)──► short answer
```

| Stage | Model | Params |
|-------|-------|--------|
| ASR | faster-whisper **large-v3** (CTranslate2, `int8_float16`) | 1.55 B |
| Retriever | **BM25** + **BAAI/bge-large-en-v1.5**, min-max weighted fusion (dense=0.8) | 0.335 B |
| LLM | **Qwen/Qwen2.5-7B-Instruct** (fp16) | 7.616 B |
| **Total** | | **9.50 B ≤ 10 B** |

(BM25 is parameter-free.)

## Environment

Uses the existing conda env `trus`:

```bash
# python: /raid/slpr/hekim/miniconda3/envs/trus/bin/python
# torch 2.6+cu124, transformers 5.9, faster-whisper 1.2, accelerate, datasets
pip install rank-bm25 sentence-transformers   # the only extra deps
```

All Hugging Face downloads are kept on `/raid` via `HF_HOME`
(`config.py` sets it to `<PROJECT_ROOT>/hf_cache`, since `/` has little space).

## How to run

Paths and hyper-parameters live in `config.py`. Run from this `code/` dir.

```bash
export HF_HOME=/raid/slpr/hekim/project02/hf_cache
export CUDA_VISIBLE_DEVICES=0

# 1) ASR — transcribe questions + documents, cached to cache/{split}_*.json
python transcribe.py --split dev
python transcribe.py --split release

# 2) Retrieval — hybrid BM25+bge, writes outputs/{split}_retrieval.json
python retrieve.py --split dev
python retrieve.py --split release

# 3) RAG generation — Qwen2.5-7B, writes outputs/<out>.jsonl
python generate.py --split release --topk 4 --out predictions.jsonl

# (dev only) evaluation against gold labels
python evaluate.py --split dev --retrieval                       # Recall@k
python evaluate.py --split dev --predictions predictions_dev_orig_k4.jsonl  # accuracy
```

Every stage caches to disk and is safe to re-run (ASR skips already-cached ids).

## Design choices

- **Retrieval fusion.** Dense (bge) clearly beats BM25 on dev, but a weighted
  blend is best. Scores are min-max normalized per query and combined as
  `0.8·dense + 0.2·bm25`. Tuned on dev: Recall@1 0.62, **Recall@4 0.90**,
  Recall@5 0.91, Recall@10 0.94.
- **k = 4.** `document_ids` in the output are exactly the 4 documents placed in
  the LLM prompt (not the raw top-10). On dev, k=4 (0.643) edged out k=5
  (0.630); larger k adds distractors.
- **Prompt.** System prompt + one in-context example fix the output format:
  shortest answer span, lowercase, no explanation. A stricter "pick exactly one
  doc" prompt with 2 examples was tried and scored worse (0.60), so the simpler
  prompt is used. Greedy decoding, `max_new_tokens=32`. Output is
  post-processed (strip quotes / hedging prefixes / trailing period, lowercase).

## Output format

`outputs/predictions.jsonl` — 300 lines (q500–q799), one JSON object per line:

```json
{"question_id": "q500", "answer": "pakistan", "document_ids": ["d692", "d822", "d577", "d962"]}
```

## Files

```
config.py       paths, model ids, hyper-parameters
transcribe.py   stage 1: ASR + transcript cache
retrieve.py     stage 2: hybrid retrieval + fusion
generate.py     stage 3: RAG answer generation
evaluate.py     dev: Recall@k + answer accuracy + failure dump
```

## Dev results (proxy: normalized exact/containment match)

- Retrieval Recall@4 = 0.90
- Answer accuracy = 0.643 (193/300). Of 107 misses, only ~22 are retrieval
  misses (gold doc absent from prompt); the rest are LLM extraction errors,
  largely from auto-transcribed distractor documents and ASR errors on the
  answer span itself. The real grading uses semantic-equivalence judging, which
  is more lenient than this exact-match proxy.
```
