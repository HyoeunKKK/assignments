"""Stage 3: RAG answer generation with Qwen2.5-7B-Instruct.

Usage:
    python generate.py --split dev --topk 4
    python generate.py --split release --topk 4 --out predictions.jsonl

Reads cached transcripts + retrieval results, builds a prompt with the
top-k document transcripts, generates a short answer, and writes a
predictions JSONL: {"question_id","answer","document_ids"}.
document_ids == the top-k docs actually placed in the prompt.
"""
import argparse
import json
import os
import re

import config
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM_PROMPT = (
    "You are a precise question-answering system. You are given a spoken "
    "question (auto-transcribed) and a few candidate documents (also "
    "auto-transcribed, so they may contain recognition errors). Using only "
    "the documents, answer the question with the SHORTEST possible answer "
    "span: just the entity, name, number, or phrase. Rules:\n"
    "- Output the answer only. No explanation, no full sentence, no quotes.\n"
    "- Use lowercase.\n"
    "- If multiple documents conflict, pick the most directly relevant one.\n"
    "- If the answer is not in the documents, give your single best guess "
    "from them anyway."
)

# One in-context example to lock the output format.
FEWSHOT = [
    (
        "Documents:\n"
        "[1] The Eiffel Tower is a wrought-iron lattice tower on the Champ de "
        "Mars in Paris, France. It was completed in 1889.\n\n"
        "Question: in which city is the eiffel tower located\n\n"
        "Answer:",
        "paris",
    ),
]


def load_json(path):
    with open(path) as f:
        return json.load(f)


def build_user_prompt(question, docs):
    lines = ["Documents:"]
    for i, text in enumerate(docs, 1):
        lines.append(f"[{i}] {text}")
    lines.append("")
    lines.append(f"Question: {question}")
    lines.append("")
    lines.append("Answer:")
    return "\n".join(lines)


def chat(model, tok, messages, max_new_tokens):
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=max_new_tokens,
                             do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(gen[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)


def postprocess(ans):
    ans = ans.strip().splitlines()[0] if ans.strip() else ""
    ans = ans.strip().strip('"').strip("'").strip()
    # drop hedging / echoed prefixes the model sometimes adds
    ans = re.sub(r"^\s*(answer|my best guess|best guess|guess)\s*[:\-]\s*",
                 "", ans, flags=re.I)
    ans = ans.strip().strip('"').strip("'")
    ans = ans.rstrip(".").strip()
    return ans.lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=list(config.SPLITS), required=True)
    ap.add_argument("--topk", type=int, default=config.RAG_TOPK)
    ap.add_argument("--out", default=None,
                    help="output filename under outputs/ (default predictions_{split}.jsonl)")
    ap.add_argument("--limit", type=int, default=0, help="debug: only first N questions")
    args = ap.parse_args()

    questions = load_json(os.path.join(config.CACHE_DIR, f"{args.split}_questions.json"))
    documents = load_json(os.path.join(config.CACHE_DIR, f"{args.split}_documents.json"))
    retrieval = load_json(os.path.join(config.OUTPUT_DIR, f"{args.split}_retrieval.json"))

    print(f"Loading {config.LLM_MODEL} ...")
    tok = AutoTokenizer.from_pretrained(config.LLM_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        config.LLM_MODEL, torch_dtype=torch.float16, device_map="auto",
    )
    model.eval()

    out_name = args.out or f"predictions_{args.split}.jsonl"
    out_path = os.path.join(config.OUTPUT_DIR, out_name)

    q_ids = sorted(questions)
    if args.limit:
        q_ids = q_ids[: args.limit]

    with open(out_path, "w") as fout:
        for n, qid in enumerate(q_ids, 1):
            ranked = retrieval[qid]
            doc_ids = [d for d, _ in ranked[: args.topk]]
            docs = [documents[d] for d in doc_ids]

            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            for fs_user, fs_ans in FEWSHOT:
                messages.append({"role": "user", "content": fs_user})
                messages.append({"role": "assistant", "content": fs_ans})
            messages.append({"role": "user",
                             "content": build_user_prompt(questions[qid], docs)})
            answer = postprocess(chat(model, tok, messages, max_new_tokens=32))

            fout.write(json.dumps(
                {"question_id": qid, "answer": answer, "document_ids": doc_ids},
                ensure_ascii=False) + "\n")
            fout.flush()
            if n % 20 == 0 or n == len(q_ids):
                print(f"  [{n}/{len(q_ids)}] {qid} -> {answer!r} (docs {doc_ids})")

    print(f"wrote {out_path} ({len(q_ids)} predictions)")


if __name__ == "__main__":
    main()
