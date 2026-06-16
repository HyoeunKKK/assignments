"""Semantic-equivalence evaluation using Qwen2.5-7B as an LLM judge.

This mirrors the project's grading rule ("semantically equivalent answers will
be considered correct") more closely than the string-match proxy in
evaluate.py. The judge is only an evaluation aid (the QA *system* still uses
open-weight models only).

Usage:
    python judge.py --predictions predictions_dev_orig_k4.jsonl --split dev
"""
import argparse
import json
import os
import re

import config
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

JUDGE_SYSTEM = (
    "You are a strict grading assistant for a question-answering benchmark. "
    "You are given a question, the gold (reference) answer(s), and a model's "
    "predicted answer. Decide whether the prediction is CORRECT, i.e. "
    "semantically equivalent to any gold answer for this question.\n"
    "- Accept synonyms, paraphrases, abbreviations/expansions (e.g. 'USA' = "
    "'United States'), different surface forms of the same date/number, and "
    "answers that contain the gold answer plus harmless extra words.\n"
    "- Reject answers that name a different entity, are factually different, "
    "or are empty/refusals.\n"
    "Respond with exactly one word: 'correct' or 'incorrect'."
)

JUDGE_SYSTEM_LENIENT = (
    "You are a lenient grading assistant for a spoken question-answering "
    "benchmark. You are given a question, the gold (reference) answer(s), and "
    "a model's predicted answer. Mark the prediction CORRECT if it conveys the "
    "same information as ANY gold answer. Be generous:\n"
    "- Accept synonyms, paraphrases, abbreviations/expansions (e.g. 'CRTC' = "
    "'Canadian Radio-television Commission'), and any surface form of the same "
    "date, number, currency, or name.\n"
    "- Accept if the prediction CONTAINS the gold answer, even with extra words "
    "(e.g. gold 'warblers', prediction 'wood and plain-leaf warblers').\n"
    "- Accept if the prediction is a more specific or more general form of the "
    "gold answer referring to the same thing (e.g. 'hong kong island' vs "
    "'hong kong'), or a historically equivalent name (e.g. 'siam' vs "
    "'thailand'), or has a minor spelling/transcription difference.\n"
    "- Only mark INCORRECT when the prediction clearly refers to a DIFFERENT "
    "entity/fact, or is empty.\n"
    "Respond with exactly one word: 'correct' or 'incorrect'."
)


def load_gold(split):
    g = {}
    for l in open(config.SPLITS[split]["gold"]):
        o = json.loads(l)
        a = o["answer"]
        a = [a] if isinstance(a, str) else a
        g[o["question_id"]] = a
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--lenient", action="store_true",
                    help="use the lenient (generous) grading rubric")
    args = ap.parse_args()
    system_prompt = JUDGE_SYSTEM_LENIENT if args.lenient else JUDGE_SYSTEM
    tag = "lenient" if args.lenient else "strict"

    gold = load_gold(args.split)
    questions = json.load(open(os.path.join(config.CACHE_DIR, f"{args.split}_questions.json")))
    rows = [json.loads(l) for l in open(os.path.join(config.OUTPUT_DIR, args.predictions))]

    print(f"Loading judge {config.LLM_MODEL} ...")
    tok = AutoTokenizer.from_pretrained(config.LLM_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        config.LLM_MODEL, dtype=torch.float16, device_map="auto"
    )
    model.eval()

    correct = 0
    n = 0
    disagree = []
    for i, r in enumerate(rows, 1):
        q = r["question_id"]
        if q not in gold:
            continue
        n += 1
        golds = gold[q]
        user = (
            f"Question: {questions.get(q, '')}\n"
            f"Gold answer(s): {' | '.join(golds)}\n"
            f"Predicted answer: {r['answer']}\n\n"
            "Is the predicted answer correct? Reply 'correct' or 'incorrect'."
        )
        msgs = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": user}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=4, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        out = tok.decode(gen[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        verdict = "correct" in out.lower() and "incorrect" not in out.lower()
        correct += verdict
        r["_judge"] = "correct" if verdict else "incorrect"
        if i % 50 == 0:
            print(f"  [{i}/{len(rows)}] running acc {correct/n:.3f}")

    print(f"\nLLM-judge ({tag}) semantic accuracy: {correct}/{n} = {correct/n:.3f}")
    print(f"Projected answer points (x20): {correct/n*20:.2f} / 20")
    dump = os.path.join(config.OUTPUT_DIR, f"{args.split}_judged_{tag}.jsonl")
    with open(dump, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {dump}")


if __name__ == "__main__":
    main()
