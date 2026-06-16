"""Stage 4: dev evaluation (gold labels available).

Reports retrieval Recall@k and answer accuracy (normalized-match proxy for
the semantic-equivalence grading), and dumps failure cases.

Usage:
    python evaluate.py --retrieval                       # recall@k of retrieval
    python evaluate.py --predictions predictions_dev.jsonl   # answer accuracy
"""
import argparse
import json
import os
import re
import string

import config


def load_gold(split="dev"):
    gold = {}
    with open(config.SPLITS[split]["gold"]) as f:
        for line in f:
            o = json.loads(line)
            ans = o["answer"]
            if isinstance(ans, str):
                ans = [ans]
            gold[o["question_id"]] = {"document_id": o["document_id"], "answer": ans}
    return gold


_ARTICLES = re.compile(r"\b(a|an|the)\b")


def normalize(s):
    s = s.lower()
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def answer_match(pred, golds):
    p = normalize(pred)
    if not p:
        return False
    for g in golds:
        g = normalize(g)
        if p == g or g in p or p in g:
            return True
    return False


def eval_retrieval(split):
    gold = load_gold(split)
    ret = json.load(open(os.path.join(config.OUTPUT_DIR, f"{split}_retrieval.json")))
    ks = [1, 3, 4, 5, 10]
    hits = {k: 0 for k in ks}
    n = 0
    for qid, g in gold.items():
        if qid not in ret:
            continue
        n += 1
        order = [d for d, _ in ret[qid]]
        for k in ks:
            if g["document_id"] in order[:k]:
                hits[k] += 1
    print(f"Retrieval recall over {n} questions:")
    for k in ks:
        print(f"  Recall@{k}: {hits[k]}/{n} = {hits[k]/n:.3f}")


def eval_predictions(split, pred_file):
    gold = load_gold(split)
    path = os.path.join(config.OUTPUT_DIR, pred_file)
    correct = 0
    n = 0
    fails = []
    with open(path) as f:
        for line in f:
            o = json.loads(line)
            qid = o["question_id"]
            if qid not in gold:
                continue
            n += 1
            ok = answer_match(o["answer"], gold[qid]["answer"])
            correct += ok
            if not ok:
                gold_in_prompt = gold[qid]["document_id"] in o.get("document_ids", [])
                fails.append({
                    "question_id": qid,
                    "pred": o["answer"],
                    "gold": gold[qid]["answer"],
                    "gold_doc": gold[qid]["document_id"],
                    "gold_doc_in_prompt": gold_in_prompt,
                    "document_ids": o.get("document_ids", []),
                })
    print(f"Answer accuracy: {correct}/{n} = {correct/n:.3f}")
    proj = correct / n * 20
    print(f"Projected answer points (x20): {proj:.2f} / 20")
    miss_doc = sum(1 for x in fails if not x["gold_doc_in_prompt"])
    print(f"Failures: {len(fails)}  (of which gold doc NOT in prompt: {miss_doc})")
    dump = os.path.join(config.OUTPUT_DIR, f"{split}_failures.json")
    json.dump(fails, open(dump, "w"), ensure_ascii=False, indent=2)
    print(f"wrote {dump}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--retrieval", action="store_true")
    ap.add_argument("--predictions", default=None)
    args = ap.parse_args()
    if args.retrieval:
        eval_retrieval(args.split)
    if args.predictions:
        eval_predictions(args.split, args.predictions)
    if not args.retrieval and not args.predictions:
        eval_retrieval(args.split)


if __name__ == "__main__":
    main()
