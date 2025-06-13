#!/usr/bin/env python3
"""
Clarity RAG – offline retrieval + Ollama generation + offline precision/recall
──────────────────────────────────────────────────────────────────────────────
Interactive chat        : python clarity_rag.py
Offline evaluation      : python clarity_rag.py --eval qa_testset.json
"""

import argparse, json, re, textwrap, numpy as np, requests
from collections import Counter
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────── USER CONFIG ────────────────────────────────────────────
DOC_PATH   = "/Users/aabhabothera/Downloads/synthetic_confluence_docs.json"
EMB_PATH   = "/Users/aabhabothera/Downloads/all-MiniLM-L6-v2-local"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "codellama"          # or "llama3", etc.

TOP_K     = 3                       # retrieved chunks per query
MAX_CHARS = 700                     # truncate each chunk in prompt
# ────────────────────────────────────────────────────────────────────────


def load_docs(path):
    docs = json.load(open(path))
    flat = [d["title"] + "\n" + "\n".join(d["content"]) for d in docs]
    return docs, flat


print("  Loading embedding model …")
embedder = SentenceTransformer(EMB_PATH)

print(" Loading documents …")
DOCS, FLAT = load_docs(DOC_PATH)
EMB = embedder.encode(FLAT, normalize_embeddings=True)
print(f"  {len(DOCS)} docs embedded")


# ───────────── Retrieval & Generation ──────────────────────────────────
def retrieve(question, k=TOP_K):
    q_vec = embedder.encode([question], normalize_embeddings=True)
    sims  = cosine_similarity(q_vec, EMB)[0]
    idxs  = sims.argsort()[::-1][:k]
    return [FLAT[i] for i in idxs]

def prompt_from(chunks, question):
    ctx = "\n\n".join(
        textwrap.shorten(c, width=MAX_CHARS, placeholder=" …") for c in chunks
    )
    return f"""You are Clarity, an internal Confluence assistant.

Context:
{ctx}

Question:
{question}

Answer concisely. Cite chunk numbers in [brackets] where helpful.
"""

def ollama_call(prompt):
    r = requests.post(
        OLLAMA_URL,
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
        timeout=300,
    )
    r.raise_for_status()
    return r.json()["response"]

def get_answer(question):
    chunks = retrieve(question)
    return ollama_call(prompt_from(chunks, question))


# ───────────── Offline Evaluation (precision/recall) ───────────────────
TOK_RE = re.compile(r"\b\w+\b", re.UNICODE)
def tokens(text): return Counter(TOK_RE.findall(text.lower()))

def prec_rec(ans_tok, ctx_tok):
    overlap = ans_tok & ctx_tok
    o       = sum(overlap.values())
    prec    = o / (sum(ans_tok.values()) or 1)
    rec     = o / (sum(ctx_tok.values()) or 1)
    return prec, rec

def run_eval(file):
    tests         = json.load(open(file))
    total_prec    = total_rec = 0.0

    for t in tests:
        q    = t["question"]
        gt   = t["answer"] if isinstance(t["answer"], str) else t["answer"][0]
        ctxs = retrieve(q)
        ans  = get_answer(q)

        p, r = prec_rec(tokens(ans), tokens(" ".join(ctxs)))
        total_prec += p
        total_rec  += r

        print(f"\nQ: {q}\nRef: {gt}\nAns: {ans[:110]}…")
        print(f"precision={p:.3f}  recall={r:.3f}")

    n = len(tests)
    print("\n  Offline context metrics (avg)")
    print(f"context_precision : {total_prec/n:.3f}")
    print(f"context_recall    : {total_rec/n:.3f}")


# ───────────── CLI  ────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", help="QA JSON file for offline evaluation")
    args = ap.parse_args()

    if args.eval:
        run_eval(args.eval)
    else:
        print("  Interactive mode – Ctrl-C or empty line to quit.")
        while True:
            try:
                q = input("\n  Your question: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nbye")
                break
            if not q: break
            print("\n  Clarity:\n", get_answer(q))

