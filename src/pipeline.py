from __future__ import annotations

"""Production RAG Pipeline — Ghép toàn bộ M1+M2+M3+M4+M5."""

import os, sys, time
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.m1_chunking import load_documents, chunk_hierarchical
from src.m2_search import HybridSearch
from src.m3_rerank import CrossEncoderReranker
from src.m4_eval import load_test_set, evaluate_ragas, failure_analysis, save_report
from src.m5_enrichment import enrich_chunks
from config import RERANK_TOP_K


def build_pipeline():
    """Build production RAG pipeline."""
    print("=" * 60)
    print("PRODUCTION RAG PIPELINE")
    print("=" * 60, flush=True)

    # Step 1: Load & Chunk (M1)
    t0 = time.time()
    print("\n[1/4] Chunking documents...", flush=True)
    docs = load_documents()
    all_chunks = []
    PARENT_STORE.clear()
    for doc in docs:
        parents, children = chunk_hierarchical(doc["text"], metadata=doc["metadata"])
        source = doc["metadata"]["source"]
        for p in parents:
            # parent_id chỉ unique trong 1 document → key theo (source, parent_id)
            PARENT_STORE[f"{source}::{p.metadata['parent_id']}"] = p.text
        for child in children:
            all_chunks.append({"text": child.text, "metadata": {**child.metadata, "parent_id": child.parent_id,
                                                                 "parent_key": f"{source}::{child.parent_id}"}})
    print(f"  ✓ {len(all_chunks)} chunks from {len(docs)} documents ({time.time()-t0:.1f}s)", flush=True)
    LATENCY["build"]["chunking_s"] = round(time.time() - t0, 2)

    # Step 2: Enrichment (M5)
    t0 = time.time()
    print(f"\n[2/4] Enriching {len(all_chunks)} chunks (M5, 1 API call/chunk)...", flush=True)
    enriched = enrich_chunks(all_chunks)
    if enriched:
        all_chunks = [{"text": e.enriched_text, "metadata": e.auto_metadata} for e in enriched]
        print(f"  ✓ Enriched {len(enriched)} chunks ({time.time()-t0:.1f}s)", flush=True)
    else:
        print("  ⚠️  M5 not implemented — using raw chunks", flush=True)
    LATENCY["build"]["enrichment_s"] = round(time.time() - t0, 2)

    # Step 3: Index (M2)
    t0 = time.time()
    print(f"\n[3/4] Indexing {len(all_chunks)} chunks (BM25 + Dense)...", flush=True)
    search = HybridSearch()
    search.index(all_chunks)
    print(f"  ✓ Indexed ({time.time()-t0:.1f}s)", flush=True)
    LATENCY["build"]["indexing_s"] = round(time.time() - t0, 2)

    # Step 4: Reranker (M3)
    t0 = time.time()
    print("\n[4/4] Loading reranker...", flush=True)
    reranker = CrossEncoderReranker()
    reranker._load_model()  # load trước để latency query không tính thời gian load model
    print(f"  ✓ Reranker ready ({time.time()-t0:.1f}s)", flush=True)
    LATENCY["build"]["reranker_load_s"] = round(time.time() - t0, 2)

    return search, reranker


def run_query(query: str, search: HybridSearch, reranker: CrossEncoderReranker) -> tuple[str, list[str]]:
    """Run single query through pipeline."""
    timings = {}
    t0 = time.perf_counter()
    results = search.search(query)
    timings["search_ms"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    docs = [{"text": r.text, "score": r.score, "metadata": r.metadata} for r in results]
    reranked = reranker.rerank(query, docs, top_k=RERANK_TOP_K)
    timings["rerank_ms"] = (time.perf_counter() - t0) * 1000
    hits = reranked if reranked else results[:RERANK_TOP_K]

    # Retrieve child (precision) → return parent (context), bỏ trùng parent.
    contexts, seen = [], set()
    for h in hits:
        key = h.metadata.get("parent_key")
        if key in seen:
            continue
        seen.add(key)
        contexts.append(PARENT_STORE.get(key, h.text))

    t0 = time.perf_counter()
    from config import LLM_API_KEY
    if LLM_API_KEY and contexts:
        try:
            from src.llm import chat
            context_str = "\n\n---\n\n".join(contexts)
            answer = chat([
                {"role": "system", "content": ANSWER_PROMPT},
                {"role": "user", "content": f"Context:\n{context_str}\n\nCâu hỏi: {query}"},
            ])
        except Exception as e:
            print(f"  ⚠️  LLM generation failed: {e}", flush=True)
            answer = contexts[0]
    else:
        answer = contexts[0] if contexts else "Không tìm thấy thông tin."
    timings["llm_ms"] = (time.perf_counter() - t0) * 1000
    LATENCY["queries"].append(timings)
    return answer, contexts


ANSWER_PROMPT = (
    "Bạn là trợ lý tra cứu chính sách nội bộ. Trả lời CHỈ dựa trên context, ngắn gọn, giữ nguyên con số và đơn vị. "
    "Nếu context có nhiều phiên bản của cùng một chính sách, trả lời theo phiên bản hiện hành (mới nhất) "
    "và nói rõ phiên bản cũ đã bị thay thế. Nếu context không có thông tin → nói 'Không tìm thấy.'"
)
PARENT_STORE: dict[str, str] = {}
LATENCY: dict = {"build": {}, "queries": []}


def latency_report() -> dict:
    """Tổng hợp latency từng bước (build + trung bình mỗi query)."""
    q = LATENCY["queries"]
    per_query = {}
    if q:
        for k in q[0]:
            vals = [t[k] for t in q]
            per_query[k] = {"avg": round(sum(vals) / len(vals), 1), "max": round(max(vals), 1)}
    return {"build": LATENCY["build"], "per_query_ms": per_query, "num_queries": len(q)}


def evaluate_pipeline(search: HybridSearch, reranker: CrossEncoderReranker):
    """Run evaluation on test set."""
    test_set = load_test_set()
    print(f"\n[Eval] Running {len(test_set)} queries...", flush=True)
    questions, answers, all_contexts, ground_truths = [], [], [], []

    for i, item in enumerate(test_set):
        answer, contexts = run_query(item["question"], search, reranker)
        questions.append(item["question"])
        answers.append(answer)
        all_contexts.append(contexts)
        ground_truths.append(item["ground_truth"])
        print(f"  [{i+1}/{len(test_set)}] {item['question'][:50]}...", flush=True)

    t0 = time.time()
    print(f"\n[Eval] Running RAGAS (4 metrics × {len(test_set)} questions)...", flush=True)
    results = evaluate_ragas(questions, answers, all_contexts, ground_truths)
    print(f"  ✓ RAGAS done ({time.time()-t0:.1f}s)", flush=True)

    print("\n" + "=" * 60)
    print("PRODUCTION RAG SCORES")
    print("=" * 60)
    for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        s = results.get(m, 0)
        print(f"  {'✓' if s >= 0.75 else '✗'} {m}: {s:.4f}")

    failures = failure_analysis(results.get("per_question", []))
    save_report(results, failures)
    _append_extras(results)

    lat = latency_report()
    print("\nLATENCY BREAKDOWN")
    for k, v in lat["build"].items():
        print(f"  build.{k:<18} {v:>8.2f} s")
    for k, v in lat["per_query_ms"].items():
        print(f"  query.{k:<18} avg {v['avg']:>8.1f} ms | max {v['max']:>8.1f} ms")
    return results


def _append_extras(results: dict, path: str = "reports/ragas_report.json"):
    """Bổ sung latency breakdown + per-question scores vào report (save_report chỉ lưu aggregate + failures)."""
    import json
    with open(path, encoding="utf-8") as f:
        report = json.load(f)
    report["latency"] = latency_report()
    report["per_question"] = [{
        "question": r.question, "answer": r.answer,
        "ground_truth": r.ground_truth, "contexts": r.contexts,
        "faithfulness": r.faithfulness, "answer_relevancy": r.answer_relevancy,
        "context_precision": r.context_precision, "context_recall": r.context_recall,
    } for r in results.get("per_question", [])]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    start = time.time()
    search, reranker = build_pipeline()
    evaluate_pipeline(search, reranker)
    print(f"\nTotal: {time.time() - start:.1f}s")
