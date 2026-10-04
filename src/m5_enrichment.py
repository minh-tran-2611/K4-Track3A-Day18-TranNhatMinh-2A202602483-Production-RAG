from __future__ import annotations

"""
Module 5: Enrichment Pipeline
==============================
Làm giàu chunks TRƯỚC khi embed: Summarize, HyQA, Contextual Prepend, Auto Metadata.

Test: pytest tests/test_m5.py
"""

import json, os, re, sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import LLM_API_KEY, LLM_MODEL, LLM_EXTRA_PARAMS, get_llm_client


@dataclass
class EnrichedChunk:
    """Chunk đã được làm giàu."""
    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str  # "contextual", "summary", "hyqa", "full"


# ─── LLM helper ──────────────────────────────────────────


def _chat(system: str, user: str, max_tokens: int, json_mode: bool = False) -> str:
    """Gọi chat completion (OpenAI hoặc Gemini, temperature 0). Raise nếu lỗi — caller tự fallback."""
    client = get_llm_client()
    kwargs = dict(LLM_EXTRA_PARAMS)
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens=max_tokens,
        temperature=0,
        **kwargs,
    )
    return resp.choices[0].message.content.strip()


def _doc_title(source: str) -> str:
    """'nghi_phep_nam_v2024.md' → 'nghi phep nam v2024' (dùng cho fallback không cần API)."""
    return os.path.splitext(os.path.basename(source))[0].replace("_", " ").strip()


# ─── Technique 1: Chunk Summarization ────────────────────


def summarize_chunk(text: str) -> str:
    """
    Tạo summary ngắn cho chunk.
    Embed summary thay vì (hoặc cùng với) raw chunk → giảm noise.
    """
    if LLM_API_KEY:
        try:
            summary = _chat("Tóm tắt đoạn văn sau trong tối đa 2-3 câu ngắn gọn bằng tiếng Việt, "
                            "giữ nguyên con số. Bản tóm tắt phải ngắn hơn đoạn gốc.", text, 150)
            # Summary dài hơn bản gốc là vô nghĩa (chunk quá ngắn) → dùng extractive.
            if len(summary) <= len(text):
                return summary
            return _extractive_summary(text)
        except Exception as e:
            print(f"  ⚠️  LLM summarize failed: {e}")

    return _extractive_summary(text)


def _extractive_summary(text: str) -> str:
    """Fallback không cần API: 2 câu đầu."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.replace("\n", " ")) if s.strip()]
    return " ".join(sentences[:2]) if sentences else text


# ─── Technique 2: Hypothesis Question-Answer (HyQA) ─────


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    """
    Generate câu hỏi mà chunk có thể trả lời.
    Index cả questions lẫn chunk → query match tốt hơn (bridge vocabulary gap).
    """
    if LLM_API_KEY:
        try:
            content = _chat(f"Dựa trên đoạn văn, tạo {n_questions} câu hỏi tiếng Việt mà đoạn văn có thể "
                            "trả lời. Trả về mỗi câu hỏi trên 1 dòng, không đánh số.", text, 200)
            questions = [q.strip().lstrip("0123456789.-) ") for q in content.split("\n") if q.strip()]
            return questions[:n_questions]
        except Exception as e:
            print(f"  ⚠️  LLM HyQA failed: {e}")

    return _extractive_questions(text, n_questions)


def _extractive_questions(text: str, n_questions: int = 3) -> list[str]:
    """Fallback không cần API: biến câu khẳng định thành câu hỏi."""
    sentences = [s.strip() for s in re.split(r"[.!?\n]", text) if len(s.strip()) > 10]
    return [f"{s.rstrip('.')}?" for s in sentences[:n_questions]]


# ─── Technique 3: Contextual Prepend (Anthropic style) ──


def contextual_prepend(text: str, document_title: str = "") -> str:
    """
    Prepend context giải thích chunk nằm ở đâu trong document.
    Anthropic benchmark: giảm 49% retrieval failure (alone).
    """
    if LLM_API_KEY:
        try:
            context = _chat("Viết 1 câu ngắn mô tả đoạn văn này nằm ở đâu trong tài liệu và nói về chủ đề gì. "
                            "Chỉ trả về 1 câu.",
                            f"Tài liệu: {document_title}\n\nĐoạn văn:\n{text}", 80)
            return f"{context}\n\n{text}"
        except Exception as e:
            print(f"  ⚠️  LLM contextual failed: {e}")

    prefix = f"Trích từ {document_title}. " if document_title else ""
    return f"{prefix}{text}"


# ─── Technique 4: Auto Metadata Extraction ──────────────


def extract_metadata(text: str) -> dict:
    """
    LLM extract metadata tự động: topic, entities, date_range, category.
    """
    if LLM_API_KEY:
        try:
            content = _chat('Trích xuất metadata từ đoạn văn. Trả về JSON: {"topic": "...", "entities": ["..."], '
                            '"category": "policy|hr|it|finance", "language": "vi|en"}', text, 150, json_mode=True)
            return json.loads(content)
        except Exception as e:
            print(f"  ⚠️  LLM metadata failed: {e}")

    return {"topic": "general", "entities": [], "category": "policy", "language": "vi"}


# ─── Combined Single-Call Mode ───────────────────────────

_COMBINED_PROMPT = """Bạn chuẩn bị dữ liệu cho hệ thống tìm kiếm tài liệu nội bộ công ty.
Phân tích đoạn văn và trả về JSON:
{
  "summary": "tóm tắt 2-3 câu, giữ nguyên các con số quan trọng",
  "questions": ["câu hỏi 1", "câu hỏi 2", "câu hỏi 3"],
  "context": "1 câu mô tả đoạn văn thuộc tài liệu/chính sách nào (kèm phiên bản/năm nếu có) và nói về chủ đề gì",
  "metadata": {"topic": "...", "entities": ["..."], "category": "policy|hr|it|finance", "language": "vi|en"}
}
Chỉ dùng thông tin có trong đoạn văn và tên tài liệu. Viết bằng tiếng Việt."""


def _enrich_single_call(text: str, source: str) -> dict:
    """Single LLM call to get summary + questions + context + metadata.

    ⚠️ Cost optimization: 1 API call thay vì 4 calls riêng lẻ.
    """
    if LLM_API_KEY:
        try:
            content = _chat(_COMBINED_PROMPT, f"Tài liệu: {source}\n\nĐoạn văn:\n{text}", 400, json_mode=True)
            return json.loads(content)
        except Exception as e:
            print(f"  ⚠️  Enrichment API failed: {e}")

    # Fallback không cần API: context từ tên file + extractive summary/questions.
    title = _doc_title(source)
    return {
        "summary": _extractive_summary(text),
        "questions": _extractive_questions(text),
        "context": f"Trích từ tài liệu: {title}." if title else "",
        "metadata": {},
    }


# ─── Full Enrichment Pipeline ────────────────────────────


def enrich_chunks(
    chunks: list[dict],
    methods: list[str] | None = None,
) -> list[EnrichedChunk]:
    """
    Chạy enrichment pipeline trên danh sách chunks. (Đã implement sẵn — dùng functions ở trên)

    Có 2 chế độ:
    - methods cụ thể (["summary"], ["contextual"]...): gọi từng function riêng (tốt cho học/debug)
    - methods=["combined"] hoặc None: 1 API call duy nhất cho tất cả (tốt cho production)

    Args:
        chunks: List of {"text": str, "metadata": dict}
        methods: Default None → combined mode (1 call/chunk).
                 Options: "summary", "hyqa", "contextual", "metadata", "combined"
    """
    if methods is None:
        methods = ["combined"]

    use_combined = "combined" in methods

    enriched = []
    for i, chunk in enumerate(chunks):
        text = chunk["text"]
        source = chunk.get("metadata", {}).get("source", "")

        if use_combined:
            result = _enrich_single_call(text, source)
            summary = result.get("summary", "")
            questions = result.get("questions", [])
            context_line = result.get("context", "")
            enriched_text = f"{context_line}\n\n{text}" if context_line else text
            auto_meta = result.get("metadata", {})
        else:
            summary = summarize_chunk(text) if "summary" in methods else ""
            questions = generate_hypothesis_questions(text) if "hyqa" in methods else []
            enriched_text = contextual_prepend(text, source) if "contextual" in methods else text
            auto_meta = extract_metadata(text) if "metadata" in methods else {}

        enriched.append(EnrichedChunk(
            original_text=text,
            enriched_text=enriched_text,
            summary=summary,
            hypothesis_questions=questions,
            auto_metadata={**chunk.get("metadata", {}), **auto_meta},
            method="+".join(methods),
        ))

        if (i + 1) % 10 == 0 or (i + 1) == len(chunks):
            print(f"  Enriched {i + 1}/{len(chunks)} chunks...", flush=True)

    return enriched


# ─── Main ────────────────────────────────────────────────

if __name__ == "__main__":
    sample = "Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm. Số ngày nghỉ phép tăng thêm 1 ngày cho mỗi 5 năm thâm niên công tác."

    print("=== Enrichment Pipeline Demo ===\n")
    print(f"Original: {sample}\n")

    s = summarize_chunk(sample)
    print(f"Summary: {s}\n")

    qs = generate_hypothesis_questions(sample)
    print(f"HyQA questions: {qs}\n")

    ctx = contextual_prepend(sample, "Sổ tay nhân viên VinUni 2024")
    print(f"Contextual: {ctx}\n")

    meta = extract_metadata(sample)
    print(f"Auto metadata: {meta}")
