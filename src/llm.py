from __future__ import annotations

"""
Shared LLM helper — dùng chung cho M5 enrichment, answer generation và RAGAS judge.

- Disk cache theo nội dung prompt → chạy lại pipeline không tốn thêm request.
- Xoay vòng model khi gặp 429 hết quota ngày (Gemini free tier giới hạn request/ngày/model).
- Tự hạ reasoning_effort "none" → "low" cho model không hỗ trợ tắt thinking.
"""

import hashlib, json, os, sys, threading, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import LLM_API_KEY, LLM_MODELS, LLM_REASONING_EFFORT, get_llm_client

CACHE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".cache", "llm_cache.json")

_lock = threading.Lock()
_cache: dict | None = None
_exhausted: set[str] = set()
_effort: dict[str, str | None] = {}  # model → reasoning_effort đang dùng


class LLMUnavailable(RuntimeError):
    """Không còn model nào gọi được (hết quota hoặc chưa có API key)."""


def _load_cache() -> dict:
    global _cache
    if _cache is None:
        try:
            with open(CACHE_PATH, encoding="utf-8") as f:
                _cache = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            _cache = {}
    return _cache


def _save_cache() -> None:
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    tmp = CACHE_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(_cache, f, ensure_ascii=False)
    os.replace(tmp, CACHE_PATH)


def chat(messages: list[dict], max_tokens: int | None = None, json_mode: bool = False,
         temperature: float = 0.0) -> str:
    """Chat completion có cache + xoay vòng model. Raise LLMUnavailable nếu không gọi được."""
    if not LLM_API_KEY:
        raise LLMUnavailable("Chưa set OPENAI_API_KEY / GEMINI_API_KEY")

    key = hashlib.sha256(json.dumps([messages, max_tokens, json_mode, round(temperature, 3)],
                                    ensure_ascii=False).encode()).hexdigest()
    with _lock:
        cached = _load_cache().get(key)
    if cached is not None:
        return cached

    client = get_llm_client().with_options(max_retries=0, timeout=120)
    for model in LLM_MODELS:
        if model in _exhausted:
            continue
        for attempt in range(6):
            effort = _effort.get(model, LLM_REASONING_EFFORT)
            kwargs = {}
            if effort:
                kwargs["reasoning_effort"] = effort
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            # Khi model vẫn "thinking", thinking tokens tính vào max_tokens → bỏ giới hạn để không bị cắt.
            if max_tokens and effort in (None, "none"):
                kwargs["max_tokens"] = max_tokens
            try:
                resp = client.chat.completions.create(model=model, messages=messages,
                                                      temperature=temperature, **kwargs)
                text = (resp.choices[0].message.content or "").strip()
                with _lock:
                    _load_cache()[key] = text
                    _save_cache()
                return text
            except Exception as e:  # noqa: BLE001 — phân loại lỗi theo status/message
                msg = str(e)
                status = getattr(e, "status_code", None)
                if status == 429 and ("PerDay" in msg or "per day" in msg.lower()):
                    print(f"  ⚠️  {model}: hết quota ngày → chuyển model khác", flush=True)
                    _exhausted.add(model)
                    break
                if status == 400 and effort == "none":
                    _effort[model] = "low"  # model không cho tắt thinking
                    continue
                if status == 404:
                    _exhausted.add(model)
                    break
                if status in (429, 500, 502, 503, 504) or "timeout" in msg.lower():
                    time.sleep(min(60, 5 * 2 ** attempt))  # rate limit theo phút / lỗi tạm thời
                    continue
                raise
    raise LLMUnavailable(f"Tất cả model đều không khả dụng: {LLM_MODELS}")


def make_ragas_llm():
    """RAGAS LLM judge dùng chung chat() (cache + xoay vòng model, n generations gọi tuần tự)."""
    import asyncio
    from langchain_core.outputs import Generation, LLMResult
    from ragas.llms.base import BaseRagasLLM

    class RotatingRagasLLM(BaseRagasLLM):
        def generate_text(self, prompt, n=1, temperature=1e-8, stop=None, callbacks=None) -> LLMResult:
            text = prompt.to_string()
            gens = []
            for i in range(n):
                # i > 0: thêm marker để mỗi generation có cache key riêng (RAGAS cần n câu khác nhau)
                content = text if i == 0 else f"{text}\n\n(Biến thể #{i + 1})"
                gens.append(Generation(text=chat([{"role": "user", "content": content}],
                                                 temperature=temperature or 0.0)))
            return LLMResult(generations=[gens])

        async def agenerate_text(self, prompt, n=1, temperature=None, stop=None, callbacks=None) -> LLMResult:
            return await asyncio.to_thread(self.generate_text, prompt, n, temperature or 1e-8, stop, callbacks)

    return RotatingRagasLLM()
