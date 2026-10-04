# Individual Reflection — Lab 18: Production RAG

**Họ và tên:** Trần Nhật Minh (2A202602483)  
**Khóa:** K4 - Track 3A  
**Ngày hoàn thành:** 04/10/2026

---

## Phần 1: Mapping bài giảng (Lecture Mapping)

| Lecture Concept | Module | Hàm cụ thể | Observation & Phân tích |
|----------------|--------|-------------|--------------------------|
| Semantic chunking | M1 | `chunk_semantic()` | Threshold 0.85 với `all-MiniLM-L6-v2` tạo **208 chunks** (avg 99 chars, min 6) vs basic **51 chunks** (avg 410). Model tiếng Anh cho similarity giữa các câu tiếng Việt thấp → gần như tách mỗi câu một chunk, chunk quá vụn, mất ngữ cảnh. Muốn dùng semantic chunking cho tiếng Việt cần encoder đa ngôn ngữ (bge-m3) và hạ threshold. |
| Hierarchical chunking | M1 | `chunk_hierarchical()` + `run_query()` | 11 parents / 120 children (avg 173, max 256 chars). Pipeline index **child** để match chính xác, sau rerank map `parent_key` → trả về **parent** (dedupe). Đây là lý do chính Context Recall tăng 0.825 → **0.925**: câu multi-hop/version (VD nghỉ phép v2023 vs v2024) cần cả đoạn chứ không chỉ 1 câu. |
| Structure-aware chunking | M1 | `chunk_structure_aware()` | 106 chunks theo header `#`–`###`, mỗi chunk giữ header + `section` trong metadata. Bảng markdown (VD thẩm quyền mua sắm) không bị cắt giữa chừng. |
| BM25 + Dense fusion | M2 | `segment_vietnamese()`, `BM25Search`, `reciprocal_rank_fusion()` | underthesea nối từ ghép bằng `_` → phải `replace("_", " ")` thì query "nghỉ phép" mới khớp. Thêm lowercase + bỏ dấu câu + stopwords ("bao", "nhiêu", "được"…) để BM25 không bị câu hỏi dạng "bao nhiêu ngày" kéo về mọi tài liệu có chữ "ngày". RRF (k=60) chỉ dùng rank nên không cần chuẩn hoá điểm BM25 (0–20) với cosine (0–1); keyword chính xác (số tiền, tên chính sách) đến từ BM25, paraphrase đến từ dense. Search ~255 ms/query. |
| Cross-encoder reranking | M3 | `CrossEncoderReranker.rerank()` | bge-reranker-v2-m3 trên CPU: **~9.3 s/query** cho 20 candidates (benchmark riêng 7.4 s), chiếm ~80% latency query. Bù lại xếp hạng rất rõ: demo "nghỉ phép" 0.991 vs "thử việc" 0.021 vs "mật khẩu" 0.0007. FlashRank chỉ ~20 ms/20 docs → lựa chọn nếu cần real-time. Model được cache cấp module để test/pipeline không load lại 2 GB nhiều lần. |
| RAGAS 4 metrics | M4 | `evaluate_ragas()`, `failure_analysis()` | Production: F 0.892 · AR 0.903 · CP 0.875 · CR 0.925 (baseline 0.917 / 0.784 / 0.867 / 0.825). Faithfulness là metric duy nhất giảm: các câu numeric/multi-hop có bước suy luận ("8 tháng < 1 năm ⇒ 100%") không có nguyên văn trong context nên NLI judge đánh không được hỗ trợ; một câu trả lời cụt "Tổng Giám đốc (CEO)" bị chấm 0. Diagnostic tree map worst metric → diagnosis/fix. |
| Contextual embeddings | M5 | `_enrich_single_call()` (+ `contextual_prepend()`) | Combined mode 1 call/chunk trả JSON {summary, questions, context, metadata}; câu `context` (VD "Đoạn văn thuộc tài liệu nghi_phep_nam_v2024.md (phiên bản 2024)…") được prepend vào text trước khi index → chunk con 256 chars vẫn mang tên chính sách + phiên bản, giúp cả BM25 lẫn dense phân biệt v2023/v2024. Chi phí: 127 calls, ~4.7 s/chunk tuần tự (~10 phút). |

---

## Phần 2: Khó khăn & Cách giải quyết (Challenges & Debugging)

**1. Gemini không hỗ trợ nhiều completion**
- **Lỗi:** `BadRequestError: Error code: 400 - 'Multiple candidates is not enabled for this model'` (RAGAS `answer_relevancy` = 0.0).
- **Debug:** Chạy `evaluate_ragas()` với 1 mẫu → chỉ `answer_relevancy` = 0, các metric khác bình thường → đọc log thấy Job[1] lỗi. `AnswerRelevancy` mặc định `strictness=3` gọi LLM với `n=3`, endpoint OpenAI-compatible của Gemini không cho `n>1`.
- **Fix:** `answer_relevancy.strictness = 1`; sau đó viết `RotatingRagasLLM` (subclass `BaseRagasLLM`) tự sinh n generations tuần tự.

**2. Hết quota free tier giữa lúc chạy RAGAS**
- **Lỗi:** hàng loạt `Exception raised in Job[67]: TimeoutError()`; probe trực tiếp ra `429 RESOURCE_EXHAUSTED … quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier, quotaValue: 20`.
- **Debug:** Timeout không phải do mạng — RAGAS (16 worker) + OpenAI SDK retry ẩn lỗi 429 thành timeout. Gọi API với `max_retries=0` mới thấy thông điệp gốc: free tier chỉ **20 request/ngày/model**, trong khi lab cần ~450 calls.
- **Fix:** viết `src/llm.py`: (a) **disk cache** theo hash prompt → chạy lại không tốn request; (b) **xoay vòng model** khi gặp 429 "PerDay" (3.8-flash → 3.7 → … → flash-lite); (c) tự hạ `reasoning_effort` "none" → "low" khi model trả 400; (d) RAGAS `RunConfig(max_workers=4, timeout=600)`.

**3. Model `gemini-2.5-flash` đã ngừng cho user mới**
- **Lỗi:** `404 - This model models/gemini-2.5-flash is no longer available to new users`.
- **Fix:** gọi `/v1beta/openai/models` liệt kê model còn dùng được rồi chọn `gemini-3.8-flash` + `gemini-embedding-001`.

**4. Hết RAM khi chạy `main.py`**
- **Hiện tượng:** tiến trình bị kill ngay bước "Loading reranker…".
- **Nguyên nhân:** `main.py` tạo `DenseSearch` 2 lần (baseline + production) → load **2 bản bge-m3** (~2 GB mỗi bản) + reranker 2 GB.
- **Fix:** cache encoder bge-m3 và cross-encoder ở cấp module (`_ENCODER`, `_MODEL_CACHE`) → mỗi model chỉ load 1 lần.

**5. Lỗi nhỏ khác**
- Summary của LLM dài hơn câu gốc 1 câu → `test_summarize_shorter_than_original` fail → thêm điều kiện: summary dài hơn gốc thì dùng extractive.
- `langchain_openai.OpenAIEmbeddings` mặc định gửi token ids (tiktoken) → Gemini không nhận → `check_embedding_ctx_length=False`.
- 2 PDF scan (`BCTC.pdf`, Nghị định 13/2023) không có text layer → bị bỏ qua; ghi nhận để bổ sung OCR.

**Kiến thức còn thiếu & cách bổ sung:**
- Cách RAGAS tính từng metric bên trong (statement extraction + NLI cho faithfulness, verdict từng context cho precision) → đọc source `ragas/metrics/_faithfulness.py`, `_context_precision.py` để hiểu vì sao câu numeric bị chấm thấp và NaN bị quy về 0.
- Rate limit/quota của LLM provider → đọc docs Gemini rate limits; rút kinh nghiệm: luôn tính budget request trước khi chạy eval, và luôn cache.

---

## Phần 3: Action Plan cho Project cá nhân (Application Plan)

### Project: Chatbot RAG tra cứu chính sách & văn bản pháp luật (Vinonymus — Day 8)

#### 1. Hiện trạng
- **Pipeline hiện tại:** convert tài liệu (văn bản pháp luật + bài báo crawl) sang markdown → chunk cố định → dense + BM25 → RRF → fallback khi điểm thấp → generation có citation, giao diện Streamlit; golden dataset ~15 câu.
- **Vấn đề / Bottlenecks:** (1) văn bản pháp luật có nhiều phiên bản/sửa đổi → dễ trả lời theo văn bản đã hết hiệu lực; (2) chunk cố định cắt ngang điều/khoản; (3) chưa có reranker nên top-k còn nhiễu; (4) nhiều văn bản là PDF scan; (5) đánh giá chưa tự động hoá, chưa đo latency.

#### 2. Kế hoạch cải tiến
1. **Chunking strategy:** Structure-aware theo Chương/Điều/Khoản (regex `^Điều \d+`) làm **parent**, child ~256–300 ký tự để retrieve → trả parent (đúng mô hình hierarchical đã chứng minh tăng Context Recall +0.10 trong lab). Lý do: câu hỏi pháp lý cần nguyên điều khoản, không phải một câu lẻ.
2. **Search retrieval:** Giữ **Hybrid BM25 + bge-m3 + RRF**; BM25 dùng underthesea + stopwords tiếng Việt (số hiệu văn bản, mức phạt cần match chính xác). Thêm **metadata filter** `effective_date`/`status` (còn hiệu lực/hết hiệu lực) trước khi fusion.
3. **Reranking:** Có — `bge-reranker-v2-m3` top-20 → top-3 cho độ chính xác; vì CPU mất ~9 s/query nên triển khai bản ONNX/GPU hoặc dùng FlashRank (~20 ms) cho chế độ chat real-time, cross-encoder cho chế độ "tra cứu kỹ".
4. **Evaluation:** RAGAS 4 metrics trên golden dataset mở rộng lên **50 câu** chia theo loại (lookup, version, negation, multi-hop, numeric) + metric custom **exact-match con số / số hiệu điều luật** cho câu numeric (RAGAS faithfulness chấm thấp các câu có tính toán). Cố định 1 judge model, log NaN riêng, cache LLM để chạy lại rẻ. Báo cáo latency từng bước như lab này.
5. **Enrichment:** **Contextual prepend** (tên văn bản + số hiệu + ngày hiệu lực + điều/khoản) bằng combined single-call — giải quyết đúng vấn đề nhầm phiên bản; HyQA cho các điều khoản hay được hỏi; OCR (easyocr/pytesseract) cho PDF scan trước khi chunk.

#### 3. Timeline triển khai
- **Tuần 1:** OCR PDF scan; chuyển sang structure-aware + hierarchical chunking theo Điều/Khoản; thêm metadata hiệu lực.
- **Tuần 2:** Contextual enrichment (combined mode, có cache); hybrid search + metadata filter; mở rộng golden dataset lên 50 câu có nhãn loại câu hỏi.
- **Tuần 3:** Tích hợp reranker (ONNX/FlashRank), đo latency breakdown; tối ưu prompt generator (trả lời câu đầy đủ, tính toán từng bước, luôn kèm citation điều khoản).
- **Tuần 4:** Chạy RAGAS + metric custom, A/B so với pipeline cũ, failure analysis bottom-5 bằng error tree; đặt ngưỡng mục tiêu cả 4 metric ≥ 0.80 và rerank < 500 ms.
