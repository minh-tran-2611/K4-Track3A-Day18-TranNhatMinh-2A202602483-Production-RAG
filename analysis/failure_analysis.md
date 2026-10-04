# Failure Analysis — Lab 18: Production RAG

**Họ và tên học viên:** Trần Nhật Minh (2A202602483)  
**Khóa:** K4 - Track 3A  

---

## Cấu hình chạy

| Thành phần | Naive Baseline | Production |
|-----------|----------------|------------|
| Chunking | `chunk_basic` (paragraph, 500 chars) → 57 chunks | `chunk_hierarchical` (parent 2048 / child 256) → 127 children |
| Enrichment | — | M5 combined single-call (context + summary + HyQA + metadata), contextual prepend vào text được index |
| Search | Dense only (bge-m3, top-3) | Hybrid: BM25 (underthesea + stopwords) + Dense bge-m3 → RRF (k=60), top-20 |
| Rerank | — | `BAAI/bge-reranker-v2-m3` top-20 → top-3 child → trả về **parent** (dedupe) |
| Generator | Gemini (prompt gốc) | Gemini, temperature 0, prompt ưu tiên phiên bản hiện hành |
| RAGAS judge | Gemini flash/flash-lite (OpenAI-compatible) + `gemini-embedding-001` | như baseline |

> Corpus: 26/28 tài liệu được index. `BCTC.pdf` và `Nghi_dinh_so_13-2023...pdf` là PDF scan ảnh, không có text layer → bị bỏ qua (chưa OCR). Không câu hỏi nào trong test set phụ thuộc vào 2 file này.

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 0.9167 | 0.8917 | −0.0250 |
| Answer Relevancy | 0.7837 | 0.9034 | +0.1196 |
| Context Precision | 0.8667 | 0.8750 | +0.0083 |
| Context Recall | 0.8250 | 0.9250 | +0.1000 |

**Nhận xét:** Production cải thiện rõ ở **Context Recall (+0.10)** nhờ hybrid search + trả về parent chunk (đủ ngữ cảnh cho câu multi-hop/version) và **Answer Relevancy (+0.12)** nhờ prompt yêu cầu trả lời đúng trọng tâm, nêu phiên bản hiện hành. Faithfulness giảm nhẹ (−0.025) nhưng phần lớn do 1 câu bị judge chấm 0 (xem #1) — không phải hallucination thật. Cả 4 metric production đều ≥ 0.75.

## Latency breakdown (Production, CPU)

| Bước | Thời gian |
|------|-----------|
| Chunking (26 docs) | 0.07 s |
| Enrichment 127 chunks (1 call/chunk, lần chạy đầu không cache) | ~597 s (~4.7 s/chunk, tuần tự) |
| Indexing BM25 + bge-m3 | 53.7 s |
| Load reranker | 9.3 s |
| Hybrid search / query | avg 255 ms (max 287 ms) |
| Rerank 20 → 3 / query | **avg 9 269 ms** (max 10 285 ms) |
| LLM generation / query | avg 2 003 ms (max 6 099 ms) |

→ Bottleneck là cross-encoder trên CPU (~80% latency query). Benchmark cùng 20 docs: bge-reranker-v2-m3 ≈ 7.4 s vs FlashRank ≈ 20 ms.

---

## Bottom-5 Failures

Thứ tự theo `avg_score` tăng dần trong `reports/ragas_report.json → failures`.

### #1
- **Question:** Muốn mua thiết bị trị giá 55 triệu cần ai phê duyệt?
- **Expected:** Đơn hàng trên 50.000.000 VNĐ cần Tổng Giám đốc (CEO) phê duyệt.
- **Got:** "Tổng Giám đốc (CEO)"
- **Scores:** F=0.00 · AR=0.82 · CP=1.00 · CR=1.00
- **Worst metric:** faithfulness
- **Error Tree:** Output sai? → **Không** (đúng với bảng thẩm quyền trong `mua_sam.md`: "Trên 50.000.000 VNĐ → Tổng Giám đốc (CEO)") → Context đúng? → Có (CP=CR=1) → Query OK? → Có → **Lỗi ở bước đánh giá/format câu trả lời**.
- **Root cause:** Câu trả lời chỉ là một cụm danh từ, không có chủ ngữ/điều kiện. RAGAS faithfulness tách answer thành "statements" rồi NLI với context; một cụm từ trơ trọi không tạo được statement kiểm chứng được (hoặc judge parse lỗi → NaN, code quy về 0). Ngoài ra bảng markdown khiến judge khó nối "55 triệu" với dòng "Trên 50.000.000".
- **Suggested fix:** Prompt generator yêu cầu trả lời **câu đầy đủ, nêu căn cứ** ("Đơn 55 triệu > 50 triệu nên cần CEO phê duyệt"); log riêng NaN thay vì quy về 0 để phân biệt lỗi judge với lỗi pipeline.

### #2
- **Question:** Thâm niên bao nhiêu năm thì được cộng thêm ngày phép?
- **Expected:** v2024: từ 3 năm trở lên, +1 ngày mỗi 3 năm; v2023 cũ yêu cầu 5 năm.
- **Got:** "Từ 3 năm trở lên được cộng thêm 1 ngày phép (theo v2024, thay thế v1.0 năm 2023 quy định 5 năm)" — đúng hoàn toàn.
- **Scores:** F=1.00 · AR=0.88 · CP=0.00 · CR=1.00
- **Worst metric:** context_precision
- **Error Tree:** Output sai? → Không → Context đúng? → Có (top-2 là `nghi_phep_nam_v2024` và `v2023`, đúng 2 tài liệu cần) → Query OK? → Có → **Lỗi ở metric**.
- **Root cause:** Context precision = 0 trong khi cả 2 context đều liên quan → nhiều khả năng judge (flash-lite) trả verdict sai định dạng → NaN → 0. Cũng có thể judge coi bản v2023 là "không hữu ích" vì đã bị thay thế, nhưng như thế điểm tối thiểu vẫn phải 1.0 cho context đầu tiên.
- **Suggested fix:** Dùng judge mạnh hơn/ổn định hơn (không xoay vòng sang flash-lite), bật `raise_exceptions` khi debug để thấy lỗi parse; thêm metadata `version`/`superseded` để filter (chỉ đưa v2023 khi câu hỏi hỏi về thay đổi).

### #3
- **Question:** Thông tin lương thuộc cấp độ phân loại dữ liệu nào?
- **Expected:** "Bí mật" (theo quy chế lương), và dữ liệu Bí mật (cấp 3) phải mã hóa khi truyền, need-to-know.
- **Got:** "Thông tin lương thuộc cấp độ **Bí mật**."
- **Scores:** F=1.00 · AR=0.91 · CP=0.00 · CR=1.00
- **Worst metric:** context_precision
- **Error Tree:** Output sai? → Không (thiếu phần mở rộng "cấp 3, mã hóa…") → Context đúng? → Có (`phan_loai_du_lieu.md` + `ky_luong.md` — chính là 2 nguồn của câu multi-hop) → Query OK? → Có → **Lỗi ở metric + answer chưa đầy đủ**.
- **Root cause:** Giống #2: 2 context đều cần thiết nhưng CP=0 → judge verdict lỗi/NaN. Phần answer ngắn hơn ground truth (không nêu cấp 3 và yêu cầu bảo vệ) — generator trả lời tối giản.
- **Suggested fix:** Prompt: "với câu hỏi về phân loại/quy định, nêu cả cấp độ và yêu cầu xử lý đi kèm"; ổn định judge như #2.

### #4
- **Question:** Nhân viên được tài trợ khóa học 25 triệu, nghỉ việc sau 8 tháng hoàn thành khóa học. Phải hoàn trả bao nhiêu?
- **Expected:** Cam kết ≥ 1 năm; nghỉ sau 8 tháng → hoàn trả 100% = 25.000.000 VNĐ.
- **Got:** "Hoàn trả 100% chi phí (25.000.000 VNĐ) do nghỉ trước thời hạn cam kết ít nhất 1 năm." — đúng.
- **Scores:** F=0.33 · AR=0.82 · CP=1.00 · CR=1.00
- **Worst metric:** faithfulness
- **Error Tree:** Output sai? → Không → Context đúng? → Có (`hoan_chi_dao_tao.md` top-1) → Query OK? → Có → **Lỗi ở bước suy luận được chấm**.
- **Root cause:** Answer chứa phép suy luận số (8 tháng < 12 tháng ⇒ 100%; 100% × 25 triệu = 25.000.000) — các statement này **không xuất hiện nguyên văn** trong context nên NLI judge đánh "không được hỗ trợ". Đây là giới hạn đã biết của faithfulness với câu numeric/multi-hop.
- **Suggested fix:** Prompt yêu cầu trích dẫn điều khoản gốc trước rồi mới tính ("Theo chính sách: cam kết ít nhất 1 năm, nghỉ trước hạn → hoàn 100%. 8 tháng < 1 năm → …"); với câu numeric nên dùng thêm metric custom (exact-match con số) bên cạnh RAGAS.

### #5
- **Question:** Nhân viên tạm ứng 15 triệu, sau 20 ngày mới thanh toán. Bị phạt bao nhiêu?
- **Expected:** Hạn 15 ngày → quá 5 ngày; phí 2%/tháng × 15.000.000 = 300.000 VNĐ/tháng (~50.000 VNĐ pro-rata cho 5 ngày).
- **Got:** "Bị tính phí 2%/tháng trên số tiền chưa hoàn ứng, khấu trừ vào lương tháng kế tiếp." — đúng quy định nhưng **không tính ra số tiền**.
- **Scores:** F=1.00 · AR=0.82 · CP=1.00 · CR=0.50
- **Worst metric:** context_recall
- **Error Tree:** Output sai? → **Thiếu** (không trả lời "bao nhiêu") → Context đúng? → Có (`tam_ung.md` chứa cả "15 ngày" và "2%/tháng") → Query OK? → Có → **Fix ở bước generation**.
- **Root cause:** Context recall 0.5 vì ground truth có các câu tính toán (300.000 VNĐ, 50.000 VNĐ pro-rata) không thể "attribute" về context. Generator chỉ lặp lại quy định, không thực hiện phép tính mà câu hỏi yêu cầu.
- **Suggested fix:** Prompt: "Nếu câu hỏi yêu cầu số tiền/số ngày, hãy tính cụ thể từng bước dựa trên số liệu trong context"; hoặc thêm bước query decomposition (hạn thanh toán? → số ngày quá hạn? → phí?).

---

## Case Study (cho presentation)

**Question chọn phân tích:** "Nhân viên tạm ứng 15 triệu, sau 20 ngày mới thanh toán. Bị phạt bao nhiêu?" (#5)

**Error Tree walkthrough:**
1. Output đúng? → **Chưa đủ** — nêu đúng mức phí 2%/tháng nhưng không trả lời con số "bao nhiêu".
2. Context đúng? → **Có** — top-1 sau rerank là parent của `tam_ung.md`, chứa đủ "thanh toán trong vòng 15 ngày" và "phí 2%/tháng trên số tiền chưa hoàn ứng". Retrieval không có lỗi (CP = 1.0).
3. Query rewrite OK? → Query gốc đủ rõ; không cần rewrite cho retrieval. Nhưng câu hỏi là dạng **numeric multi-step** (so sánh 20 vs 15 ngày → 5 ngày quá hạn → 2% × 15 triệu × 5/30).
4. Fix ở bước: **Generation** — prompt hiện tại tối ưu cho "trả lời ngắn, chỉ dựa context" nên LLM né phép tính. Thêm hướng dẫn "tính toán từng bước khi câu hỏi hỏi số tiền/số ngày" (hoặc chain-of-thought ẩn + trả lời cuối) và một test-case numeric để regression.

**Nếu có thêm 1 giờ, sẽ optimize:**
- Prompt generator: câu trả lời đầy đủ (có chủ ngữ + căn cứ) và tính toán từng bước cho câu numeric → kỳ vọng tăng faithfulness ở #1/#4 và recall ở #5.
- Judge ổn định: cố định 1 model judge (không xoay vòng sang flash-lite) và log NaN riêng — #2, #3 rất có thể là lỗi judge chứ không phải lỗi retrieval.
- Metadata filter theo `version`/`effective_date`: chỉ đưa tài liệu superseded khi câu hỏi hỏi về thay đổi chính sách.
- Thay cross-encoder bằng FlashRank hoặc chạy reranker trên GPU/ONNX để giảm rerank từ ~9 s xuống < 100 ms/query.
- OCR (`pytesseract`/`easyocr`) cho 2 PDF scan để corpus đủ 28/28 tài liệu.
