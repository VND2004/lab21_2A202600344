# Lab 21 - Evaluation Report

**Học viên**: Vũ Như Đức - 2A202600344 \
**Ngày nộp**: 2026-05-07  

## 1. Setup

- **Base model**: `unsloth/Qwen2.5-3B-bnb-4bit` (T4 profile, QLoRA 4-bit)
- **Dataset**: `5CD-AI/Vietnamese-alpaca-gpt4-gg-translated`, lấy subset 200 samples tiếng Việt theo Alpaca format (`instruction`, `input`, `output`)
- **Split**: 180 train + 20 eval, tỉ lệ 90/10, random seed = 42
- **Token length**: p50 = 227, p95 = 562, p99 = 704; chọn `max_seq_length = 1024`
- **GPU**: Tesla T4, max memory 14.563 GB
- **Training config**: 3 epochs, learning rate `2e-4`, cosine schedule, warmup ratio `0.10`, train/eval batch size = 1, effective batch size = 8 qua gradient accumulation, optimizer `adamw_8bit`, `packing=False`
- **LoRA config**: target modules `["q_proj", "v_proj"]`, dropout = 0, bias = `none`, bật gradient checkpointing
- **Training cost**: ước tính `$0.07` với rate `$0.35/hour`

## 2. Rank Experiment Results

| Rank | Alpha | Trainable Params | Train Time | Peak VRAM | Eval Loss | Perplexity |
|------|-------|------------------|------------|-----------|-----------|------------|
| 8 | 16 | 1,843,200 | 3.67 min | 7.22 GB | 1.5577 | 4.7479 |
| 16 | 32 | 3,686,400 | 4.24 min | 6.62 GB | 1.5161 | 4.5544 |
| 64 | 128 | 14,745,600 | 3.65 min | 8.00 GB | 1.4768 | 4.3790 |

Perplexity được tính bằng công thức `exp(eval_loss)`. Trong các artifact hiện có, file `rank_experiment_summary.csv` có kết quả cho 3 adapter LoRA; base model chỉ có qualitative comparison, không có eval loss/perplexity riêng nên không điền số đoán.

## 3. Loss Curve Analysis

Notebook T4 tắt eval-during-training để tiết kiệm VRAM, vì vậy loss curve chủ yếu là train loss. Train loss giảm theo từng rank:

| Rank | First Logged Train Loss | Best/Low Train Loss | Last Logged Train Loss |
|------|--------------------------|---------------------|------------------------|
| 8 | 1.6165 | 1.4175 | 1.4388 |
| 16 | 1.6143 | 1.3802 | 1.3942 |
| 64 | 1.6016 | 1.2705 | 1.2768 |

Không có dấu hiệu overfitting rõ ràng trong log hiện tại vì eval loss chỉ được đo sau training, không có chuỗi eval loss theo step để thấy xu hướng eval tăng trong khi train loss giảm. Tuy nhiên, có thể thấy rank cao hơn học tập tốt hơn trên train loss và eval loss cuối cùng cũng giảm từ r=8 -> r=16 -> r=64. Điều này cho thấy adapter có capacity cao hơn đang khớp dataset tốt hơn, nhưng cần cẩn thận trong production vì dataset chỉ có 200 mẫu, nên r=64 có nguy cơ học quá sát style/noise nếu train lâu hơn hoặc mở rộng target modules.

## 4. Qualitative Comparison

### Example 1

**Prompt**: Giải thích khái niệm machine learning cho người mới bắt đầu.  
**Base**: Trả lời đúng ý chính nhưng hơi dài, lặp ý, và bị cắt ở cuối.  
**Fine-tuned (r=16)**: Giải thích machine learning là một phần của AI, học từ dữ liệu để đưa ra dự đoán.  
**Nhận xét**: Improved nhẹ về độ gần với tiếng Việt instruction-following, nhưng vẫn còn dài và chưa thật gọn.

### Example 2

**Prompt**: Viết đoạn code Python tính số Fibonacci thứ n.  
**Base**: Đưa ra cách đệ quy/vòng lặp nhưng phần code bị dang dở.  
**Fine-tuned (r=16)**: Bắt đầu bằng implementation iterative, xử lý `n < 0`, `n == 0`, `n == 1`; output vẫn bị cắt nhưng cấu trúc code tốt hơn.  
**Nhận xét**: Improved về hướng giải và edge cases, nhưng cần tăng `max_new_tokens` hoặc prompt format để code không bị cắt.

### Example 3

**Prompt**: Liệt kê 5 nguyên tắc thiết kế UI/UX.  
**Base**: Trả lời dài dòng, mục 1 quá dài nên danh sách không gọn.  
**Fine-tuned (r=16)**: Liệt kê các ý như chuyển đổi, thích ứng, đơn giản, tương thích; format có danh sách rõ hơn.  
**Nhận xét**: Improved về format list, nhưng nội dung vẫn còn hơi chung chung.

### Example 4

**Prompt**: Tóm tắt sự khác biệt giữa LoRA và QLoRA.  
**Base**: Nói LoRA/QLoRA liên quan fine-tuning, nhưng diễn giải chưa chính xác và bị cắt.  
**Fine-tuned (r=16)**: Bị sai nghiêm trọng khi giải thích LoRA thành "Layer-wise Adaptive Regularization Optimization" và regularization.  
**Nhận xét**: Degraded. Đây là case loss rõ ràng: fine-tune trên dataset general không đảm bảo factuality cho thuật ngữ kỹ thuật.

### Example 5

**Prompt**: Phân biệt prompt engineering, RAG, và fine-tuning.  
**Base**: Nêu được ba cách cải thiện mô hình nhưng chưa phân biệt rõ.  
**Fine-tuned (r=16)**: Giải thích prompt engineering là xây dựng prompt, bắt đầu tách các kỹ thuật nhưng vẫn bị cắt.  
**Nhận xét**: Slightly improved về tone và cách vào bài, nhưng chưa đủ chi tiết; cần thêm examples domain-specific trong dataset.

## 5. Conclusion về Rank Trade-off

Trên dataset này, r=64 cho perplexity tốt nhất: 4.3790, giảm so với r=16 là 4.5544 và r=8 là 4.7479. Điều này hợp lý vì rank cao hơn tăng số chiều của low-rank update, cho adapter nhiều capacity hơn để điều chỉnh các projection `q_proj` và `v_proj`. Tuy nhiên, ROI không hoàn toàn thuộc về r=64. Từ r=8 lên r=16, trainable params tăng từ 1.84M lên 3.69M và perplexity giảm khoảng 0.19. Từ r=16 lên r=64, params tăng gấp 4 lần lên 14.75M nhưng perplexity chỉ giảm thêm khoảng 0.18. Mức cải thiện có thật, nhưng bắt đầu có diminishing returns rõ về chi phí tham số và VRAM. Nếu chỉ cần nộp lab/evaluate chất lượng cao nhất, r=64 là adapter mạnh nhất trong thí nghiệm. Nếu deploy production trên T4 hoặc môi trường cần tiết kiệm storage/VRAM và load nhiều adapter, tôi chọn r=16 vì nó gần chất lượng r=64 hơn r=8, ít tham số hơn rất nhiều, và qualitative r=16 đã cải thiện format trả lời trong nhiều prompt. Quan trọng hơn, fine-tuning không sửa được knowledge gap: case LoRA vs QLoRA cho thấy model fine-tuned vẫn có thể nói sai nếu dataset không có kiến thức chuẩn.

## 6. What I Learned

- Rank LoRA là trade-off thực tế giữa capacity và chi phí: tăng rank có thể giảm perplexity, nhưng lợi ích biên giảm nhanh khi dataset nhỏ.
- Perplexity tốt hơn không đảm bảo tất cả câu trả lời tốt hơn; qualitative eval vẫn cần thiết để bắt lỗi factuality và output bị cắt.
- Với T4, QLoRA + Unsloth giúp train được model 3B rất nhanh, nhưng cần thiết kế eval tiết kiệm VRAM và ghi log rõ ràng ngay từ đầu.
