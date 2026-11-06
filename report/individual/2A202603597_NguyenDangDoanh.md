# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202603597
- **SEED** (in ở ô đầu notebook): 1572

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☑ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☑ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước (câu 1–4) | ☑ |
| [`handcalc2.jpg`](handcalc2.jpg) — ảnh tính tay trang 2 (câu 5) | ☑ |

## 2. Kết quả chính

| Cách tiếp cận | F1 test | F1 SUB 40 | p95 ms/tin | USD/1k tin |
| --- | --- | --- | --- | --- |
| 1. BoW + LogReg | 0.874 | 0.895 | 0.027 | 0 |
| 2. BoW + MLP | 0.944 | 0.974 | 0.170 | 0 |
| 3a. MiniLM + kNN | 0.885 | 0.976 | 39.366 | 0 |
| 3b. MiniLM + LogReg | 0.956 | 1.000 | 6.504 | 0 |
| 3c. MiniLM + MLP head | 0.951 | 1.000 | 7.744 | 0 |
| 4a. LLM nhỏ zero-shot | 0.952 | 0.952 | 2749.174 | 0 |
| 4b. LLM lớn zero-shot | 0.952 | 0.952 | 2772.734 | 0 |
| 4c. LLM nhỏ few-shot | 0.950 | 0.950 | 3533.625 | 0 |

Hàng LLM chỉ đo trên 40 tin của tập SUB. USD của LLM bằng 0 vì bảng giá trong notebook chưa có model Groq, không có nghĩa là dịch vụ miễn phí.

## 3. Dự đoán và thực tế

**`emb_search`.** Tôi đoán tin gần nhất với câu truy vấn là ham. Thực tế cả 6 tin gần nhất đều là spam (cosine 0.513–0.546). Tôi đoán lệch vì câu truy vấn không có các từ như "free" hay "win", nhưng encoder so sánh theo nghĩa nên vẫn nhận ra đây là kiểu tin mời nhận thưởng.

**`dl_relu`.** Tôi đoán bỏ ReLU thì F1 sẽ xấp xỉ LogReg ở Phần 1 (0.874). Thực tế F1 đạt 0.935, gần với MLP có ReLU (0.944). Mô hình đúng là trở thành tuyến tính, nhưng tôi đã bỏ qua khác biệt về cách huấn luyện: Adam và mini-batch có thể học tốt hơn gradient descent 300 epoch, vốn chưa hội tụ hẳn (recall trên train chỉ 0.831).

**`hard_eval`.** Tôi đoán BoW khó nhất ở "né từ khóa" và "tưởng spam", còn LLM khó nhất ở prompt injection. Thực tế BoW đúng 90–100% ở "tưởng spam", còn LLM khó nhất ở chính nhóm này (60–70%), trong khi injection đạt 83–100%. BoW có xu hướng đoán ham nên thường đúng ở nhóm toàn tin ham. LLM thì quá thận trọng với các từ như "FREE" và "Cash prize".

## 4. Phân tích 50 case khó

BoW + MLP tụt nhiều nhất (F1 từ 0.974 trên SUB xuống 0.410), BoW + LogReg tụt gần tương đương (0.895 xuống 0.333). BoW + MLP yếu nhất ở nhóm "Né từ khoá" (đúng 0%) và "Spam giọng tự nhiên" (đúng 10%).

Vì BoW + MLP đúng 0% ở nhóm "Né từ khoá", nó đã sai cả hai tin spam dưới đây:

- **Case #29:** `C0ngr4tulat10ns! Y0u h4ve w0n a £1000 g1ft c4rd. Cl4im n0w at w1n-gift.c0m`. Thay chữ bằng số tạo ra các token như `w0n`, `cl4im`, `n0w`, khác với `won`, `claim`, `now`. Những dấu hiệu spam mà model đã học không còn khớp được.
- **Case #31:** `U have been s.e.l.e.c.t.e.d for a c.a.s.h r.e.w.a.r.d, c.a.l.l 09061 701 461`. Dấu chấm khiến `tokenize` tách từ thành các chữ cái rời (`c`, `a`, `s`, `h`), nên các từ `cash`, `reward`, `call` biến mất khỏi đầu vào.

Cơ chế chung là bag-of-words chỉ đếm token khớp chính xác và không hiểu nghĩa của tin. Đổi cách viết nhưng giữ nguyên ý là đủ để đánh lừa model.

## 5. Thử thách cá nhân

**Đề #1 (theo SEED 1572):** đổi ngưỡng quyết định 0.5 → 0.2 và → 0.8 cho mô hình Phần 1, báo precision/recall/F1 từng ngưỡng, rồi chọn ngưỡng cho một ngân hàng sợ chặn nhầm tin OTP.

**Cách làm:**

```python
p_te = predict_proba(X_te, w, b)
for thr in [0.2, 0.5, 0.8]:
    pred = (p_te >= thr).astype(int)
    m = metrics(y_te, pred)
    print(f"ngưỡng {thr}: precision={m['precision']:.3f} · recall={m['recall']:.3f} · f1={m['f1']:.3f}")
```

**Kết quả:**

| Ngưỡng | Precision | Recall | F1 |
| --- | --- | --- | --- |
| 0.2 | 0.855 | 0.940 | 0.895 |
| 0.5 | 0.964 | 0.800 | 0.874 |
| 0.8 | 1.000 | 0.630 | 0.773 |

**Giải thích:** Hạ ngưỡng xuống 0.2 thì bắt được nhiều spam hơn (recall 0.940) nhưng chặn nhầm nhiều tin ham hơn (precision 0.855). Nâng lên 0.8 thì không chặn nhầm tin nào (precision 1.000) nhưng lọt nhiều spam hơn (recall 0.630). Ngân hàng nên chọn ngưỡng 0.8. Chặn nhầm tin OTP là false positive, và ngưỡng 0.8 cho precision cao nhất (1.000, không chặn nhầm tin ham nào trên tập test). Cái giá là recall giảm từ 0.800 xuống 0.630, tức lọt thêm spam, nhưng với ngân hàng, mất một tin OTP nghiêm trọng hơn để lọt vài tin spam.

## 6. Phương án cho sản phẩm thật

Tôi chọn 3b (MiniLM + LogReg). Phương án này có p95 6.5 ms, thấp hơn nhiều so với yêu cầu 50 ms, và F1 test cao nhất (0.956). Tốc độ khoảng 1.829 tin/s cũng vượt nhu cầu trung bình 116 tin/s, và không tốn phí API cho mỗi tin.

LLM không đạt yêu cầu độ trễ: p95 qua Groq khoảng 2.7–3.5 giây. Nếu dùng gpt-6-luna, chi phí ước tính khoảng $105/ngày, tức khoảng $3.150/tháng, chưa tính giới hạn tốc độ gọi API.

Đánh đổi tôi chấp nhận là 3b yếu hơn ở các tin khó: F1 trên 50 case khó chỉ 0.640, trong khi LLM nhỏ đạt 0.889. Có thể thêm router chuyển các tin lưỡng lự lên LLM, nhưng cách này chỉ đạt SLA nếu dưới 5% tin phải chuyển lên LLM.

## 7. AI log và điều đã học

| Phần | Bạn hỏi AI gì (prompt) | AI trả lời/viết gì | Bạn đã sửa / kiểm chứng gì |
|---|---|---|---|
Trước khi làm | Gửi link repo lab, nhờ đọc và gợi ý làm lab | Tóm tắt vùng cấm / vùng được dùng AI và gợi ý câu hỏi cho từng phần |
Phần 0 – tải dữ liệu | Dán thông báo lỗi khi tải dữ liệu / Colab mất runtime, hỏi nguyên nhân | Giải thích nguyên nhân (mạng, runtime reset, thiếu biến) và hướng kiểm tra | Tự chạy lại Runtime → Run all, xác nhận dữ liệu tải được |
TODO 1.3 | Vì sao grad_w = Xᵀ(p − y)/n và grad_b = mean(p − y) khi dùng BCE + sigmoid | Giải thích đạo hàm từng bước: σ'(z) triệt tiêu với mẫu số của BCE nên còn lại (p − y) | Tự chứng minh lại trên giấy, tự code train_logreg bằng numpy, không dùng code của AI |
TODO 2.2 | Mỗi bước zero_grad → forward → loss → backward → step làm gì; quên zero_grad thì sao | Gradient bị cộng dồn qua các batch nếu không xoá, làm bước cập nhật sai dần | Tự viết 5 dòng TODO 2.2; dùng ý này để trả lời "Giải thích 2" bằng lời của mình |
Phần 4 | Cách lấy API key miễn phí (Gemini/Groq) và thêm vào Colab Secrets | Các bước tạo key, bật Notebook access, đổi LLM_PROVIDER | Tự tạo key, thêm vào Secrets, kiểm tra notebook không chứa key trước khi nộp |
Phần 2 | Cách đọc đường train loss và val loss để nhận diện overfitting | Dấu hiệu: train loss giảm tiếp nhưng val loss chững lại rồi tăng | Tự đối chiếu với biểu đồ của mình, xác định epoch bắt đầu overfit và viết câu trả lời bằng số liệu của mình |
Phần 5 | Giải thích p50/p95, throughput, đường Pareto, phương án bị "lấn át" | Giải thích kèm ví dụ đời thường | Tự đọc biểu đồ của mình, chỉ ra phương án nằm trên đường Pareto |
Phần 5 | Sau khi tự làm 5 phép tính, nhờ kiểm tra lại cách làm | Xác nhận các bước, chỉ ra chỗ làm tròn chưa đúng | Sửa lại trên giấy, đối chiếu với hash trong notebook, chụp ảnh hand_calc.jpg |
Phần 5 | Đưa phương án đã chọn cho p95 < 50 ms/tin và thiết kế router, nhờ phản biện | Chỉ ra lỗ hổng: ngưỡng chuyển sang LLM, chi phí khi lượng tin tăng | Tự chỉnh lại ngưỡng và lập luận, dẫn số liệu từ bảng so sánh của mình |

**Điều đã học:** Tôi từng nghĩ loss với lr=50 sẽ dao động mãi, nhưng nó chỉ vọt lên ở epoch đầu rồi hội tụ, và loss train thấp chưa chắc là model tốt. Tôi cũng hiểu rằng điểm cao trên tập test thường chưa đủ: BoW + MLP đạt F1 0.974 trên SUB nhưng chỉ còn 0.410 ở 50 case khó.
