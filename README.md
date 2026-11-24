# Nhận diện ngôn ngữ ký hiệu tiếng Việt từ video – BTL Khai phá dữ liệu HK261

Phân loại video ngôn ngữ ký hiệu tiếng Việt vào 100 lớp từ vựng. Dữ liệu là đề vòng sơ loại Olympic Trí tuệ Nhân tạo Sinh viên 2025 (PTIT); kết quả được chấm bằng Macro-F1 trên tập private test của nền tảng AIChallenge.

Đề tài so sánh bốn thuật toán phân lớp cổ điển (Logistic Regression làm baseline, kNN, Random Forest, SVM tuyến tính) trên hai bộ đặc trưng EfficientNet-B0 (đóng băng và tinh chỉnh) với mô hình học sâu CNN-LSTM.

## Kết quả

Macro-F1 trên fold 0 (tập kiểm định, 555 video) và trên tập private test (2.859 video, do AIChallenge chấm):

| Đặc trưng | Mô hình | Fold 0 | Private |
| --- | --- | --- | --- |
| Đóng băng | Logistic Regression | 0,680 | 0,340 |
| Đóng băng | kNN | 0,635 | 0,195 |
| Đóng băng | Random Forest | 0,597 | 0,153 |
| Đóng băng | SVM tuyến tính | 0,682 | 0,297 |
| Đóng băng | CNN-LSTM A0 / A2 | 0,783 / 0,775 | 0,415 / 0,440 |
| Tinh chỉnh | Logistic Regression | 0,824 | 0,662 |
| Tinh chỉnh | kNN | 0,717 | 0,509 |
| Tinh chỉnh | Random Forest | 0,664 | 0,439 |
| Tinh chỉnh | SVM tuyến tính | 0,821 | 0,641 |
| Tinh chỉnh | **CNN-LSTM B** | **0,859** | **0,687** |

Chi tiết và phân tích trong báo cáo `report/main.pdf`.

## Cấu trúc thư mục

| Thư mục / file | Nội dung |
| --- | --- |
| `data/dataset/` | Dữ liệu gốc: `train/` (3.875 video, 100 lớp), `public_test/` (1.630), `private_test/` (2.859), `label_mapping.pkl`. Không đưa lên GitHub. |
| `eda/` | Script và kết quả phân tích khám phá dữ liệu (`python run_all.py ../data/dataset`) |
| `processing/` | `data_processing.ipynb` (làm sạch, khử trùng lặp, chia 5 fold), `preprocess.py` (lấy mẫu khung, chuẩn hoá, tăng cường; dùng chung cho mọi mô hình và demo), `output/` (manifest, bảng, hình) |
| `models/extract_features.ipynb` | Trích đặc trưng EfficientNet-B0 đóng băng → `models/features/` |
| `models/svm/` | Thí nghiệm SVM: ba cách gộp theo thời gian, PCA, SVM tuyến tính và RBF |
| `models/cnn_lstm/` | CNN-LSTM: giai đoạn A (CNN đóng băng, biến thể A0, A2) và giai đoạn B (tinh chỉnh CNN) |
| `models/extract_finetuned.ipynb` | Trích đặc trưng từ CNN đã tinh chỉnh của CNN-LSTM B → `models/features_ft/` |
| `models/classical/` | So sánh bốn thuật toán cổ điển trên hai bộ đặc trưng; permutation importance |
| `demo/` | Ứng dụng Streamlit |
| `report/` | Báo cáo LaTeX (pdfLaTeX + `-shell-escape`, cần gói `vntex`) |

Mỗi notebook ghi kết quả (bảng chỉ số, hình, tệp nộp bài) vào thư mục `output/` bên cạnh.

## Cài đặt

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Tải dữ liệu từ AIChallenge và giải nén vào `data/dataset/`.

Trọng số mô hình (`*.pth`, `*.joblib`) và tệp đặc trưng (`*.npy`) có dung lượng lớn nên không nằm trong repo. Trọng số được đính kèm ở mục [Releases](https://github.com/DieuVy311/data_mining-sign-language/releases); tải về và đặt vào thư mục `output/` của notebook tương ứng (`models/cnn_lstm/output/`, `models/svm/output/`).

## Thứ tự chạy

1. `processing/data_processing.ipynb`: tạo manifest dữ liệu sạch và fold.
2. `models/extract_features.ipynb` (GPU): đặc trưng đóng băng.
3. `models/svm/svm.ipynb`: chọn cách gộp đặc trưng và cấu hình SVM.
4. `models/cnn_lstm/cnn_lstm.ipynb` (GPU): huấn luyện CNN-LSTM A0, A2, B.
5. `models/extract_finetuned.ipynb` (GPU): đặc trưng tinh chỉnh từ CNN-LSTM B.
6. `models/classical/classical.ipynb`: so sánh bốn thuật toán trên hai bộ đặc trưng.

Các bước 2, 4, 5 cần GPU (nhóm dùng Kaggle Notebook với Tesla T4). Mọi bước có yếu tố ngẫu nhiên đều dùng `SEED = 42`.

## Về dữ liệu

- Tên lớp trong `label_mapping.pkl` ở dạng Unicode NFD; luôn chuẩn hoá cả tên thư mục và khoá về NFC (`unicodedata.normalize('NFC', ...)`) trước khi tra nhãn. Tệp nộp bài phải giữ nhãn ở dạng NFD.
- Tập train có 1.099 tệp trùng lặp (MD5) và khoảng 20% video chứa khung đơn sắc; xem `eda/` và `processing/`.
