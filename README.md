# Nhận diện ngôn ngữ ký hiệu cho người khiếm thính từ video – BTL Khai phá dữ liệu (HK261)

Phân loại video ngôn ngữ ký hiệu tiếng Việt vào 100 lớp từ vựng. Dữ liệu: đề vòng sơ loại Olympic Trí tuệ Nhân tạo Sinh viên 2025 (PTIT), đánh giá Macro F1 trên AIChallenge.

## Cấu trúc thư mục

| Thư mục / file | Nội dung |
| --- | --- |
| `data/dataset/` | `train/` (3,875 video, 100 lớp), `public_test/` (1,630), `private_test/` (2,859), `label_mapping.pkl`. Chỉ đọc, không sửa. |
| `data/dataset.zip` | Bản nén của `data/dataset/` |
| `eda/` | Script và kết quả EDA (chạy trong `eda/`: `python run_all.py ../data/dataset`) |
| `processing/` | Xử lý dữ liệu: `data_processing.ipynb` (làm sạch, chia fold), `preprocess.py` (dùng chung khi train/test), `output/` (manifest, bảng, hình) |
| `models/svm/`, `models/cnn_lstm/` | Notebook, trọng số, submission CSV |
| `models/vit/` | Vision Transformer |
| `demo/` | App Streamlit |
| `report/` | Báo cáo LaTeX (pdfLaTeX + `-shell-escape`, cần gói `vntex`) |
| `docs/` | File phân công, bản thảo giới thiệu & tổng quan |

## Lưu ý

- Tên lớp trong `label_mapping.pkl` ở dạng Unicode NFD; luôn chuẩn hoá cả tên thư mục và khoá về NFC (`unicodedata.normalize('NFC', ...)`) trước khi tra nhãn.
- Train có 1,099 file trùng lặp (MD5) và ~20% video chứa khung đơn sắc: chia train/validation theo nhóm MD5, xem `eda/` và `processing/`.