# processing/ – Xử lý dữ liệu

| File | Vai trò |
|---|---|
| `data_processing.ipynb` | Làm sạch tập train một lần: quét video, khử trùng lặp MD5, loại xung đột nhãn và video rỗng, đánh dấu khung đơn sắc, chia 5 fold, tính trọng số lớp. Chạy lại: mở notebook trong `processing/` và Run All (~2–3 phút trên CPU). |
| `preprocess.py` | Dùng chung cho notebook trích đặc trưng (`models/extract_features.ipynb`) và app demo: đọc video, bỏ khung đơn sắc, lấy 16 khung, tăng cường dữ liệu, chuẩn hoá ImageNet, `SignVideoDataset` cho PyTorch. |
| `requirements.txt` | Thư viện cần cài để chạy `data_processing.ipynb` và `preprocess.py`. |
| `output/manifest_clean.csv` | 2,772 video train sạch: `cls`, `label_idx`, `file`, `rel_path`, `md5`, `n_frames`, `n_valid`, `n_solid`, `valid_idx` (chỉ số khung hợp lệ), `fold` (0–4; fold 0 = validation chung). |
| `output/removed.csv` | 1,103 tệp bị loại và lý do (bản sao / xung đột nhãn / video rỗng). |
| `output/test_manifest.csv` | Danh sách video public/private test, md5, cờ `dup_of_train` (chỉ để diễn giải điểm). |
| `output/class_weights.csv` | Trọng số lớp kiểu balanced tính trên phần train (fold ≠ 0). |
| `output/processing_summary.csv`, `class_counts_before_after.csv`, `processing_config.json` | Số liệu cho mục 3.2 của báo cáo. |
| `output/figures/` | Hình cho báo cáo: số khung hợp lệ, lấy mẫu không lọc/có lọc khung đơn sắc, ví dụ tăng cường, số video mỗi lớp trước/sau. |

Thư viện: `opencv-python`, `numpy`, `pandas`, `matplotlib`, `scikit-learn` (+ `torch` nếu dùng `SignVideoDataset`). Cài đặt: `pip install -r processing/requirements.txt`.