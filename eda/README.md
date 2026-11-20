# EDA_output

Mã nguồn và kết quả EDA cho bộ dữ liệu video ngôn ngữ ký hiệu. Mỗi thư mục là một bước của quy trình EDA; kết quả nằm cạnh script sinh ra nó. Các script chỉ đọc dataset, không sửa.

## Chạy

```
pip install -r requirements.txt
python run_all.py                     # toàn bộ (~13 phút)
python run_all.py --skip-frame-scan   # bỏ bước quét khung hình chậm, dùng lại kết quả cũ (~2 phút)
python run_all.py --from 03           # chạy lại từ bước 03
```

Dataset mặc định ở `../dataset`. Nếu ở chỗ khác thì truyền đường dẫn, ví dụ `python run_all.py "D:\du lieu\dataset"`.

## Các bước

| Thư mục | Nội dung |
|---|---|
| `01_overview` | Quét metadata từng video (`video_metadata.csv`); cấu trúc tập, định dạng, tệp nhãn |
| `02_data_quality` | Giá trị thiếu, trùng lặp MD5 (`file_hashes.csv`), quét khung đơn sắc (`video_quality_flags.csv`) |
| `03_univariate` | Thống kê mô tả, ngoại lai, histogram các thuộc tính số |
| `04_target_analysis` | Phân phối nhãn, mất cân bằng lớp |
| `05_bivariate_multivariate` | Tương quan giữa thuộc tính, quan hệ thuộc tính – nhãn |
| `06_split_comparison` | So sánh phân phối train / public_test / private_test |
| `07_visual_inspection` | Khung hình mẫu; xem lại các video bị bước 02 đánh dấu |

Các bước sau dùng kết quả của bước trước, nên khi chạy lẻ phải theo thứ tự trên. `common.py` chứa phần dùng chung.

## Lưu ý

Tên lớp trong `label_mapping.pkl` ở dạng Unicode NFD, còn tên thư mục lớp ở dạng NFC. Phải chuẩn hoá về NFC trước khi tra nhãn: `unicodedata.normalize('NFC', ten_lop)`.
