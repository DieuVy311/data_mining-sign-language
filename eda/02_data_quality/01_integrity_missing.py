"""Bước 2.1 – Toàn vẹn dữ liệu: giá trị thiếu, tệp lỗi, giá trị bất thường ở mức metadata.
Chỉ dùng video_metadata.csv, chưa giải mã khung hình (việc đó ở 03_frame_scan.py).
Chạy:  python 01_integrity_missing.py
Đầu ra: integrity_summary.txt
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import *

HERE = Path(__file__).resolve().parent
m = load_meta()

with report(HERE / 'integrity_summary.txt'):
    print('== Giá trị thiếu (NaN) theo cột, tách theo tập (cls trống ở tập test là bình thường)')
    print(m.drop(columns=['base']).isna().groupby(m.split).sum().to_string())

    print('\n== Tệp không đọc được hoặc metadata vô nghĩa')
    for name, mask in [('frames == 0', m.frames <= 0), ('fps <= 0', m.fps <= 0),
                       ('w/h == 0', (m.w == 0) | (m.h == 0)), ('dur không xác định', m.dur.isna())]:
        print(f'{name}:', {s: int(mask[m.split == s].sum()) for s in SPLITS})

    print('\n== Giá trị bất thường (tự phát hiện bằng ngưỡng tuyệt đối đơn giản)')
    print('frames < 16:', {s: int((g.frames < 16).sum()) for s, g in m.groupby('split', sort=False)})
    print('dur > 5s   :', {s: int((g.dur > 5).sum()) for s, g in m.groupby('split', sort=False)})
    print('fps < 5    :', {s: int((g.fps < 5).sum()) for s, g in m.groupby('split', sort=False)})
    print('size < 10KB:', {s: int((g.size_kb < 10).sum()) for s, g in m.groupby('split', sort=False)})
    print('\ncác tệp có size < 10KB hoặc frames < 16 hoặc fps < 5 (ứng viên cần xem lại):')
    cand = m[(m.size_kb < 10) | (m.frames < 16) | (m.fps < 5)]
    print(cand[['split', 'cls', 'file', 'frames', 'fps', 'size_kb']].head(30).to_string(index=False))
    print('... tổng', len(cand), 'tệp')

    print('\n== Quy ước đặt tên')
    suf = m.file.str.contains(r'_\d+\.mp4$')
    print('tệp có hậu tố _N theo tập:', suf.groupby(m.split).sum().to_dict())
