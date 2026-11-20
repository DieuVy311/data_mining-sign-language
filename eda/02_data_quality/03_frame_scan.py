"""Bước 2.3 – Quét nội dung khung hình của TOÀN BỘ video: khung đơn sắc, video rỗng, độ chuyển động.
Khung đơn sắc: độ lệch chuẩn điểm ảnh lớn nhất giữa 3 kênh màu < 3.
Chạy:  python 03_frame_scan.py [--root ĐƯỜNG_DẪN_DATASET] [--start N] [--end M] [--out FILE.csv]
  Mặc định quét hết 8.364 video (~10–12 phút) và ghi video_quality_flags.csv.
  Đường dẫn có dấu cách: đặt trong ngoặc kép. --start/--end là chỉ số dòng trong video_metadata.csv.
Cột đầu ra: split, cls, file, size_kb, n_read, solid_frames, leading_solid, solid_ratio, motion
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import *
import cv2

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument('--root', nargs='+', default=[str(DEFAULT_ROOT)])
ap.add_argument('--start', type=int, default=0)
ap.add_argument('--end', type=int, default=None)
ap.add_argument('--out', default=str(FLAGS_CSV))
a = ap.parse_args()
root = ' '.join(a.root)

meta = load_meta().iloc[a.start:a.end]
rec = []
for r in meta.itertuples():
    c = cv2.VideoCapture(str(video_path(root, r.split, r.cls, r.file)))
    sol = n = lead = 0; mot = []; prev = None; leading = True
    while True:
        ok, f = c.read()
        if not ok: break
        n += 1; g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.int16)
        s = is_solid(f)
        sol += s
        if leading and s: lead += 1
        elif not s: leading = False
        if prev is not None: mot.append(float(np.abs(g - prev).mean()))
        prev = g
    c.release()
    rec.append(dict(split=r.split, cls=r.cls, file=r.file, size_kb=r.size_kb, n_read=n, solid_frames=sol,
                    leading_solid=lead, solid_ratio=sol / max(n, 1), motion=float(np.mean(mot)) if mot else 0.0))
d = pd.DataFrame(rec)
d.to_csv(a.out, index=False, encoding='utf-8-sig')
print(len(d), 'videos ->', a.out)
for sp, g in d.groupby('split', sort=False):
    print(sp, 'any solid:', (g.solid_frames > 0).sum(), ' all solid:', (g.solid_ratio >= .99).sum(),
          ' low motion(<0.5):', (g.motion < 0.5).sum())
