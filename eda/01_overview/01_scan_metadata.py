"""Bước 1.1 – Quét dataset, ghi metadata từng video (không giải mã khung hình).
Chạy:  python 01_scan_metadata.py [đường_dẫn_dataset]
Đầu ra: video_metadata.csv  (split, cls, file, size_kb, frames, fps, w, h, dur)
"""
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import *
import cv2

HERE = Path(__file__).resolve().parent
ROOT = dataset_root()

rows = []
for split in SPLITS:
    base = ROOT / split
    if split == 'train':
        for cls in sorted(os.listdir(base)):
            rows += [(split, cls, f) for f in sorted(os.listdir(base / cls))]
    else:
        rows += [(split, None, f) for f in sorted(os.listdir(base))]   # sorted: thứ tự dòng không phụ thuộc hệ điều hành

recs = []
for split, cls, f in rows:
    p = video_path(ROOT, split, cls, f)
    c = cv2.VideoCapture(str(p))
    n = int(c.get(cv2.CAP_PROP_FRAME_COUNT)); fps = c.get(cv2.CAP_PROP_FPS)
    w = int(c.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(c.get(cv2.CAP_PROP_FRAME_HEIGHT))
    c.release()
    recs.append(dict(split=split, cls=cls, file=f, size_kb=os.path.getsize(p) / 1024,
                     frames=n, fps=fps, w=w, h=h, dur=n / fps if fps else None))
df = pd.DataFrame(recs)
df.to_csv(HERE / 'video_metadata.csv', encoding='utf-8-sig', index=False)
print(len(df), '->', HERE / 'video_metadata.csv')
