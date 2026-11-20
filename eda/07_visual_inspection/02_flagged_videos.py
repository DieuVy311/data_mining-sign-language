"""Bước 7.2 – Xem bằng mắt các video bị các bước trước đánh dấu (không chọn tay).
Tiêu chí chọn, đều lấy từ kết quả bước 2:
  (a) video gần như toàn khung đơn sắc (solid_ratio >= 0.5), tối đa 6 video có tỷ lệ cao nhất;
  (b) video thuộc nhóm MD5 xuất hiện ở nhiều hơn một lớp (xung đột nhãn);
  (c) một ví dụ cho mỗi dạng khung đơn sắc: chỉ đầu / chỉ cuối / đầu + cuối / có ở giữa.
Chạy:  python 02_flagged_videos.py [đường_dẫn_dataset]
Đầu vào: video_quality_flags.csv, file_hashes.csv
Đầu ra: flagged_videos_check.png, solid_frame_patterns.png, flagged_videos_info.txt
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import *

HERE = Path(__file__).resolve().parent
TRAIN = dataset_root() / 'train'
plt = plt_setup()
q = load_flags(); q = q[q.split == 'train'].copy()
q['base'] = q.file.str.replace(BASE_RE, '.mp4', regex=True)
uq = q.drop_duplicates(['cls', 'base']).sort_values(['cls', 'file'])
h = load_hashes(); h = h[h.split == 'train']

# (a) gần như rỗng
a = uq[uq.solid_ratio >= 0.5].sort_values('solid_ratio', ascending=False).head(6)
# (b) xung đột nhãn
bad = h.groupby('md5').cls.nunique(); bad = bad[bad > 1].index
b = h[h.md5.isin(bad)].sort_values(['md5', 'cls', 'file'])
picked = [(r.cls, r.file, 'gần như rỗng') for r in a.itertuples()] + \
         [(r.cls, r.file, 'xung đột nhãn') for r in b.itertuples()]

def pattern(flags):
    n = len(flags); lead = next((i for i, s in enumerate(flags) if not s), n)
    tail = next((i for i, s in enumerate(reversed(flags)) if not s), n)
    mid = sum(flags) - lead - (tail if lead < n else 0)
    return 'toàn bộ' if lead == n else 'có ở giữa' if mid > 0 else \
           'đầu + cuối' if lead and tail else 'chỉ đầu' if lead else 'chỉ cuối'

# (c) ví dụ cho từng dạng (duyệt video bị ảnh hưởng theo thứ tự lớp/tên tệp cho tới khi đủ dạng)
examples = {}
for r in uq[(uq.solid_frames > 0) & (uq.solid_ratio < 0.99)].itertuples():
    p = pattern([is_solid(f) for f in read_frames(TRAIN / r.cls / r.file, rgb=False)])
    examples.setdefault(p, (r.cls, r.file))
    if len(examples) == 4: break

# ---- hình 1: lưới 8 khung cho các video được chọn
K = 8
fig, ax = plt.subplots(len(picked), K, figsize=(K * 1.6, len(picked) * 1.75), squeeze=False)
info = []
for i, (c, f, why) in enumerate(picked):
    p = TRAIN / c / f; fr = read_frames(p)
    for j, k in enumerate(np.linspace(0, len(fr) - 1, K).astype(int)):
        ax[i, j].imshow(fr[k]); ax[i, j].set_xticks([]); ax[i, j].set_yticks([]); ax[i, j].set_title(f"f{k}", fontsize=7)
    ax[i, 0].set_ylabel(f'{c}/{f[:-4]}', fontsize=8, rotation=0, ha='right', va='center')
    solid = sum(is_solid(x) for x in fr)
    info.append(f'{c}/{f} | {why} | md5={md5_of(p)[:10]} | {p.stat().st_size // 1024} KB | {len(fr)} khung | {solid} khung đơn sắc')
plt.tight_layout(); plt.savefig(HERE / 'flagged_videos_check.png', dpi=110); plt.close()

# ---- hình 2: các dạng khung đơn sắc (mỗi hàng một dạng, cách 2 khung)
fig, ax = plt.subplots(len(examples), 15, figsize=(15, len(examples) * 1.25), squeeze=False)
for i, (pat, (c, f)) in enumerate(examples.items()):
    fr = read_frames(TRAIN / c / f)
    for j, k in enumerate(range(0, min(30, len(fr)), 2)):
        ax[i, j].imshow(fr[k]); ax[i, j].set_xticks([]); ax[i, j].set_yticks([]); ax[i, j].set_title(f"f{k}", fontsize=6)
    ax[i, 0].set_ylabel(f'{pat}\n{c}/{f[:-4]}', fontsize=7, rotation=0, ha='right', va='center')
    info.append(f'{c}/{f} | dạng "{pat}"')
plt.tight_layout(); plt.savefig(HERE / 'solid_frame_patterns.png', dpi=110); plt.close()

(HERE / 'flagged_videos_info.txt').write_text('\n'.join(info), encoding='utf-8')
print('\n'.join(info))
