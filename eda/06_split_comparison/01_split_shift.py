"""Bước 6.1 – So sánh phân phối giữa train / public_test / private_test (dịch chuyển phân phối).
Với mỗi thuộc tính: trung vị từng tập và kiểm định Kolmogorov–Smirnov từng cặp tập (train dùng video duy nhất).
Thêm tỷ lệ video có khung đơn sắc theo tập và trùng nội dung giữa các tập (đọc từ file_hashes.csv).
Chạy:  python 01_split_shift.py
Đầu vào: video_metadata.csv, video_quality_flags.csv, file_hashes.csv
Đầu ra: split_shift_summary.txt, ecdf_by_split.png
"""
import itertools, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import *
from scipy import stats

HERE = Path(__file__).resolve().parent
m = load_meta()
q = load_flags()
h = load_hashes()
m = m.merge(q[['split', 'cls', 'file', 'solid_frames', 'motion']], on=['split', 'cls', 'file'], how='left')
m = m.merge(h[['split', 'cls', 'file', 'md5']], on=['split', 'cls', 'file'], how='left')
# train: chỉ giữ video duy nhất, để bản sao không làm lệch phân phối
parts = {'train': unique_train(m), 'public_test': m[m.split == 'public_test'], 'private_test': m[m.split == 'private_test']}
VARS = ['frames', 'fps', 'dur', 'size_kb', 'motion']

with report(HERE / 'split_shift_summary.txt'):
    print('== Trung vị theo tập (train: video duy nhất)')
    print(pd.DataFrame({sp: g[VARS].median() for sp, g in parts.items()}).round(2).T.to_string())

    print('\n== Kiểm định KS hai mẫu: thống kê D (p-value)')
    for v in VARS:
        out = []
        for a, b in itertools.combinations(parts, 2):
            D, p = stats.ks_2samp(parts[a][v].dropna(), parts[b][v].dropna())
            out.append(f'{a} vs {b}: D={D:.3f} (p={p:.2g})')
        print(f'{v:8s}', ' | '.join(out))

    print('\n== Tỷ lệ video có >=1 khung đơn sắc theo tập (%)')
    print({sp: round(float(100 * (g.solid_frames > 0).mean()), 2) for sp, g in parts.items()})

    print('\n== Trùng nội dung giữa các tập (đếm MD5 khác nhau; số video trùng có thể lớn hơn vì tập test tự trùng lặp)')
    ms = {sp: set(g.md5) for sp, g in parts.items()}
    for a, b in itertools.combinations(parts, 2):
        n_vid = int(m[(m.split == b) & m.md5.isin(ms[a])].shape[0])
        print(f'{a} ∩ {b}:', len(ms[a] & ms[b]), 'MD5 chung |', n_vid, f'video của {b} trùng với {a}')

plt = plt_setup()
fig, axs = plt.subplots(1, 3, figsize=(13, 3.6))
for ax, v, lab in zip(axs, ['fps', 'size_kb', 'dur'], ['FPS', 'File size (KB)', 'Duration (s)']):
    for (sp, g), c in zip(parts.items(), SPLIT_COLORS):
        x = np.sort(g[v].values); ax.plot(x, np.arange(1, len(x) + 1) / len(x), color=c, label=sp)
    ax.set_xlabel(lab); ax.set_ylabel('ECDF')
    if v == 'dur': ax.set_xlim(0, 6)
axs[0].legend(fontsize=7)
fig.suptitle('Empirical CDF per split (train: unique videos)')
fig.tight_layout(); fig.savefig(HERE / 'ecdf_by_split.png'); plt.close()
