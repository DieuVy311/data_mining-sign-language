"""Bước 5.1 – Quan hệ giữa các thuộc tính (tương quan) và giữa thuộc tính với nhãn lớp.
Dùng video duy nhất của train. Tương quan: Pearson và Spearman. Thuộc tính – nhãn: Kruskal–Wallis và eta^2
(eta^2 = phần phương sai của thuộc tính do nhãn lớp giải thích).
Chạy:  python 01_correlation_and_label_association.py
Đầu vào: 01_overview/video_metadata.csv, 02_data_quality/video_quality_flags.csv
Đầu ra: multivariate_summary.txt, corr_heatmap.png, fps_by_class.png, size_vs_solid.png
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import *
from scipy import stats

HERE = Path(__file__).resolve().parent
m = load_meta()
q = load_flags()
m = m.merge(q[['split', 'cls', 'file', 'solid_ratio', 'motion']], on=['split', 'cls', 'file'], how='left')
tr = m[m.split == 'train']
u = unique_train(m)
cols = ['frames', 'fps', 'size_kb', 'solid_ratio', 'motion']

with report(HERE / 'multivariate_summary.txt'):
    print('== Tương quan Pearson (train, video duy nhất)')
    print(u[cols].corr().round(2).to_string())
    print('\n== Tương quan Spearman (train, video duy nhất)')
    print(u[cols].corr(method='spearman').round(2).to_string())

    print('\n== Dung lượng theo có/không khung đơn sắc (train)')
    print('median size_kb có / không khung đơn sắc:',
          round(tr[tr.solid_ratio > 0].size_kb.median(), 1), '/', round(tr[tr.solid_ratio == 0].size_kb.median(), 1))

    print('\n== Quan hệ thuộc tính – nhãn lớp (Kruskal–Wallis, eta^2)')
    for v in ['fps', 'size_kb', 'frames']:
        grp = [x[v].values for _, x in u.groupby('cls')]
        H, p = stats.kruskal(*grp); gm = u[v].mean()
        ssb = sum(len(x) * (x.mean() - gm) ** 2 for x in grp); sst = ((u[v] - gm) ** 2).sum()
        print(f'{v}: eta2={ssb / sst:.3f}  KW H={H:.0f}  p={p:.2g}')
    g = u.groupby('cls').fps.median()
    print('trung vị FPS theo lớp:', g.describe().round(2).to_dict())
    print('thấp nhất:', g.nsmallest(3).round(1).to_dict(), '| cao nhất:', g.nlargest(3).round(1).to_dict())

plt = plt_setup()
# ---- Hình 1: heatmap tương quan
lab = ['Số khung', 'FPS', 'Dung lượng', 'Tỷ lệ khung\nđơn sắc', 'Chuyển động\ntrung bình']
c = u[cols].corr().values
fig, ax = plt.subplots(figsize=(5.4, 4.4)); im = ax.imshow(c, cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_xticks(range(5)); ax.set_yticks(range(5))
ax.set_xticklabels(lab, fontsize=8, rotation=30, ha='right'); ax.set_yticklabels(lab, fontsize=8)
for i in range(5):
    for j in range(5):
        ax.text(j, i, f"{c[i, j]:.2f}", ha='center', va='center', fontsize=9, color='white' if abs(c[i, j]) > .6 else 'black')
plt.colorbar(im, fraction=.046); ax.set_title('Tương quan Pearson (train, video duy nhất)', fontsize=10)
plt.tight_layout(); plt.savefig(HERE / 'corr_heatmap.png', dpi=160); plt.close()

# ---- Hình 2: FPS theo lớp
g = u.groupby('cls').fps.agg(med='median', q1=lambda x: x.quantile(.25), q3=lambda x: x.quantile(.75)).sort_values('med').reset_index()
fig, ax = plt.subplots(figsize=(9, 3.6)); x = np.arange(len(g))
ax.fill_between(x, g.q1, g.q3, alpha=.3, color='#3b82f6', label='Khoảng tứ phân vị (Q1–Q3)')
ax.plot(x, g.med, color='#1d4ed8', label='Trung vị FPS của lớp')
ax.axhline(u.fps.median(), color='r', ls='--', lw=1, label=f'Trung vị toàn tập = {u.fps.median():.1f}')
for i in [0, 1, len(g) - 2, len(g) - 1]:
    ax.annotate(g.cls[i], (i, g.med[i]), fontsize=7, xytext=(3, 6 if i < 50 else -10), textcoords='offset points')
ax.set_xlabel('Lớp (sắp xếp theo trung vị FPS)'); ax.set_ylabel('FPS'); ax.legend(fontsize=7, loc='upper left')
ax.set_title('FPS theo lớp (train, video duy nhất)', fontsize=10)
plt.tight_layout(); plt.savefig(HERE / 'fps_by_class.png', dpi=160); plt.close()

# ---- Hình 3: dung lượng theo tỷ lệ khung đơn sắc
fig, ax = plt.subplots(figsize=(5.4, 3.8)); ax.scatter(u.solid_ratio * 100, u.size_kb, s=5, alpha=.35, color='#2563eb')
ax.set_xlabel('Tỷ lệ khung đơn sắc trong video (%)'); ax.set_ylabel('Dung lượng tệp (KB)')
ax.set_title('Dung lượng và tỷ lệ khung đơn sắc (train, duy nhất)', fontsize=9)
plt.tight_layout(); plt.savefig(HERE / 'size_vs_solid.png', dpi=160); plt.close()
