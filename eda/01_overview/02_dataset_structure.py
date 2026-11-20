"""Bước 1.2 – Cấu trúc dữ liệu: các tập, định dạng tệp, độ phân giải, tệp nhãn.
Trả lời: dữ liệu gồm những gì, chia thế nào, có đồng nhất về định dạng không, nhãn có khớp tệp ánh xạ không.
Chạy:  python 02_dataset_structure.py [đường_dẫn_dataset]
Đầu vào: video_metadata.csv, <dataset>/label_mapping.pkl (nếu có)
Đầu ra: structure_summary.txt, split_sizes.png
"""
import pickle, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import *

HERE = Path(__file__).resolve().parent
ROOT = dataset_root()
df = load_meta()

with report(HERE / 'structure_summary.txt'):
    print('== Quy mô các tập')
    for sp in SPLITS:
        g = df[df.split == sp]
        print(f'{sp}: {len(g)} video | có nhãn: {g.cls.notna().any()} | '
              f'dung lượng tổng {g.size_kb.sum() / 1024:.1f} MB')
    print('tổng số video:', len(df))

    print('\n== Định dạng tệp (đuôi)')
    print(df.file.str.extract(r'(\.[^.]+)$')[0].value_counts().to_dict())

    print('\n== Độ phân giải')
    for sp in SPLITS:
        g = df[df.split == sp]
        print(sp, (g.w.astype(str) + 'x' + g.h.astype(str)).value_counts().head(8).to_dict())

    print('\n== Số khung hình')
    for sp in SPLITS:
        g = df[df.split == sp]
        print('==', sp, len(g))
        print(g.frames.describe().round(2).to_dict())
        print('frames<16:', (g.frames < 16).sum(), ' frames==0:', (g.frames == 0).sum())
        print('frames vc', g.frames.value_counts().sort_index().to_dict())

    print('\n== Tệp ánh xạ nhãn')
    pkl = ROOT / 'label_mapping.pkl'
    if pkl.exists():
        import unicodedata
        nfc = lambda s: unicodedata.normalize('NFC', s)
        mp = pickle.load(open(pkl, 'rb'))
        classes = set(df[df.split == 'train'].cls.dropna())
        print('số lớp trong label_mapping:', len(mp), '| số lớp trong train:', len(classes))
        print('so khớp nguyên văn  : thiếu trong mapping', len(classes - set(mp)),
              '| thừa trong mapping', len(set(mp) - classes))
        print('so khớp sau chuẩn hoá NFC: thiếu trong mapping', sorted({nfc(c) for c in classes} - {nfc(k) for k in mp}),
              '| thừa trong mapping', sorted({nfc(k) for k in mp} - {nfc(c) for c in classes}))
        if classes - set(mp):
            print('LƯU Ý: khoá trong label_mapping.pkl và tên thư mục lớp khác dạng chuẩn hoá Unicode '
                  '(mapping NFC: %s, thư mục NFC: %s) -> cần normalize("NFC") trước khi tra nhãn.'
                  % (all(nfc(k) == k for k in mp), all(nfc(c) == c for c in classes)))
        print('chỉ số nhãn:', min(mp.values()), '-', max(mp.values()), '| liên tục không trùng:',
              sorted(mp.values()) == list(range(len(mp))))
    else:
        print('không tìm thấy label_mapping.pkl')

plt = plt_setup()
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(SPLITS, [len(df[df.split == s]) for s in SPLITS], color=SPLIT_COLORS)
ax.set_title('Videos per split')
fig.tight_layout(); fig.savefig(HERE / 'split_sizes.png'); plt.close()
