"""Ứng dụng demo nhận diện ngôn ngữ ký hiệu tiếng Việt từ video.

Chạy từ thư mục gốc của repo:  streamlit run demo/app.py
"""
from __future__ import annotations

import hashlib
import io
import sys
import tempfile
import time
import unicodedata
from datetime import datetime
from pathlib import Path

import altair as alt
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

DEMO = Path(__file__).resolve().parent
ROOT = DEMO.parent
ASSETS = DEMO / 'assets'
sys.path[:0] = [str(DEMO), str(ROOT / 'processing')]

import preprocess as pp  
from model import load_model, predict 

DATASET = ROOT / 'data' / 'dataset'
TOP_K = 5

st.set_page_config(page_title='Nhận diện ngôn ngữ ký hiệu', page_icon='🤟', layout='wide')
st.markdown(f'<style>{(ASSETS / "style.css").read_text(encoding="utf-8")}</style>', unsafe_allow_html=True)


# ----------------------------------------------- dữ liệu --------------------------------------------------------
def nfc(s: str) -> str:
    return unicodedata.normalize('NFC', str(s))


@st.cache_resource(show_spinner='Đang nạp mô hình CNN-LSTM...')
def get_model():
    return load_model()


@st.cache_data
def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data
def dataset_dirs() -> dict:
    train = DATASET / 'train'
    return pp.class_dirs(train) if train.exists() else {}


def train_path(cls: str, file: str) -> Path | None:
    d = dataset_dirs().get(nfc(cls))
    p = DATASET / 'train' / d / file if d else None
    return p if p is not None and p.exists() else None


@st.cache_data
def class_table() -> pd.DataFrame:
    """Mỗi lớp một dòng: số video huấn luyện, chỉ số trên fold 0, các lớp hay nhầm, video ví dụ."""
    man = read_csv(ROOT / 'models' / 'features' / 'manifest_clean.csv').assign(cls=lambda d: d.cls.map(nfc))
    rep = read_csv(ROOT / 'models' / 'cnn_lstm' / 'output' / 'fold0_report_B.csv').assign(cls=lambda d: d['class'].map(nfc))
    pred = read_csv(ASSETS / 'fold0_pred.csv')
    wrong = pred[pred.cls != pred.pred]
    confused = {}
    for a, b in list(zip(wrong.cls, wrong.pred)) + list(zip(wrong.pred, wrong.cls)):
        confused.setdefault(a, {}).setdefault(b, 0)
        confused[a][b] += 1
    example = man.sort_values(['fold', 'file']).groupby('cls').first()
    df = (rep[['label_idx', 'cls', 'precision', 'recall', 'f1-score', 'support']]
          .rename(columns={'f1-score': 'f1', 'support': 'n_fold0'})
          .merge(man.groupby('cls').size().rename('n_train'), on='cls'))
    df['confused'] = df.cls.map(lambda c: sorted(confused.get(c, {}).items(), key=lambda t: -t[1]))
    df['example'] = df.cls.map(lambda c: example.loc[c, 'file'])
    return df.set_index('cls')


@st.cache_data
def first_valid_frame(path: str) -> np.ndarray:
    frames = pp.read_video(path)
    valid = pp.valid_indices(frames) or list(range(len(frames)))
    return frames[valid[len(valid) // 2]]


@st.cache_data
def to_gif(path: str) -> bytes:
    """Ảnh động xem trước (bỏ khung đơn sắc); trình duyệt không phát được mã hoá MPEG-4 Part 2 của bộ dữ liệu."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    frames = pp.read_video(path)
    keep = pp.valid_indices(frames) or list(range(len(frames)))
    imgs = [Image.fromarray(frames[i]) for i in keep]
    buf = io.BytesIO()
    imgs[0].save(buf, format='GIF', save_all=True, append_images=imgs[1:], loop=0,
                 duration=int(1000 / (fps if 1 <= fps <= 60 else 10)))
    return buf.getvalue()


@st.cache_data(show_spinner=False)
def run(path: str) -> dict:
    t0 = time.time()
    frames = pp.read_video(path)
    valid = pp.valid_indices(frames)
    n_solid = len(frames) - len(valid)
    warn = None
    if not valid:
        valid = list(range(len(frames)))
        warn = 'Mọi khung hình đều đơn sắc, kết quả không đáng tin cậy.'
    elif len(valid) < pp.TARGET_FRAMES:
        warn = f'Video chỉ có {len(valid)} khung hợp lệ (ít hơn {pp.TARGET_FRAMES}), một số khung được lặp lại.'
    idx = pp.sample_indices(valid)
    clip = frames[idx]
    model, labels, _ = get_model()
    prob, alpha = predict(model, pp.normalize(clip))
    return {'prob': prob, 'alpha': alpha, 'clip': clip, 'idx': idx, 'labels': labels, 'warn': warn,
            'n_frames': len(frames), 'n_solid': n_solid, 'seconds': time.time() - t0}


# ----------------------------------------------------------- thành phần -----------------------------------------------
def show_chart(chart):
    try:
        st.altair_chart(chart, width='stretch')
    except TypeError:  
        st.altair_chart(chart, use_container_width=True)


def f1_color(f1: float) -> str:
    return '#16a34a' if f1 >= 0.9 else '#f59e0b' if f1 >= 0.6 else '#ef4444'


def confidence_badge(p: float) -> str:
    if p >= 0.6:
        return '<span class="badge b-green">Độ tin cậy cao</span>'
    if p >= 0.3:
        return '<span class="badge b-amber">Độ tin cậy trung bình</span>'
    return '<span class="badge b-red">Độ tin cậy thấp</span>'


def confused_text(items, k=3) -> str:
    return ', '.join(f'<b>{c}</b> ({n})' for c, n in items[:k]) if items else 'không có trên fold 0'


def hero():
    st.markdown("""
<div class="hero">
  <h1>Nhận diện ngôn ngữ ký hiệu tiếng Việt</h1>
  <p>Tải lên một video ký hiệu, mô hình CNN-LSTM sẽ cho biết đó là từ nào trong 100 từ vựng và những khung hình
  nào quyết định kết quả.</p>
  <div class="kpis">
    <div class="kpi"><div class="v">100</div><div class="l">từ vựng ký hiệu</div></div>
    <div class="kpi"><div class="v">0,687</div><div class="l">Macro-F1 trên tập private test</div></div>
    <div class="kpi"><div class="v">0,859</div><div class="l">Macro-F1 trên tập kiểm định</div></div>
    <div class="kpi"><div class="v">~1 giây</div><div class="l">để nhận diện một video</div></div>
  </div>
</div>""", unsafe_allow_html=True)


def pipeline_html(res) -> str:
    steps = [('Đọc video', f"{res['n_frames']} khung hình"),
             ('Lọc khung', f"bỏ {res['n_solid']} khung đơn sắc"),
             ('Lấy mẫu', '16 khung cách đều'),
             ('EfficientNet-B0', '16 vector 1.280 chiều'),
             ('BiLSTM + attention', 'gộp theo thời gian'),
             ('Phân lớp', '100 từ, softmax')]
    cells = ''.join(f'<div class="step"><div class="n">{i}</div><div class="t">{t}</div><div class="d">{d}</div></div>'
                    for i, (t, d) in enumerate(steps, 1))
    return f'<div class="steps">{cells}</div>'


def topk_html(labels, prob, order) -> str:
    rows = ''.join(
        f'<div class="bar-row{" first" if r == 0 else ""}"><div>{labels[i]}</div>'
        f'<div class="bar-bg"><div class="bar-fg" style="width:{max(prob[i] * 100, 1):.1f}%"></div></div>'
        f'<div class="pct">{prob[i]:.1%}</div></div>' for r, i in enumerate(order))
    return rows


def frame_grid(clip, alpha, idx, highlight=3):
    top = set(np.argsort(alpha)[::-1][:highlight])
    for row in range(0, len(clip), 8):
        cols = st.columns(8)
        for j, col in enumerate(cols):
            k = row + j
            img = clip[k].copy()
            if k in top:
                cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1), (37, 99, 235), 10)
            col.image(img, caption=f'khung {idx[k]} · {alpha[k]:.2f}', width='stretch')


def word_card(cls: str, info: pd.Series) -> str:
    return f"""
<div class="facts">
  <div class="fact"><div class="v">{info.f1:.2f}</div><div class="l">F1 của từ “{cls}” trên tập kiểm định</div></div>
  <div class="fact"><div class="v">{info.n_train}</div><div class="l">video huấn luyện của từ này</div></div>
  <div class="fact"><div class="v">{info.precision:.0%}</div><div class="l">lần đoán “{cls}” là đúng (precision)</div></div>
</div>
<p class="muted" style="margin:10px 0 0">Hay bị nhầm với: {confused_text(info.confused)}.
Số trong ngoặc là số video nhầm theo cả hai chiều trên tập kiểm định.</p>"""


def show_result(res, name: str, true_label: str | None = None):
    prob, labels = res['prob'], res['labels']
    order = np.argsort(prob)[::-1][:TOP_K]
    pred, p1 = labels[order[0]], float(prob[order[0]])

    with st.container(border=True):
        left, right = st.columns([1, 1.35], gap='large')
        with left:
            st.image(to_gif(res['path']), width='stretch')
            hidden = f" (đã ẩn {res['n_solid']} khung đơn sắc)" if res['n_solid'] else ''
            st.caption(f"{name} · {res['n_frames']} khung{hidden} · xử lý trong {res['seconds']:.2f} giây")
        with right:
            badges = confidence_badge(p1)
            if true_label is not None:
                ok = nfc(true_label) == pred
                badges += (f'<span class="badge b-green">Đúng nhãn “{nfc(true_label)}”</span>' if ok else
                           f'<span class="badge b-red">Sai, nhãn thật là “{nfc(true_label)}”</span>')
            st.markdown(f'<div class="pred-label">Kết quả nhận diện</div><div class="pred-word">{pred}</div>'
                        f'{badges}<div style="height:14px"></div>'
                        f'<div class="pred-label" style="margin-bottom:4px">{TOP_K} từ có xác suất cao nhất</div>'
                        f'{topk_html(labels, prob, order)}', unsafe_allow_html=True)
            if res['warn']:
                st.warning(res['warn'])

    st.markdown(f'<div class="card"><h4>Quy trình xử lý video này</h4>{pipeline_html(res)}</div>',
                unsafe_allow_html=True)

    table = class_table()
    if pred in table.index:
        st.markdown(f'<div class="card"><h4>Về từ “{pred}”</h4>{word_card(pred, table.loc[pred])}</div>',
                    unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown('#### Mô hình dựa vào khung hình nào?')
        st.caption('16 khung đưa vào mô hình sau khi bỏ khung đơn sắc và lấy đều. Dưới mỗi khung là chỉ số khung '
                   'trong video và trọng số attention (tổng bằng 1); ba khung viền xanh là ba khung quan trọng nhất.')
        frame_grid(res['clip'], res['alpha'], res['idx'])
        att = pd.DataFrame({'Khung': [str(i) for i in res['idx']], 'Thứ tự': range(len(res['alpha'])),
                            'Attention': res['alpha']})
        show_chart(alt.Chart(att).mark_bar(color='#2563eb', cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                        .encode(x=alt.X('Khung:N', sort=alt.SortField('Thứ tự'), title='Chỉ số khung trong video',
                                        axis=alt.Axis(labelAngle=0)),
                                y=alt.Y('Attention:Q', title='Trọng số attention'),
                                tooltip=['Khung', alt.Tooltip('Attention:Q', format='.3f')])
                        .properties(height=170))
    return pred, p1


def add_history(name, pred, p1, true_label):
    hist = st.session_state.setdefault('history', [])
    if hist and hist[-1]['Video'] == name:
        return
    hist.append({'Thời gian': datetime.now().strftime('%H:%M:%S'), 'Video': name, 'Dự đoán': pred,
                 'Xác suất': round(p1, 4), 'Nhãn thật': nfc(true_label) if true_label else '',
                 'Kết quả': ('Đúng' if nfc(true_label) == pred else 'Sai') if true_label else ''})


def show_history():
    hist = st.session_state.get('history', [])
    if not hist:
        return
    with st.container(border=True):
        c1, c2 = st.columns([4, 1])
        c1.markdown(f'#### Lịch sử nhận diện ({len(hist)} video)')
        df = pd.DataFrame(hist[::-1])
        c2.download_button('Tải về CSV', df.to_csv(index=False).encode('utf-8-sig'), 'ket_qua_nhan_dien.csv',
                           'text/csv', width='stretch')
        st.dataframe(df, hide_index=True, width='stretch',
                     column_config={'Xác suất': st.column_config.ProgressColumn('Xác suất', min_value=0, max_value=1,
                                                                                format='percent')})



# -------------------------------------------------------------------- trang ----------------------------------------------------------------
def page_recognize():
    hero()
    samples = read_csv(ASSETS / 'samples.csv')
    samples['path'] = [train_path(*r.split('/')[1:]) for r in samples.rel_path]
    samples = samples.dropna(subset=['path']).reset_index(drop=True)

    source = st.segmented_control('Nguồn video', ['Video mẫu', 'Tải video lên'], default='Video mẫu',
                                  label_visibility='collapsed') or 'Video mẫu'
    path = name = true_label = None

    if source == 'Video mẫu':
        if samples.empty:
            st.info('Không tìm thấy dữ liệu trong `data/dataset/`. Hãy tải dữ liệu cuộc thi về, hoặc chọn '
                    '"Tải video lên".')
        else:
            st.caption('Video lấy từ tập kiểm định (fold 0), mô hình không dùng các video này để học.')
            sel = st.session_state.setdefault('sample', 0)
            cols = st.columns(len(samples))
            for i, r in samples.iterrows():
                with cols[i].container(border=True):
                    st.image(first_valid_frame(str(r.path)), width='stretch')
                    st.markdown(f'**{r.label}**<div class="muted" style="min-height:3.6em">{r.note}</div>',
                                unsafe_allow_html=True)
                    if st.button('Đang xem' if i == sel else 'Chọn', key=f's{i}', width='stretch',
                                 type='primary' if i == sel else 'secondary'):
                        st.session_state.sample = i
                        st.rerun()
            r = samples.iloc[sel]
            path, name, true_label = str(r.path), f'{r.label}/{Path(r.path).name}', r.label
    else:
        up = st.file_uploader('Chọn video ký hiệu (một từ, quay chính diện)', type=['mp4', 'avi', 'mov', 'mkv'])
        if up is not None:
            data = up.getvalue()
            tmp = Path(tempfile.gettempdir()) / f'vsl_{hashlib.md5(data).hexdigest()}{Path(up.name).suffix}'
            if not tmp.exists():
                tmp.write_bytes(data)
            path, name = str(tmp), up.name

    if path is not None:
        try:
            with st.spinner('Đang nhận diện...'):
                res = dict(run(path), path=path)
        except ValueError as e:
            st.error(f'Không đọc được video: {e}')
        else:
            pred, p1 = show_result(res, name, true_label)
            add_history(name, pred, p1, true_label)
    show_history()


def page_dictionary():
    st.markdown('<div class="page-title">Từ điển 100 từ</div><div class="page-sub">Các từ mô hình nhận diện được, '
                'số video huấn luyện và F1 của CNN-LSTM B trên tập kiểm định (mỗi từ chỉ có 1–10 video kiểm định nên '
                'F1 từng từ chỉ để tham khảo).</div>', unsafe_allow_html=True)
    table = class_table()

    c1, c2, c3 = st.columns([2, 1.4, 1])
    q = c1.text_input('Tìm từ', placeholder='Ví dụ: rẽ, bệnh, ăn...')
    level = c2.segmented_control('Mức F1', ['Tất cả', '≥ 0,9', '0,6–0,9', '< 0,6'], default='Tất cả') or 'Tất cả'
    sort = c3.selectbox('Sắp xếp', ['Theo tên', 'F1 giảm dần', 'F1 tăng dần', 'Số video huấn luyện'])

    df = table.reset_index()
    if q:
        df = df[df.cls.str.lower().str.contains(nfc(q).lower())]
    if level == '≥ 0,9':
        df = df[df.f1 >= 0.9]
    elif level == '0,6–0,9':
        df = df[(df.f1 >= 0.6) & (df.f1 < 0.9)]
    elif level == '< 0,6':
        df = df[df.f1 < 0.6]
    key = {'Theo tên': ('label_idx', True), 'F1 giảm dần': ('f1', False), 'F1 tăng dần': ('f1', True),
           'Số video huấn luyện': ('n_train', False)}[sort]
    df = df.sort_values(key[0], ascending=key[1])

    if not df.empty:
        with st.container(border=True):
            pick = st.selectbox(f'Xem chi tiết một từ ({len(df)} từ)', df.cls.tolist())
            info = table.loc[pick]
            left, right = st.columns([1, 2.2], gap='large')
            p = train_path(pick, info.example)
            with left:
                if p is not None:
                    st.image(to_gif(str(p)), width='stretch')
                    st.caption(f'Video ví dụ: {info.example}')
                else:
                    st.info('Cần thư mục `data/dataset/` để xem video ví dụ.')
            with right:
                st.markdown(f'<div class="pred-word" style="font-size:2rem">{pick}</div>{word_card(pick, info)}',
                            unsafe_allow_html=True)

    cards = ''.join(
        f'<div class="word"><div class="w">{r.cls}</div><div class="m">{r.n_train} video huấn luyện · F1 {r.f1:.2f}</div>'
        f'<div class="f1bar"><span style="width:{max(r.f1, .02) * 100:.0f}%;background:{f1_color(r.f1)}"></span></div></div>'
        for r in df.itertuples())
    st.markdown(f'<div class="words">{cards}</div>' if cards else '<p class="muted">Không có từ nào phù hợp.</p>',
                unsafe_allow_html=True)


def page_compare():
    st.markdown('<div class="page-title">So sánh mô hình</div><div class="page-sub">Bốn thuật toán cổ điển trên hai bộ '
                'đặc trưng EfficientNet-B0 và mô hình CNN-LSTM, đánh giá bằng Macro-F1 trên tập kiểm định (fold 0, '
                '555 video) và tập private test (2.859 video, do AIChallenge chấm).</div>', unsafe_allow_html=True)
    st.markdown("""
<div class="insights">
  <div class="insight"><div class="v">+0,29 – 0,34</div><div class="t">Đặc trưng quan trọng hơn thuật toán</div>
    <div class="d">Tinh chỉnh CNN làm Macro-F1 private của mọi thuật toán tăng mạnh, nhiều hơn chênh lệch giữa các
    thuật toán trên cùng một bộ đặc trưng.</div></div>
  <div class="insight"><div class="v">Tuyến tính &gt; kNN &gt; RF</div><div class="t">Thứ hạng ổn định</div>
    <div class="d">Logistic Regression và SVM tuyến tính tốt nhất trên cả hai tập; kNN và Random Forest giảm mạnh
    hơn khi gặp video mới.</div></div>
  <div class="insight"><div class="v">0,687</div><div class="t">CNN-LSTM B tốt nhất</div>
    <div class="d">Logistic Regression trên đặc trưng tinh chỉnh đạt 0,662, chỉ kém 0,025 nhưng huấn luyện chưa tới
    1 giây trên CPU.</div></div>
</div>""", unsafe_allow_html=True)

    scores = read_csv(ASSETS / 'scores.csv')
    scores['Tên'] = scores['Mô hình'] + ' (' + scores['Đặc trưng'].str.lower() + ')'
    with st.container(border=True):
        st.markdown('#### Macro-F1 trên hai tập')
        long = scores.melt(id_vars=['Tên'], value_vars=['Macro-F1 fold 0', 'Macro-F1 private'], var_name='Tập',
                           value_name='Macro-F1')
        show_chart(
            alt.Chart(long).mark_bar(cornerRadiusTopRight=3, cornerRadiusBottomRight=3)
            .encode(y=alt.Y('Tên:N', sort=scores['Tên'].tolist(), title=None, axis=alt.Axis(labelLimit=300)),
                    x=alt.X('Macro-F1:Q', scale=alt.Scale(domain=[0, 1])), yOffset='Tập:N',
                    color=alt.Color('Tập:N', scale=alt.Scale(range=['#94a3b8', '#2563eb']),
                                    legend=alt.Legend(orient='top', title=None)),
                    tooltip=['Tên', 'Tập', alt.Tooltip('Macro-F1:Q', format='.3f')])
            .properties(height=470))
        st.dataframe(scores.drop(columns=['Tên']), hide_index=True, width='stretch',
                     height=35 * (len(scores) + 1) + 3,
                     column_config={c: st.column_config.ProgressColumn(c, min_value=0, max_value=1, format='%.3f')
                                    for c in ['Macro-F1 fold 0', 'Macro-F1 private']})

    fold0 = ROOT / 'models' / 'classical' / 'output' / 'fold0_metrics.csv'
    if fold0.exists():
        with st.expander('Các chỉ số khác trên tập kiểm định (Precision, Recall, Accuracy, Top-5, ROC-AUC, thời gian)'):
            df = read_csv(fold0).drop(columns=['Tham số'], errors='ignore')
            st.dataframe(df.style.format(precision=3, na_rep='–'), hide_index=True, width='stretch')


get_model()
nav = st.navigation([st.Page(page_recognize, title='Nhận diện', icon=':material/videocam:', url_path='nhan-dien',
                             default=True),
                     st.Page(page_dictionary, title='Từ điển 100 từ', icon=':material/menu_book:', url_path='tu-dien'),
                     st.Page(page_compare, title='So sánh mô hình', icon=':material/bar_chart:', url_path='so-sanh')],
                    position='top')
nav.run()
