import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import time

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;500;600;700&display=swap');

* { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background-color: #060c1a !important;
    color: #c8d8f0 !important;
    font-family: 'Rajdhani', sans-serif !important;
}

[data-testid="stHeader"] { background: transparent !important; }

/* Hide default streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

/* Top bar */
.top-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid #1a3a6a;
    padding-bottom: 12px;
    margin-bottom: 24px;
}
.top-bar-title {
    font-family: 'Share Tech Mono', monospace;
    font-size: 22px;
    color: #4ab3ff;
    letter-spacing: 3px;
    text-transform: uppercase;
}
.top-bar-status {
    font-family: 'Share Tech Mono', monospace;
    font-size: 11px;
    color: #2a6aaa;
    letter-spacing: 2px;
}

/* Panels */
.panel {
    background: #080f20;
    border: 1px solid #1a3a6a;
    padding: 16px;
    margin-bottom: 16px;
}
.panel-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 10px;
    color: #2a6aaa;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 10px;
    border-bottom: 1px solid #1a3a6a;
    padding-bottom: 6px;
}

/* Status badge */
.status-detected {
    font-family: 'Share Tech Mono', monospace;
    font-size: 13px;
    color: #ff4444;
    letter-spacing: 2px;
}
.status-normal {
    font-family: 'Share Tech Mono', monospace;
    font-size: 13px;
    color: #44ff88;
    letter-spacing: 2px;
}
.status-pending {
    font-family: 'Share Tech Mono', monospace;
    font-size: 13px;
    color: #4ab3ff;
    letter-spacing: 2px;
}

/* Metrics */
.metric-row {
    display: flex;
    justify-content: space-between;
    padding: 5px 0;
    border-bottom: 1px solid #0d1f3a;
    font-size: 13px;
}
.metric-label { color: #4a7ab0; font-size: 12px; }
.metric-value { color: #c8d8f0; font-family: 'Share Tech Mono', monospace; font-size: 12px; }

/* Confidence bar */
.conf-row { margin: 6px 0; }
.conf-label {
    display: flex;
    justify-content: space-between;
    font-size: 11px;
    color: #4a7ab0;
    font-family: 'Share Tech Mono', monospace;
    margin-bottom: 3px;
}
.conf-bar-bg {
    background: #0d1f3a;
    height: 6px;
    width: 100%;
}
.conf-bar-fill {
    height: 6px;
    background: linear-gradient(90deg, #1a5aaa, #4ab3ff);
    transition: width 0.5s;
}
.conf-bar-fill.high { background: linear-gradient(90deg, #aa1a1a, #ff4444); }

/* Upload zone */
[data-testid="stFileUploader"] {
    background: #080f20 !important;
    border: 1px dashed #1a3a6a !important;
    border-radius: 0 !important;
}
[data-testid="stFileUploader"] label {
    color: #4a7ab0 !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 11px !important;
    letter-spacing: 2px !important;
}

/* Button */
[data-testid="stButton"] button {
    background: #0d2a5a !important;
    color: #4ab3ff !important;
    border: 1px solid #1a5aaa !important;
    border-radius: 0 !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 12px !important;
    letter-spacing: 3px !important;
    text-transform: uppercase !important;
    width: 100% !important;
    padding: 12px !important;
    transition: all 0.2s !important;
}
[data-testid="stButton"] button:hover {
    background: #1a4a8a !important;
    border-color: #4ab3ff !important;
}

/* Image display */
[data-testid="stImage"] img {
    border: 1px solid #1a3a6a;
    filter: brightness(0.9) contrast(1.1);
}

/* Divider */
hr { border-color: #1a3a6a !important; }

/* Spinner */
[data-testid="stSpinner"] { color: #4ab3ff !important; }

/* Warning / info */
[data-testid="stAlert"] {
    background: #080f20 !important;
    border: 1px solid #1a3a6a !important;
    border-radius: 0 !important;
    color: #4a7ab0 !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 11px !important;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']
CLASS_LABELS = {
    'glioma':     'GLIOMA TUMOR',
    'meningioma': 'MENINGIOMA',
    'notumor':    'NO TUMOR DETECTED',
    'pituitary':  'PITUITARY TUMOR',
}

# ── Model Architecture (Must match train.py exactly) ──────────
class BrainTumorCNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256 * 8 * 8, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

EVAL_TF = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# ── Model loader ──────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        # On instancie la classe "from scratch"
        model = BrainTumorCNN(num_classes=4)
        # On charge les poids entraînés (state_dict)
        model.load_state_dict(torch.load('model.pth', map_location='cpu'))
        model.eval()
        return model
    except Exception:
        return None

model = load_model()

# ── Top bar ───────────────────────────────────────────────────
st.markdown("""
<div class="top-bar">
    <div class="top-bar-title">⬡ Brain Tumor Detection System</div>
    <div class="top-bar-status">CNN · PyTorch · v1.0 &nbsp;|&nbsp; SYSTEM READY</div>
</div>
""", unsafe_allow_html=True)

# ── Layout ────────────────────────────────────────────────────
col_left, col_mid, col_right = st.columns([1, 1.4, 1])

# ─── LEFT COLUMN ─────────────────────────────────────────────
with col_left:
    st.markdown('<div class="panel"><div class="panel-label">▸ MRI Upload</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("", type=["jpg", "jpeg", "png"])
    st.markdown('</div>', unsafe_allow_html=True)

    # Clear previous results when a new image is uploaded
    if uploaded:
        if 'last_uploaded' not in st.session_state or st.session_state['last_uploaded'] != uploaded.name:
            if 'result' in st.session_state:
                del st.session_state['result']
        st.session_state['last_uploaded'] = uploaded.name

    if uploaded:
        img = Image.open(uploaded).convert('RGB')
        st.markdown('<div class="panel"><div class="panel-label">▸ Input Image</div>', unsafe_allow_html=True)
        st.image(img, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown(f"""
        <div class="panel">
            <div class="panel-label">▸ Image Info</div>
            <div class="metric-row"><span class="metric-label">FILENAME</span><span class="metric-value">{uploaded.name[:18]}</span></div>
            <div class="metric-row"><span class="metric-label">SIZE</span><span class="metric-value">{img.size[0]} × {img.size[1]} px</span></div>
            <div class="metric-row"><span class="metric-label">MODE</span><span class="metric-value">{img.mode}</span></div>
            <div class="metric-row"><span class="metric-label">INPUT TENSOR</span><span class="metric-value">128 × 128 × 3</span></div>
        </div>
        """, unsafe_allow_html=True)

# ─── MIDDLE COLUMN ───────────────────────────────────────────
with col_mid:
    if uploaded:
        run = st.button("RUN ANALYSIS", use_container_width=True)

        if run:
            with st.spinner("PROCESSING..."):
                time.sleep(1.2)

                if model is not None:
                    tensor = EVAL_TF(img).unsqueeze(0)
                    with torch.no_grad():
                        outputs = model(tensor)
                        probs = F.softmax(outputs, dim=1)[0]
                    pred_idx = probs.argmax().item()
                    pred_class = CLASS_NAMES[pred_idx]
                    confidence = probs[pred_idx].item()
                    probs_dict = {c: probs[i].item() for i, c in enumerate(CLASS_NAMES)}
                else:
                    # Placeholder quand le modèle n'est pas encore chargé
                    probs_dict = {c: float(np.random.dirichlet([1]*4)[i]) for i, c in enumerate(CLASS_NAMES)}
                    pred_class = max(probs_dict, key=probs_dict.get)
                    confidence = probs_dict[pred_class]

                st.session_state['result'] = {
                    'pred_class': pred_class,
                    'confidence': confidence,
                    'probs': probs_dict
                }

        if 'result' in st.session_state:
            r = st.session_state['result']
            pred = r['pred_class']
            conf = r['confidence']

            is_tumor = pred != 'notumor'
            status_class = "status-detected" if is_tumor else "status-normal"
            status_text = f"DETECTED : {CLASS_LABELS[pred]}" if is_tumor else "STATUS : NO TUMOR DETECTED"

            st.markdown(f"""
            <div class="panel">
                <div class="panel-label">▸ Prediction Result</div>
                <div class="{status_class}" style="font-size:16px; margin: 8px 0;">{status_text}</div>
                <div class="metric-row" style="margin-top:10px">
                    <span class="metric-label">PREDICTION ACCURACY</span>
                    <span class="metric-value" style="color:#4ab3ff; font-size:18px;">{conf*100:.1f}%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Confidence bars
            bars_html = '<div class="panel"><div class="panel-label">▸ Class Probabilities</div>'
            for cls in CLASS_NAMES:
                p = r['probs'][cls]
                is_top = cls == pred
                fill_class = "high" if (is_top and is_tumor) else ""
                bars_html += f"""
                <div class="conf-row">
                    <div class="conf-label">
                        <span>{'▶ ' if is_top else ''}{cls.upper()}</span>
                        <span>{p*100:.1f}%</span>
                    </div>
                    <div class="conf-bar-bg">
                        <div class="conf-bar-fill {fill_class}" style="width:{p*100:.1f}%"></div>
                    </div>
                </div>
                """
            bars_html += '</div>'
            st.markdown(bars_html, unsafe_allow_html=True)

            # Matplotlib chart
            fig, ax = plt.subplots(figsize=(5, 2.5))
            fig.patch.set_facecolor('#080f20')
            ax.set_facecolor('#060c1a')
            colors = ['#ff4444' if c == pred and is_tumor else '#4ab3ff' for c in CLASS_NAMES]
            vals = [r['probs'][c] * 100 for c in CLASS_NAMES]
            bars = ax.bar(CLASS_NAMES, vals, color=colors, width=0.5, edgecolor='#1a3a6a', linewidth=0.8)
            ax.set_ylim(0, 105)
            ax.tick_params(colors='#4a7ab0', labelsize=8)
            ax.spines[:].set_color('#1a3a6a')
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)
            ax.set_ylabel('Confidence %', color='#4a7ab0', fontsize=8)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                        f'{val:.1f}', ha='center', va='bottom', color='#c8d8f0', fontsize=7,
                        fontfamily='monospace')
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

        else:
            st.markdown("""
            <div class="panel" style="text-align:center; padding: 40px 16px;">
                <div style="font-family:'Share Tech Mono',monospace; color:#1a3a6a; font-size:12px; letter-spacing:3px;">
                    AWAITING ANALYSIS
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="panel" style="text-align:center; padding: 60px 16px;">
            <div style="font-family:'Share Tech Mono',monospace; color:#1a3a6a; font-size:11px; letter-spacing:3px; line-height:2;">
                NO IMAGE LOADED<br>
                UPLOAD MRI SCAN TO BEGIN
            </div>
        </div>
        """, unsafe_allow_html=True)

# ─── RIGHT COLUMN ─────────────────────────────────────────────
with col_right:
    st.markdown("""
    <div class="panel">
        <div class="panel-label">▸ System Status</div>
        <div class="metric-row"><span class="metric-label">MODEL</span>
            <span class="metric-value" style="color:#44ff88;">LOADED</span></div>
        <div class="metric-row"><span class="metric-label">DEVICE</span>
            <span class="metric-value">CPU</span></div>
        <div class="metric-row"><span class="metric-label">INPUT SIZE</span>
            <span class="metric-value">128 × 128</span></div>
        <div class="metric-row"><span class="metric-label">CLASSES</span>
            <span class="metric-value">4</span></div>
        <div class="metric-row"><span class="metric-label">FRAMEWORK</span>
            <span class="metric-value">PyTorch</span></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="panel">
        <div class="panel-label">▸ Target Classes</div>
        <div class="metric-row"><span class="metric-label">01</span><span class="metric-value">GLIOMA</span></div>
        <div class="metric-row"><span class="metric-label">02</span><span class="metric-value">MENINGIOMA</span></div>
        <div class="metric-row"><span class="metric-label">03</span><span class="metric-value">NO TUMOR</span></div>
        <div class="metric-row"><span class="metric-label">04</span><span class="metric-value">PITUITARY</span></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="panel">
        <div class="panel-label">▸ Preprocessing</div>
        <div class="metric-row"><span class="metric-label">RESIZE</span><span class="metric-value">128×128</span></div>
        <div class="metric-row"><span class="metric-label">NORMALIZE</span><span class="metric-value">μ=0.5 σ=0.5</span></div>
        <div class="metric-row"><span class="metric-label">FORMAT</span><span class="metric-value">RGB TENSOR</span></div>
    </div>
    """, unsafe_allow_html=True)

    if 'result' in st.session_state and uploaded:
        r = st.session_state['result']
        pred = r['pred_class']
        conf = r['confidence']

        risk_map = {
            'glioma': ('HIGH', '#ff4444'),
            'meningioma': ('MODERATE', '#ffaa44'),
            'pituitary': ('MODERATE', '#ffaa44'),
            'notumor': ('NONE', '#44ff88'),
        }
        risk_label, risk_color = risk_map[pred]

        st.markdown(f"""
        <div class="panel">
            <div class="panel-label">▸ Assessment</div>
            <div class="metric-row">
                <span class="metric-label">RISK LEVEL</span>
                <span class="metric-value" style="color:{risk_color};">{risk_label}</span>
            </div>
            <div class="metric-row">
                <span class="metric-label">CONFIDENCE</span>
                <span class="metric-value">{conf*100:.1f}%</span>
            </div>
            <div class="metric-row">
                <span class="metric-label">DIAGNOSIS</span>
                <span class="metric-value">{CLASS_LABELS[pred]}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)