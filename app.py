"""
Streamlit Web Application for AI-Generated Text Detector.
Provides:
- Three Input Tabs: Text Copy/Paste, Document Upload (PDF/Word/TXT), Article/Blog URLs
- Percentage Breakdown: AI % vs Human %
- Confidence Score & Verdict
- Stylometric Diagnostics (Burstiness, Lexical Diversity, Readability, Markers)
- Visual Sentence-Level Heatmap
- Report Download
"""

import os
import sys
import types
import json
import streamlit as st

# Setup search paths for local run, Docker, and Streamlit Cloud (/mount/src/...)
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")
for p in [ROOT_DIR, SRC_DIR]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

# Check for scikit-learn / joblib / numpy and guide user if requirements.txt wasn't installed
missing_core = []
for pkg_name, imp_name in [("scikit-learn", "sklearn"), ("joblib", "joblib"), ("numpy", "numpy")]:
    try:
        __import__(imp_name)
    except ImportError:
        missing_core.append(pkg_name)

if missing_core:
    st.error(
        f"⚠️ **Missing Core Python Libraries**: `{', '.join(missing_core)}`\n\n"
        "Streamlit Cloud did not install your `requirements.txt` dependencies.\n\n"
        "**To resolve this:**\n"
        "1. Ensure `requirements.txt` is uploaded to the **root** folder of your GitHub repository.\n"
        "2. In Streamlit Cloud, click **Manage app** (bottom-right) ➔ **Reboot app** to re-run installation."
    )
    st.stop()

# Import modules directly (ROOT_DIR and SRC_DIR are both in sys.path)
try:
    import extractors
    import features
    import model
except Exception as err:
    st.error(f"⚠️ **Error loading SPARK modules**: `{err}`")
    st.stop()

# Alias sys.modules for 'src.*' in case any component or unpickler looks for 'src'
if 'src' not in sys.modules:
    src_mod = types.ModuleType('src')
    src_mod.extractors = extractors
    src_mod.features = features
    src_mod.model = model
    sys.modules['src'] = src_mod
sys.modules['src.extractors'] = extractors
sys.modules['src.features'] = features
sys.modules['src.model'] = model

extract_from_file = extractors.extract_from_file
extract_from_url = extractors.extract_from_url
clean_text = extractors.clean_text
AIDetector = model.AIDetector
MODEL_PATH = model.MODEL_PATH

# Page Configuration
st.set_page_config(
    page_title="SPARK - Smart Platform for AI Recognition and Knowledge",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom Styling — Warm, human-crafted aesthetic with gradient animations
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }

    /* ============================================================
       ANIMATED GRADIENT BACKGROUND — warm aurora mesh
       ============================================================ */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #1a1333 25%, #24243e 50%, #0f0c29 100%);
        background-size: 400% 400%;
        animation: auroraShift 18s ease infinite;
        color: #eee6f5;
        min-height: 100vh;
    }

    @keyframes auroraShift {
        0%   { background-position: 0% 50%;   }
        25%  { background-position: 50% 25%;  }
        50%  { background-position: 100% 50%; }
        75%  { background-position: 50% 75%;  }
        100% { background-position: 0% 50%;   }
    }

    /* Floating aurora blobs behind content */
    .stApp::before,
    .stApp::after {
        content: "";
        position: fixed;
        border-radius: 50%;
        pointer-events: none;
        z-index: 0;
        filter: blur(100px);
    }
    .stApp::before {
        width: 600px; height: 600px;
        top: -120px; left: -80px;
        background: radial-gradient(circle, rgba(139, 92, 246, 0.18), transparent 70%);
        animation: floatBlob1 14s ease-in-out infinite alternate;
    }
    .stApp::after {
        width: 500px; height: 500px;
        bottom: -80px; right: -60px;
        background: radial-gradient(circle, rgba(249, 115, 22, 0.14), transparent 70%);
        animation: floatBlob2 16s ease-in-out infinite alternate;
    }

    @keyframes floatBlob1 {
        0%   { transform: translate(0, 0) scale(1);     }
        100% { transform: translate(80px, 50px) scale(1.15); }
    }
    @keyframes floatBlob2 {
        0%   { transform: translate(0, 0) scale(1);     }
        100% { transform: translate(-60px, -40px) scale(1.1); }
    }

    /* ============================================================
       HERO — Warm gradient card with entrance animation
       ============================================================ */
    .spark-hero {
        position: relative;
        text-align: center;
        padding: 2.4rem 1.5rem 1.8rem;
        border-radius: 24px;
        background: linear-gradient(160deg,
            rgba(139, 92, 246, 0.12) 0%,
            rgba(236, 72, 153, 0.08) 40%,
            rgba(249, 115, 22, 0.06) 70%,
            rgba(15, 12, 41, 0.6) 100%);
        border: 1px solid rgba(255, 255, 255, 0.07);
        backdrop-filter: blur(20px);
        box-shadow:
            0 25px 50px -12px rgba(0, 0, 0, 0.5),
            inset 0 1px 0 rgba(255, 255, 255, 0.05);
        margin-bottom: 2rem;
        overflow: hidden;
        animation: heroSlideIn 0.8s cubic-bezier(0.16, 1, 0.3, 1) both;
    }

    @keyframes heroSlideIn {
        from { opacity: 0; transform: translateY(30px) scale(0.97); }
        to   { opacity: 1; transform: translateY(0) scale(1);       }
    }

    /* Animated gradient top border */
    .spark-hero::before {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg,
            #8b5cf6, #ec4899, #f97316, #eab308, #10b981, #06b6d4, #8b5cf6);
        background-size: 300% 100%;
        animation: borderGlow 6s linear infinite;
    }

    @keyframes borderGlow {
        0%   { background-position: 0% 50%;   }
        100% { background-position: 300% 50%; }
    }

    .spark-title {
        font-family: 'Space Grotesk', 'Inter', sans-serif;
        font-size: 3.6rem;
        font-weight: 800;
        letter-spacing: -0.04em;
        background: linear-gradient(135deg, #f9fafb 0%, #d8b4fe 30%, #f472b6 55%, #fb923c 80%, #fbbf24 100%);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        line-height: 1.1;
        animation: shimmerText 4s ease infinite;
    }

    @keyframes shimmerText {
        0%   { background-position: 0% center;   }
        50%  { background-position: 100% center; }
        100% { background-position: 0% center;   }
    }

    .spark-acronym {
        font-size: 1.1rem;
        font-weight: 500;
        letter-spacing: 0.03em;
        margin-top: 0.6rem;
        background: linear-gradient(90deg, #c4b5fd, #f9a8d4, #fdba74);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: fadeInUp 1s 0.3s cubic-bezier(0.16, 1, 0.3, 1) both;
    }

    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(12px); }
        to   { opacity: 1; transform: translateY(0);    }
    }

    /* ============================================================
       RESULT CARDS — Warm glowing glassmorphism
       ============================================================ */
    .card-ai {
        flex: 1;
        background: linear-gradient(165deg,
            rgba(244, 63, 94, 0.14) 0%,
            rgba(190, 24, 93, 0.08) 50%,
            rgba(15, 12, 41, 0.7) 100%);
        border: 1px solid rgba(244, 63, 94, 0.25);
        border-radius: 20px;
        padding: 1.8rem 1.4rem;
        text-align: center;
        box-shadow:
            0 12px 30px rgba(244, 63, 94, 0.1),
            inset 0 1px 0 rgba(255, 255, 255, 0.04);
        backdrop-filter: blur(16px);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        animation: cardPop 0.6s 0.2s cubic-bezier(0.16, 1, 0.3, 1) both;
    }
    .card-ai:hover {
        transform: translateY(-4px);
        box-shadow: 0 20px 40px rgba(244, 63, 94, 0.18);
    }

    .card-human {
        flex: 1;
        background: linear-gradient(165deg,
            rgba(16, 185, 129, 0.14) 0%,
            rgba(5, 150, 105, 0.08) 50%,
            rgba(15, 12, 41, 0.7) 100%);
        border: 1px solid rgba(16, 185, 129, 0.25);
        border-radius: 20px;
        padding: 1.8rem 1.4rem;
        text-align: center;
        box-shadow:
            0 12px 30px rgba(16, 185, 129, 0.1),
            inset 0 1px 0 rgba(255, 255, 255, 0.04);
        backdrop-filter: blur(16px);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        animation: cardPop 0.6s 0.4s cubic-bezier(0.16, 1, 0.3, 1) both;
    }
    .card-human:hover {
        transform: translateY(-4px);
        box-shadow: 0 20px 40px rgba(16, 185, 129, 0.18);
    }

    @keyframes cardPop {
        from { opacity: 0; transform: translateY(20px) scale(0.95); }
        to   { opacity: 1; transform: translateY(0) scale(1);       }
    }

    .pct-val {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 3.4rem;
        font-weight: 800;
        line-height: 1;
        letter-spacing: -0.03em;
    }
    .pct-ai {
        background: linear-gradient(135deg, #fb7185, #e11d48);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        filter: drop-shadow(0 4px 12px rgba(244, 63, 94, 0.4));
    }
    .pct-human {
        background: linear-gradient(135deg, #34d399, #059669);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        filter: drop-shadow(0 4px 12px rgba(16, 185, 129, 0.4));
    }
    .pct-label {
        font-size: 0.88rem;
        font-weight: 700;
        color: rgba(255, 255, 255, 0.85);
        margin-top: 0.6rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    /* ============================================================
       VERDICT BANNERS — Warm organic tones
       ============================================================ */
    .verdict-box {
        border-radius: 16px;
        padding: 1.3rem 1.5rem;
        margin: 1.5rem 0;
        display: flex;
        align-items: center;
        justify-content: space-between;
        backdrop-filter: blur(16px);
        transition: all 0.3s ease;
        animation: fadeInUp 0.5s 0.5s cubic-bezier(0.16, 1, 0.3, 1) both;
    }
    .verdict-ai {
        background: linear-gradient(135deg, rgba(244, 63, 94, 0.15), rgba(190, 24, 93, 0.08));
        border: 1px solid rgba(244, 63, 94, 0.3);
        border-left: 5px solid #f43f5e;
        color: #ffe4e6;
    }
    .verdict-human {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15), rgba(5, 150, 105, 0.08));
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-left: 5px solid #10b981;
        color: #d1fae5;
    }
    .verdict-mixed {
        background: linear-gradient(135deg, rgba(249, 115, 22, 0.15), rgba(217, 119, 6, 0.08));
        border: 1px solid rgba(249, 115, 22, 0.3);
        border-left: 5px solid #f97316;
        color: #fed7aa;
    }

    /* ============================================================
       SENTENCE HEATMAP — readable, warm-toned highlights
       ============================================================ */
    .sentence-container {
        line-height: 2.1;
        font-size: 1.02rem;
        padding: 1.6rem;
        background: linear-gradient(180deg, rgba(15, 12, 41, 0.75) 0%, rgba(24, 20, 50, 0.6) 100%);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 16px;
        margin-top: 1rem;
        color: #e8e0f0;
        box-shadow: inset 0 2px 12px rgba(0, 0, 0, 0.3);
    }
    .highlight-ai {
        background: linear-gradient(135deg, rgba(244, 63, 94, 0.25), rgba(190, 24, 93, 0.15));
        color: #fecdd3;
        padding: 3px 7px;
        border-radius: 6px;
        border-bottom: 2px solid #f43f5e;
        transition: all 0.25s ease;
    }
    .highlight-ai:hover { background: rgba(244, 63, 94, 0.35); }

    .highlight-human {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.25), rgba(5, 150, 105, 0.15));
        color: #a7f3d0;
        padding: 3px 7px;
        border-radius: 6px;
        border-bottom: 2px solid #10b981;
        transition: all 0.25s ease;
    }
    .highlight-human:hover { background: rgba(16, 185, 129, 0.35); }

    .highlight-mixed {
        background: linear-gradient(135deg, rgba(249, 115, 22, 0.25), rgba(217, 119, 6, 0.15));
        color: #fed7aa;
        padding: 3px 7px;
        border-radius: 6px;
        border-bottom: 2px solid #f97316;
        transition: all 0.25s ease;
    }
    .highlight-mixed:hover { background: rgba(249, 115, 22, 0.35); }

    /* ============================================================
       TABS — Warm frosted glass
       ============================================================ */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background: linear-gradient(135deg, rgba(30, 25, 55, 0.7), rgba(15, 12, 41, 0.5));
        padding: 6px;
        border-radius: 14px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(12px);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        color: #a09cb5;
        font-weight: 600;
        padding: 10px 18px;
        transition: all 0.25s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #e8e0f0;
        background: rgba(139, 92, 246, 0.1);
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(139, 92, 246, 0.3), rgba(236, 72, 153, 0.15)) !important;
        color: #ffffff !important;
        border: 1px solid rgba(167, 139, 250, 0.3) !important;
        box-shadow: 0 4px 15px rgba(139, 92, 246, 0.2);
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(160deg, rgba(30, 25, 55, 0.7), rgba(15, 12, 41, 0.5));
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 14px;
        padding: 14px 18px;
        box-shadow: 0 4px 18px rgba(0, 0, 0, 0.2);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(139, 92, 246, 0.12);
    }

    /* Button styling */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #8b5cf6 0%, #ec4899 50%, #f97316 100%) !important;
        background-size: 200% auto !important;
        border: none !important;
        color: white !important;
        font-weight: 700 !important;
        border-radius: 12px !important;
        padding: 0.7rem 1.5rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 20px rgba(139, 92, 246, 0.35) !important;
    }
    .stButton > button[kind="primary"]:hover {
        background-position: right center !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 30px rgba(236, 72, 153, 0.4) !important;
    }

    /* File uploader */
    section[data-testid="stFileUploader"] {
        background: linear-gradient(160deg, rgba(30, 25, 55, 0.5), rgba(15, 12, 41, 0.3));
        border: 2px dashed rgba(139, 92, 246, 0.25);
        border-radius: 16px;
        padding: 1rem;
        transition: border-color 0.3s ease;
    }
    section[data-testid="stFileUploader"]:hover {
        border-color: rgba(236, 72, 153, 0.4);
    }

    /* Text input fields */
    .stTextArea textarea, .stTextInput input {
        background: rgba(15, 12, 41, 0.6) !important;
        border: 1.5px solid rgba(139, 92, 246, 0.2) !important;
        border-radius: 12px !important;
        color: #e8e0f0 !important;
        transition: all 0.25s ease !important;
    }
    .stTextArea textarea:focus, .stTextInput input:focus {
        border-color: rgba(236, 72, 153, 0.5) !important;
        box-shadow: 0 0 0 3px rgba(139, 92, 246, 0.15) !important;
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        background: rgba(30, 25, 55, 0.5) !important;
        border-radius: 12px !important;
    }

    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #f43f5e 0%, #f97316 40%, #eab308 70%, #10b981 100%) !important;
        border-radius: 10px;
        height: 8px;
    }

    /* Divider */
    hr {
        border-color: rgba(139, 92, 246, 0.15) !important;
    }

    /* Download button */
    .stDownloadButton > button {
        background: linear-gradient(135deg, rgba(30, 25, 55, 0.8), rgba(50, 40, 80, 0.6)) !important;
        border: 1px solid rgba(139, 92, 246, 0.3) !important;
        color: #c4b5fd !important;
        border-radius: 12px !important;
        transition: all 0.25s ease !important;
    }
    .stDownloadButton > button:hover {
        border-color: rgba(236, 72, 153, 0.5) !important;
        background: linear-gradient(135deg, rgba(50, 40, 80, 0.9), rgba(70, 55, 100, 0.7)) !important;
        transform: translateY(-2px) !important;
    }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-track { background: rgba(15, 12, 41, 0.5); }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #8b5cf6, #ec4899);
        border-radius: 10px;
    }

    /* Hide default streamlit footer & hamburger */
    #MainMenu, footer { visibility: hidden; }
    header[data-testid="stHeader"] { background: transparent; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_detector() -> AIDetector:
    """Load and cache the detection model."""
    detector = AIDetector(model_path=MODEL_PATH)
    if not detector.is_trained:
        csv_candidates = [
            os.path.join(ROOT_DIR, "data", "dataset.csv"),
            os.path.join(ROOT_DIR, "dataset.csv")
        ]
        csv_file = next((p for p in csv_candidates if os.path.exists(p)), csv_candidates[0])
        if not os.path.exists(csv_file):
            try:
                try:
                    from data.generate_dataset import save_dataset
                except ModuleNotFoundError:
                    from generate_dataset import save_dataset
                save_dataset(csv_file)
            except Exception:
                pass
        try:
            from train import train_and_evaluate
            train_and_evaluate()
        except Exception:
            pass
        detector = AIDetector(model_path=MODEL_PATH)
    return detector


detector = get_detector()

# Default sensitivity mode
sens_mode = "balanced"

# SPARK Hero Section
st.markdown("""
<div class="spark-hero">
    <h1 class="spark-title">⚡ SPARK</h1>
    <div class="spark-acronym">Smart Platform for AI Recognition and Knowledge</div>
</div>
""", unsafe_allow_html=True)

# Sample texts for quick demonstration
SAMPLE_TEXTS = {
    "AI (ChatGPT / GPT-4)": (
        "Artificial intelligence has emerged as a transformative paradigm across modern computing landscapes. "
        "By leveraging advanced deep neural networks and sophisticated machine learning algorithms, organizations can analyze vast "
        "troves of unstructured data with unprecedented efficiency. Furthermore, generative language models are revolutionizing natural "
        "language processing tasks, enabling automated content synthesis and intelligent conversational agents. However, it is important to remember "
        "that these systems also introduce substantial ethical considerations, including algorithmic bias, data privacy vulnerabilities, and explainability bottlenecks. "
        "Therefore, establishing comprehensive governance frameworks plays a crucial role in ensuring that technological innovation aligns with societal values. "
        "In conclusion, the ongoing integration of artificial intelligence will continue to reshape our technological horizon."
    ),
    "Human (Personal Essay)": (
        "Sitting in the drafty archives on a Tuesday morning, I kept wondering whether Spinoza ever foresaw the strange turns "
        "his substance monism would take in modern physics. The ink on the 1677 edition is faded, almost ghostly, but the geometric "
        "rigor still leaps off the vellum. It's easy to dismiss his God-or-Nature equation as mere pantheism, yet that misses the radical "
        "political bite of his Tractatus. I walked out into the biting drizzle around noon, grabbed a stale pretzel from the street vendor near "
        "the library steps, and watched pigeons fight over mustard packets. Philosophy rarely keeps your shoes dry, but it ruined my appetite in the best possible way."
    ),
    "Human (Technical Blog)": (
        "Spent three agonizing hours tracking down a memory leak in our WebSocket gateway yesterday. Turns out somebody registered a custom "
        "event listener inside a connection loop and completely forgot to detach it when clients dropped connection prematurely. In local tests "
        "with a dozen mock connections, everything looked squeaky clean. But when our staging cluster got hammered by our morning synthetic load test, "
        "the RSS footprint crept upward at roughly 80 megabytes per minute until Linux's OOM killer stepped in with extreme prejudice. "
        "Always verify teardown hooks in your mock harnesses, and never trust a garbage collector to clean up after your sloppy event emitters."
    )
}

# Input Section - 3 Tabs
tab_text, tab_file, tab_url = st.tabs([
    "Paste Text", 
    "Upload Document", 
    "Paste Link"
])

text_to_analyze = ""
input_source_info = ""

# --- TAB 1: PASTE TEXT ---
with tab_text:
    st.markdown("#### Analyze Raw Text")
    col_sample1, col_sample2, col_sample3 = st.columns([1, 1, 1])
    
    selected_sample = None
    with col_sample1:
        if st.button("Load AI Sample (ChatGPT)"):
            st.session_state["raw_text_input"] = SAMPLE_TEXTS["AI (ChatGPT / GPT-4)"]
    with col_sample2:
        if st.button("Load Human Essay Sample"):
            st.session_state["raw_text_input"] = SAMPLE_TEXTS["Human (Personal Essay)"]
    with col_sample3:
        if st.button("Load Human Tech Blog"):
            st.session_state["raw_text_input"] = SAMPLE_TEXTS["Human (Technical Blog)"]

    raw_text = st.text_area(
        label="Write or paste text:",
        value=st.session_state.get("raw_text_input", ""),
        height=220,
        placeholder="Paste your essay, article, or writing sample here..."
    )
    
    words = len(raw_text.split()) if raw_text else 0
    chars = len(raw_text) if raw_text else 0
    st.caption(f"Word count: **{words}** | Character count: **{chars}** (Recommended: 80+ words)")
    
    if st.button("Analyze", type="primary", use_container_width=True):
        if words < 15:
            st.error("Please provide at least 15 words for meaningful detection.")
        else:
            text_to_analyze = raw_text
            input_source_info = "Pasted Text"

# --- TAB 2: UPLOAD DOCUMENT ---
with tab_file:
    st.markdown("#### Upload Document")
    st.markdown("Upload any **File**")
    
    uploaded_file = st.file_uploader(
        "Choose a document file", 
        type=["pdf", "docx", "txt", "md", "rtf", "csv"],
        help="Upload files up to 20MB. Encrypted or image-only scanned PDFs are not supported."
    )
    
    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        file_name = uploaded_file.name
        
        try:
            extracted = extract_from_file(file_bytes, file_name)
            st.success(f"Successfully extracted {extracted['word_count']} words from **{file_name}** ({extracted['file_type']}).")
            
            with st.expander("📄 Document Preview (Click to expand)"):
                st.text_area("Extracted Content", extracted["raw_text"], height=200, disabled=True)
                
            if st.button("Analyze", type="primary", use_container_width=True):
                if extracted["word_count"] < 15:
                    st.error("Extracted document has fewer than 15 words. Please upload a document with more text.")
                else:
                    text_to_analyze = extracted["raw_text"]
                    input_source_info = f"File: {file_name}"
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

# --- TAB 3: ARTICLE / BLOG URL ---
with tab_url:
    st.markdown("#### Extract & Analyze Web Link")
    st.markdown("Provide the URL")
    
    input_url = st.text_input(
        "**URL:**",
        placeholder="https://example.com/article-slug"
    )
    
    if st.button("Fetch & Analyze", type="primary", use_container_width=True):
        if not input_url.strip():
            st.error("Please enter a valid URL.")
        else:
            with st.spinner("Fetching web page and extracting clean article body..."):
                try:
                    url_res = extract_from_url(input_url.strip())
                    st.success(f"Extracted **{url_res['word_count']} words** from: *{url_res['title']}*")
                    with st.expander("Article Preview"):
                        st.write(f"**Title**: {url_res['title']}")
                        st.text_area("Extracted Content", url_res["raw_text"], height=160, disabled=True)
                        
                    if url_res["word_count"] < 15:
                        st.warning("The extracted text from this URL is too short. It may be behind a paywall or heavy JavaScript.")
                    else:
                        text_to_analyze = url_res["raw_text"]
                        input_source_info = f"URL: {url_res['title']} ({input_url})"
                except Exception as e:
                    st.error(f"Failed to fetch content from URL: {str(e)}")


# --- RESULTS RENDERING ---
if text_to_analyze:
    st.divider()
    with st.spinner("Analyzing linguistic patterns, syntactic cadence, and statistical embeddings..."):
        try:
            results = detector.analyze(text_to_analyze, sensitivity=sens_mode)
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")
            st.stop()

    ai_pct = results["ai_percentage"]
    human_pct = results["human_percentage"]
    verdict = results["verdict"]
    confidence = results["confidence"]
    stats = results["stats"]

    # Results Header
    st.markdown(f"### Detection Results")
    if input_source_info:
        st.caption(f"Source: **{input_source_info}** | {stats['total_words']} words | {stats['total_sentences']} sentences")

    # Percentage Cards
    col_ai, col_human = st.columns(2)
    with col_ai:
        st.markdown(f"""
        <div class="card-ai">
            <div class="pct-val pct-ai">{ai_pct}%</div>
            <div class="pct-label">AI-Generated Content</div>
            <div style="font-size:0.85rem; color:#881337; margin-top:0.4rem;">
                Predictable syntactic structure & typical LLM discourse
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_human:
        st.markdown(f"""
        <div class="card-human">
            <div class="pct-val pct-human">{human_pct}%</div>
            <div class="pct-label">Human-Authored Writing</div>
            <div style="font-size:0.85rem; color:#064E3B; margin-top:0.4rem;">
                Natural burstiness & idiosyncratic vocabulary cadence
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Progress bar visualizer
    st.progress(ai_pct / 100.0)

    # Verdict Box
    box_class = "verdict-ai" if ai_pct >= 65 else ("verdict-human" if human_pct >= 65 else "verdict-mixed")
    icon = "🤖" if ai_pct >= 65 else ("✍️" if human_pct >= 65 else "⚖️")
    st.markdown(f"""
    <div class="verdict-box {box_class}">
        <div>
            <div style="font-size:1.25rem; font-weight:700;">{icon} Final Verdict: {verdict}</div>
            <div style="font-size:0.92rem; margin-top:0.25rem;">{results['summary_description']}</div>
        </div>
        <div style="text-align:right;">
            <span style="background:rgba(0,0,0,0.08); padding:5px 12px; border-radius:20px; font-weight:600; font-size:0.88rem;">
                {confidence}
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Stylometric Metrics Grid
    st.markdown("#### Stylometric & Linguistic Diagnostics")
    m1, m2, m3 = st.columns(3)
    with m1:
        cv = stats["burstiness_cv"]
        b_label = "High Variation (Human-like)" if cv > 0.48 else ("Uniform (AI-like)" if cv < 0.32 else "Moderate")
        st.metric(
            label="Burstiness (Cadence Variance)",
            value=f"{cv:.2f} CV",
            delta=b_label,
            delta_color="normal" if cv > 0.48 else "inverse"
        )
    with m2:
        contr = stats.get("contractions_per_100", 0.0)
        c_label = "Frequent (Human-like)" if contr >= 1.0 else ("None/Sparse (Formal/AI)" if contr == 0 else "Low")
        st.metric(
            label="Contraction Frequency",
            value=f"{contr:.1f} / 100w",
            delta=c_label,
            delta_color="normal" if contr >= 1.0 else "inverse"
        )
    with m3:
        p1 = stats.get("first_person_pct", 0.0)
        p_label = "Personal Voice (Human)" if p1 >= 2.0 else ("Impersonal (Formal/AI)" if p1 == 0 else "Low")
        st.metric(
            label="1st-Person Perspective",
            value=f"{p1:.1f}%",
            delta=p_label,
            delta_color="normal" if p1 >= 2.0 else "off"
        )

    m4, m5, m6 = st.columns(3)
    with m4:
        punct = stats.get("punctuation_diversity", 0)
        st.metric(
            label="Punctuation Diversity",
            value=f"{punct} Marks",
            delta="Varied (Dashes/Parens)" if punct >= 3 else "Standard (Periods/Commas)"
        )
    with m5:
        ttr = stats["lexical_diversity"]
        st.metric(
            label="Lexical Diversity (TTR)",
            value=f"{ttr:.2f}",
            delta="Rich Vocabulary" if ttr > 0.65 else ("Standard" if ttr > 0.48 else "Repetitive")
        )
    with m6:
        markers = results["linguistic_metrics"]["llm_markers_found"]
        st.metric(
            label="LLM Signature Clichés",
            value=f"{len(markers)} Found",
            delta="Tropes Detected" if len(markers) > 0 else "None Detected",
            delta_color="inverse" if len(markers) > 0 else "normal"
        )

    if markers:
        st.caption(f"Detected formulaic markers: {', '.join([f'`{m}`' for m in markers])}")

    # Sentence-Level Heatmap
    st.markdown("#### Sentence-by-Sentence Heatmap")
    st.caption("Sentences are color-coded based on individual model predictability:")
    st.markdown("""
    <div style="margin-bottom:0.75rem;">
        <span class="highlight-ai">Likely AI (&gt;65%)</span> &nbsp;
        <span class="highlight-mixed">Mixed / Uncertain (38% - 65%)</span> &nbsp;
        <span class="highlight-human">Likely Human (&lt;38%)</span>
    </div>
    """, unsafe_allow_html=True)

    highlighted_html_parts = []
    for s in results["sentences"]:
        score = s["ai_score"]
        tag = s["css_tag"]
        pct = s["ai_percentage"]
        # HTML badge highlight
        if tag == "ai-high":
            css = "highlight-ai"
        elif tag == "human-high":
            css = "highlight-human"
        else:
            css = "highlight-mixed"
            
        tooltip = f"AI Probability: {pct}%"
        highlighted_html_parts.append(
            f'<span class="{css}" title="{tooltip}">{s["text"]}</span>'
        )

    full_heatmap_html = (
        f'<div class="sentence-container">{" ".join(highlighted_html_parts)}</div>'
    )
    st.markdown(full_heatmap_html, unsafe_allow_html=True)

    # Sentence Details Table in Expander
    with st.expander("Detailed Sentence Probability Table"):
        table_rows = []
        for s in results["sentences"]:
            table_rows.append({
                "#": s["index"],
                "Verdict": s["verdict"],
                "AI %": f"{s['ai_percentage']}%",
                "Human %": f"{s['human_percentage']}%",
                "Sentence Text": s["text"]
            })
        st.dataframe(table_rows, use_container_width=True)

    # Downloadable Report
    st.markdown("#### Export Detection Report")
    report_json = json.dumps(results, indent=2)
    st.download_button(
        label="Download Full JSON Analysis Report",
        data=report_json,
        file_name="ai_detection_report.json",
        mime="application/json"
    )
