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

# Robust import supporting both flat root and src/ package layouts
try:
    import extractors
    import features
    import model
except ImportError:
    from src import extractors, features, model

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
    initial_sidebar_state="expanded"
)

# Custom Styling (Hostinger Console & Modern Aesthetic Tech Theme)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Ambient Tech Glow Background */
    .stApp {
        background: 
            radial-gradient(circle at 20% 8%, rgba(99, 102, 241, 0.16) 0%, transparent 45%),
            radial-gradient(circle at 80% 12%, rgba(168, 85, 247, 0.14) 0%, transparent 45%),
            radial-gradient(circle at 50% 90%, rgba(6, 182, 212, 0.10) 0%, transparent 50%),
            #090d16;
        color: #f1f5f9;
    }

    /* Hero Header */
    .spark-hero {
        position: relative;
        text-align: center;
        padding: 2.2rem 1rem 1.5rem 1rem;
        border-radius: 20px;
        background: linear-gradient(180deg, rgba(30, 41, 59, 0.45) 0%, rgba(15, 23, 42, 0.25) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(16px);
        box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.5);
        margin-bottom: 2rem;
        overflow: hidden;
    }
    
    .spark-hero::before {
        content: "";
        position: absolute;
        top: 0; left: 20%; right: 20%;
        height: 1px;
        background: linear-gradient(90deg, transparent, #818cf8, #c084fc, #38bdf8, transparent);
    }

    .spark-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.35rem 1rem;
        border-radius: 9999px;
        background: rgba(99, 102, 241, 0.12);
        border: 1px solid rgba(129, 140, 248, 0.3);
        color: #c7d2fe;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 0.9rem;
        box-shadow: 0 0 20px rgba(99, 102, 241, 0.2);
    }
    .pulse-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #38bdf8;
        box-shadow: 0 0 10px #38bdf8;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.4; transform: scale(0.85); }
    }

    .spark-title {
        font-size: 3.4rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        background: linear-gradient(135deg, #ffffff 0%, #c7d2fe 35%, #a855f7 70%, #38bdf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        line-height: 1.1;
        text-shadow: 0 10px 30px rgba(129, 140, 248, 0.25);
    }

    .spark-acronym {
        font-size: 1.15rem;
        font-weight: 600;
        color: #38bdf8;
        margin-top: 0.5rem;
        letter-spacing: 0.02em;
    }

    .spark-desc {
        color: #94a3b8;
        font-size: 1.02rem;
        max-width: 780px;
        margin: 0.8rem auto 0 auto;
        line-height: 1.6;
    }

    /* Tech Feature Pills */
    .tech-pills {
        display: flex;
        justify-content: center;
        flex-wrap: wrap;
        gap: 0.6rem;
        margin-top: 1.2rem;
    }
    .pill {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.28rem 0.75rem;
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        font-size: 0.78rem;
        color: #cbd5e1;
    }

    /* Result Metric Cards */
    .metric-container {
        display: flex;
        gap: 1.25rem;
        margin-bottom: 1.5rem;
    }
    .card-ai {
        flex: 1;
        background: linear-gradient(160deg, rgba(244, 63, 94, 0.12) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid rgba(244, 63, 94, 0.35);
        border-radius: 16px;
        padding: 1.6rem 1.25rem;
        text-align: center;
        box-shadow: 0 10px 25px rgba(244, 63, 94, 0.1);
        backdrop-filter: blur(12px);
    }
    .card-human {
        flex: 1;
        background: linear-gradient(160deg, rgba(16, 185, 129, 0.12) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid rgba(16, 185, 129, 0.35);
        border-radius: 16px;
        padding: 1.6rem 1.25rem;
        text-align: center;
        box-shadow: 0 10px 25px rgba(16, 185, 129, 0.1);
        backdrop-filter: blur(12px);
    }
    .pct-val {
        font-size: 3.2rem;
        font-weight: 800;
        line-height: 1;
        letter-spacing: -0.02em;
    }
    .pct-ai { color: #fb7185; text-shadow: 0 0 25px rgba(244, 63, 94, 0.35); }
    .pct-human { color: #34d399; text-shadow: 0 0 25px rgba(16, 185, 129, 0.35); }
    .pct-label {
        font-size: 0.92rem;
        font-weight: 700;
        color: #f1f5f9;
        margin-top: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }

    /* Verdict Banner */
    .verdict-box {
        border-radius: 14px;
        padding: 1.2rem 1.4rem;
        margin: 1.4rem 0;
        display: flex;
        align-items: center;
        justify-content: space-between;
        backdrop-filter: blur(12px);
    }
    .verdict-ai {
        background: rgba(244, 63, 94, 0.12);
        border: 1px solid rgba(244, 63, 94, 0.4);
        border-left: 6px solid #f43f5e;
        color: #ffe4e6;
    }
    .verdict-human {
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.4);
        border-left: 6px solid #10b981;
        color: #d1fae5;
    }
    .verdict-mixed {
        background: rgba(245, 158, 11, 0.12);
        border: 1px solid rgba(245, 158, 11, 0.4);
        border-left: 6px solid #f59e0b;
        color: #fef3c7;
    }

    /* Sentence Heatmap Highlights */
    .sentence-container {
        line-height: 2.0;
        font-size: 1.05rem;
        padding: 1.6rem;
        background: rgba(15, 23, 42, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        margin-top: 1rem;
        color: #e2e8f0;
        box-shadow: inset 0 2px 8px rgba(0,0,0,0.4);
    }
    .highlight-ai {
        background-color: rgba(244, 63, 94, 0.22);
        color: #fecdd3;
        padding: 3px 6px;
        border-radius: 5px;
        border-bottom: 2px solid #f43f5e;
        transition: all 0.2s ease;
    }
    .highlight-human {
        background-color: rgba(16, 185, 129, 0.22);
        color: #a7f3d0;
        padding: 3px 6px;
        border-radius: 5px;
        border-bottom: 2px solid #10b981;
        transition: all 0.2s ease;
    }
    .highlight-mixed {
        background-color: rgba(245, 158, 11, 0.22);
        color: #fde68a;
        padding: 3px 6px;
        border-radius: 5px;
        border-bottom: 2px solid #f59e0b;
        transition: all 0.2s ease;
    }

    /* Tabs & Buttons Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(15, 23, 42, 0.6);
        padding: 6px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.06);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #94a3b8;
        font-weight: 600;
        padding: 8px 16px;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(99, 102, 241, 0.25) !important;
        color: #ffffff !important;
        border: 1px solid rgba(129, 140, 248, 0.4) !important;
    }

    /* Stat Cards */
    div[data-testid="stMetric"] {
        background: rgba(15, 23, 42, 0.65);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 12px;
        padding: 12px 16px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
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

# Sidebar / Top Toolbar for Controls
with st.sidebar:
    st.markdown("### ⚙️ Detection Settings")
    sensitivity_choice = st.selectbox(
        "Sensitivity Preset:",
        [
            "Balanced (Recommended)",
            "Strict AI (Catch subtle/rephrased AI)",
            "Human-Safe (Minimize false positives)"
        ],
        index=0,
        help="Adjust how aggressively the model flags AI vs Human writing."
    )
    if "Strict" in sensitivity_choice:
        sens_mode = "strict_ai"
        st.info("🎯 **Strict Mode**: Highly sensitive to subtle, polished AI text. Best for academic integrity checks.")
    elif "Human-Safe" in sensitivity_choice:
        sens_mode = "human_safe"
        st.info("🛡️ **Human-Safe Mode**: Gives benefit of doubt to formal human essays and reports.")
    else:
        sens_mode = "balanced"
        st.caption("⚖️ **Balanced**: Optimal balance between false positives and false negatives.")
        
    st.divider()
    st.markdown("### 🔬 How It Works")
    st.caption(
        "- **Burstiness**: Humans vary sentence length wildly; AI is uniform.\n"
        "- **Contractions**: Humans say *don't, it's*; AI says *do not, it is*.\n"
        "- **Perspective**: 1st-person voice indicates authentic human experience.\n"
        "- **Punctuation**: Humans use dashes, semicolons, and parentheses."
    )

# SPARK Hero Section
st.markdown("""
<div class="spark-hero">
    <div class="spark-badge">
        <span class="pulse-dot"></span>
        <span>Next-Gen Text Intelligence &bull; Engine v2.5</span>
    </div>
    <h1 class="spark-title">⚡ SPARK</h1>
    <div class="spark-acronym">Smart Platform for AI Recognition and Knowledge</div>
    <p class="spark-desc">
        A humanized, transparent AI detector that analyzes syntactic cadence, conversational burstiness, 
        and subword transitions to reliably distinguish authentic human writing from LLMs like ChatGPT, Claude, and Gemini.
    </p>
    <div class="tech-pills">
        <span class="pill">⚡ Subword Morphological Engine</span>
        <span class="pill">🧠 Sentence Burstiness Radar</span>
        <span class="pill">💬 Contraction &amp; Voice Fingerprinting</span>
        <span class="pill">🛡️ False-Positive Protection</span>
    </div>
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
    "📝 Paste Text", 
    "📁 Upload Document (PDF / Word / TXT)", 
    "🔗 Article / Blog Link"
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
        label="Enter or paste text (English, 50–2000+ words):",
        value=st.session_state.get("raw_text_input", ""),
        height=220,
        placeholder="Paste your essay, article, or writing sample here..."
    )
    
    words = len(raw_text.split()) if raw_text else 0
    chars = len(raw_text) if raw_text else 0
    st.caption(f"Word count: **{words}** | Character count: **{chars}** (Recommended: 80+ words)")
    
    if st.button("🚀 Analyze Pasted Text", type="primary", use_container_width=True):
        if words < 15:
            st.error("Please provide at least 15 words for meaningful detection.")
        else:
            text_to_analyze = raw_text
            input_source_info = "Pasted Text"

# --- TAB 2: UPLOAD DOCUMENT ---
with tab_file:
    st.markdown("#### Upload Document")
    st.markdown("Upload any **PDF (.pdf)**, **Word Document (.docx)**, or **Text file (.txt, .md)**.")
    
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
                
            if st.button("🚀 Analyze Uploaded Document", type="primary", use_container_width=True):
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
    st.markdown("Provide the URL of any online blog post, news article, or publication.")
    
    input_url = st.text_input(
        "Article URL:",
        placeholder="https://example.com/article-slug"
    )
    
    if st.button("🌐 Fetch & Analyze Web Article", type="primary", use_container_width=True):
        if not input_url.strip():
            st.error("Please enter a valid URL.")
        else:
            with st.spinner("Fetching web page and extracting clean article body..."):
                try:
                    url_res = extract_from_url(input_url.strip())
                    st.success(f"Extracted **{url_res['word_count']} words** from: *{url_res['title']}*")
                    with st.expander("🌐 Article Preview"):
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
    st.markdown(f"### 📊 Detection Results & Breakdown")
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
    st.markdown("#### 🔬 Stylometric & Linguistic Diagnostics")
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
    st.markdown("#### 🔍 Sentence-by-Sentence AI Heatmap")
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
    with st.expander("📋 Detailed Sentence Probability Table"):
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
    st.markdown("#### 📥 Export Detection Report")
    report_json = json.dumps(results, indent=2)
    st.download_button(
        label="Download Full JSON Analysis Report",
        data=report_json,
        file_name="ai_detection_report.json",
        mime="application/json"
    )
