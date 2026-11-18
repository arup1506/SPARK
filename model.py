"""
Upgraded AI-Generated Text Detection Model & Scoring Engine.
Features:
- Dual-granularity TF-IDF (Function words + Subword character n-grams)
- Dense Stylometric Multi-Signal Analyzer (Burstiness, Contractions, Pronouns, Punctuation, Clichés)
- Configurable Sensitivity Thresholds ('balanced', 'strict_ai', 'human_safe')
- Advanced Sentence-Level Heatmap Analyzer
- Topic-Agnostic Calibration to prevent overfitting
"""

import os
import re
import joblib
import numpy as np
from typing import Dict, Any, List, Tuple
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

import sys
_curr_dir = os.path.dirname(os.path.abspath(__file__))
_src_dir = os.path.join(_curr_dir, "src")
for _p in [_curr_dir, _src_dir]:
    if os.path.exists(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from features import (
        extract_all_features, split_sentences, LLM_MARKERS,
        CONTRACTIONS, FIRST_PERSON_PRONOUNS, THIRD_PERSON_IMPERSONAL
    )
except ImportError:
    from src.features import (
        extract_all_features, split_sentences, LLM_MARKERS,
        CONTRACTIONS, FIRST_PERSON_PRONOUNS, THIRD_PERSON_IMPERSONAL
    )

# Flexible model path resolution
_curr = os.path.dirname(os.path.abspath(__file__))
_candidates = [
    os.path.join(_curr, "models", "ai_detector.joblib"),
    os.path.join(_curr, "..", "models", "ai_detector.joblib"),
    os.path.join(_curr, "ai_detector.joblib"),
    os.path.join(os.getcwd(), "models", "ai_detector.joblib"),
    os.path.join(os.getcwd(), "ai_detector.joblib")
]
MODEL_PATH = next((p for p in _candidates if os.path.exists(p)), _candidates[0])


class AIDetector:
    """Production-grade, topic-agnostic AI vs Human Detection Engine."""

    def __init__(self, model_path: str = MODEL_PATH):
        self.model_path = model_path
        self.pipeline = None
        self.is_trained = False
        self.load_or_initialize()

    def build_pipeline(self) -> Pipeline:
        """
        Builds a topic-agnostic text classification pipeline.
        Uses function-word patterns and subword morphology to avoid memorizing specific nouns.
        """
        union = FeatureUnion([
            # Word n-grams with stopwords PRESERVED (captures functional transitions and discourse cadence)
            ("word_tfidf", TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=4000,
                sublinear_tf=True,
                stop_words=None,
                token_pattern=r"(?u)\b\w[\w']*\b"
            )),
            # Character n-grams (3-5) for morphological and punctuation rhythm without topic bias
            ("char_tfidf", TfidfVectorizer(
                analyzer='char_wb',
                ngram_range=(3, 5),
                max_features=5000,
                sublinear_tf=True
            ))
        ])

        clf = LogisticRegression(
            C=1.2,
            class_weight='balanced',
            max_iter=1000,
            solver='lbfgs',
            random_state=42
        )

        return Pipeline([
            ("features", union),
            ("classifier", clf)
        ])

    def fit(self, texts: List[str], labels: List[int]):
        """Train the classifier on a labeled text corpus."""
        self.pipeline = self.build_pipeline()
        self.pipeline.fit(texts, labels)
        self.is_trained = True

    def save(self, target_path: str = None):
        """Save the trained model artifact."""
        target = target_path or self.model_path
        os.makedirs(os.path.dirname(os.path.abspath(target)), exist_ok=True)
        joblib.dump(self.pipeline, target)
        print(f"Model successfully saved to: {target}")

    def load_or_initialize(self):
        """Load an existing model if found, otherwise ready to train."""
        if os.path.exists(self.model_path):
            try:
                self.pipeline = joblib.load(self.model_path)
                self.is_trained = True
            except Exception as e:
                print(f"Notice: Could not load existing model from {self.model_path}: {e}")
                self.pipeline = None
                self.is_trained = False
        else:
            self.pipeline = None
            self.is_trained = False

    def predict_sentence_score(self, sentence: str, doc_avg_len: float = 18.0) -> float:
        """
        Evaluate probability of an individual sentence being AI-generated (0.0 to 1.0).
        Blends subword model prediction with local syntactic cues.
        """
        words = sentence.split()
        num_words = len(words)
        if num_words < 3:
            return 0.50

        # Base model score
        if self.is_trained and self.pipeline:
            try:
                prob_ai = float(self.pipeline.predict_proba([sentence])[0][1])
            except Exception:
                prob_ai = 0.50
        else:
            prob_ai = 0.50

        sentence_lower = sentence.lower()

        # 1. Check for specific LLM marker clichés in this sentence
        has_marker = False
        for marker in LLM_MARKERS:
            if re.search(marker, sentence_lower):
                prob_ai += 0.25
                has_marker = True
                break

        # 2. Check for human contractions (e.g. don't, can't, it's)
        has_contraction = any(re.search(pat, sentence_lower) for pat in CONTRACTIONS)
        if has_contraction:
            prob_ai -= 0.18

        # 3. Check for 1st-person voice
        clean_words = set(re.findall(r"\b[a-zA-Z']+\b", sentence_lower))
        if any(w in FIRST_PERSON_PRONOUNS for w in clean_words):
            prob_ai -= 0.12

        # 4. Check for varied punctuation (dashes, semicolons, parens)
        if any(c in sentence for c in ['—', '–', ';', '(', ')', '!']):
            prob_ai -= 0.08

        # 5. Sentence length extreme deviations (very short punchy sentences or very long compound ones)
        if num_words <= 6:
            prob_ai -= 0.10
        elif num_words >= 35:
            # Very long complex sentences are more common in human essays unless loaded with markers
            if not has_marker:
                prob_ai -= 0.06

        return max(0.04, min(0.96, round(prob_ai, 3)))

    def analyze(self, text: str, sensitivity: str = "balanced") -> Dict[str, Any]:
        """
        Comprehensive analysis of input text:
        - Sensitivity modes: 'balanced' (default), 'strict_ai', 'human_safe'
        - Blends ML probability with dense stylometrics (burstiness, contractions, pronouns, punctuation)
        - Computes calibrated AI % and Human %
        - Produces sentence-level heatmap and diagnostic breakdown
        """
        if not text or not text.strip():
            raise ValueError("Input text is empty. Please provide text to analyze.")

        clean_content = text.strip()
        linguistic = extract_all_features(clean_content)
        sentences = linguistic["sentences"]
        word_count = linguistic["word_count"]

        if word_count < 10:
            raise ValueError(f"Text is too short ({word_count} words). Please provide at least 15 words for reliable detection.")

        # 1. Base ML model probability
        if self.is_trained and self.pipeline:
            try:
                base_ai_prob = float(self.pipeline.predict_proba([clean_content])[0][1])
            except Exception:
                base_ai_prob = 0.50
        else:
            base_ai_prob = 0.50

        # 2. Multi-Signal Stylometric Score Calculation
        # Stylometric features evaluate real writing behavior independent of topic
        cv = linguistic["burstiness_cv"]
        min_max_ratio = linguistic["min_max_ratio"]
        contractions_per_100 = linguistic["contractions"]["per_100_words"]
        first_person_pct = linguistic["pronouns"]["first_person_pct"]
        third_person_pct = linguistic["pronouns"]["third_person_pct"]
        special_punct = linguistic["punctuation"]["special_punct_total"]
        marker_count = linguistic["llm_marker_count"]
        ttr = linguistic["lexical_ttr"]

        # Aggregate stylometric evidence
        stylometric_score = 0.50  # Start neutral

        # Burstiness (Sentence length variation)
        # AI sentences are remarkably uniform (CV: 0.18 - 0.35); Human sentences burst (CV > 0.48)
        if len(sentences) >= 3:
            if cv < 0.28:
                stylometric_score += 0.18  # Strong AI signal
            elif cv < 0.38:
                stylometric_score += 0.08  # Moderate AI signal
            elif cv > 0.58:
                stylometric_score -= 0.18  # Strong Human signal
            elif cv > 0.46:
                stylometric_score -= 0.08  # Moderate Human signal

        # Contractions (Human conversational fingerprint)
        if contractions_per_100 >= 1.5:
            stylometric_score -= 0.18  # Very likely human
        elif contractions_per_100 >= 0.8:
            stylometric_score -= 0.10
        elif contractions_per_100 == 0 and word_count >= 100:
            stylometric_score += 0.08  # Lack of contractions in longer text leans formal/AI

        # Pronoun perspective
        if first_person_pct >= 2.5:
            stylometric_score -= 0.14  # High 1st person leans human
        elif third_person_pct >= 8.0 and first_person_pct == 0:
            stylometric_score += 0.10  # Exclusively impersonal 3rd person leans AI

        # Punctuation diversity
        if special_punct >= 4:
            stylometric_score -= 0.12  # Rich punctuation (dashes, semicolons, parens)
        elif special_punct == 0 and word_count >= 120:
            stylometric_score += 0.06  # Strictly periods & commas

        # LLM marker clichés
        if marker_count >= 3:
            stylometric_score += 0.28
        elif marker_count >= 1:
            stylometric_score += 0.15

        # Lexical diversity
        if ttr < 0.45 and word_count >= 120:
            stylometric_score += 0.08
        elif ttr > 0.72 and word_count >= 80:
            stylometric_score -= 0.08

        # 3. Dynamic Weighting based on Text Length
        # For longer texts, stylometrics are highly reliable. For short texts, model n-grams dominate.
        if word_count < 45:
            w_model, w_style = 0.75, 0.25
        elif word_count < 120:
            w_model, w_style = 0.55, 0.45
        else:
            w_model, w_style = 0.45, 0.55

        combined_prob = (base_ai_prob * w_model) + (stylometric_score * w_style)

        # 4. Sensitivity Calibration
        if sensitivity == "strict_ai":
            # Catches subtle/rephrased AI by boosting AI sensitivity
            combined_prob = combined_prob * 1.15
        elif sensitivity == "human_safe":
            # Prioritizes human protection (minimizes false positives)
            combined_prob = combined_prob * 0.88

        # Clamp calibrated probability
        calibrated_ai = max(0.02, min(0.98, combined_prob))
        ai_percentage = round(calibrated_ai * 100, 1)
        human_percentage = round(100.0 - ai_percentage, 1)

        # 5. Sentence-Level Heatmap Analysis
        avg_sentence_len = linguistic["avg_sentence_len"]
        sentence_results = []
        for idx, s in enumerate(sentences):
            s_score = self.predict_sentence_score(s, doc_avg_len=avg_sentence_len)
            
            # Apply overall document context to sentence
            adjusted_s_score = round(0.65 * s_score + 0.35 * calibrated_ai, 3)
            s_ai_pct = round(adjusted_s_score * 100, 1)

            if adjusted_s_score >= 0.65:
                s_verdict = "AI-Generated"
                css_tag = "ai-high"
            elif adjusted_s_score <= 0.38:
                s_verdict = "Human-Written"
                css_tag = "human-high"
            else:
                s_verdict = "Mixed / Uncertain"
                css_tag = "mixed"

            sentence_results.append({
                "index": idx + 1,
                "text": s,
                "ai_score": adjusted_s_score,
                "ai_percentage": s_ai_pct,
                "human_percentage": round(100.0 - s_ai_pct, 1),
                "verdict": s_verdict,
                "css_tag": css_tag
            })

        # 6. Overall Verdict & Explanatory Diagnostic Summary
        if ai_percentage >= 65.0:
            verdict = "Likely AI-Generated"
            verdict_badge = "ai-verdict"
            reasons = []
            if cv < 0.35 and len(sentences) >= 3:
                reasons.append("uniform sentence cadence")
            if marker_count > 0:
                reasons.append(f"{marker_count} common LLM transitional markers")
            if contractions_per_100 == 0 and word_count > 70:
                reasons.append("lack of conversational contractions")
            if third_person_pct > 6.0 and first_person_pct == 0:
                reasons.append("impersonal third-person perspective")
            
            reason_str = ", ".join(reasons) if reasons else "predictable syntactic structures"
            summary_desc = (
                f"The text exhibits hallmarks typical of large language models, including {reason_str}."
            )
        elif human_percentage >= 65.0:
            verdict = "Likely Human-Authored"
            verdict_badge = "human-verdict"
            reasons = []
            if cv > 0.45:
                reasons.append("natural sentence burstiness and rhythmic variance")
            if contractions_per_100 > 0.5:
                reasons.append("organic use of contractions")
            if first_person_pct > 1.5:
                reasons.append("authentic first-person voice")
            if special_punct >= 2:
                reasons.append("varied punctuation (dashes/semicolons/parentheses)")
                
            reason_str = ", ".join(reasons) if reasons else "natural linguistic cadence"
            summary_desc = (
                f"The text demonstrates hallmarks of authentic human writing, including {reason_str}."
            )
        else:
            verdict = "Mixed / Hybrid Content"
            verdict_badge = "mixed-verdict"
            summary_desc = (
                "The text displays a combination of both AI-like structures and human-like stylistic variations, "
                "suggesting human editing of AI text, AI rephrasing of a human draft, or borderline stylistic features."
            )

        # Confidence rating
        diff = abs(ai_percentage - 50.0)
        if diff >= 25.0:
            confidence = "High Confidence"
        elif diff >= 12.0:
            confidence = "Moderate Confidence"
        else:
            confidence = "Low / Borderline Confidence"

        return {
            "verdict": verdict,
            "verdict_badge": verdict_badge,
            "confidence": confidence,
            "ai_percentage": ai_percentage,
            "human_percentage": human_percentage,
            "summary_description": summary_desc,
            "sensitivity_mode": sensitivity,
            "linguistic_metrics": linguistic,
            "sentences": sentence_results,
            "stats": {
                "total_sentences": len(sentences),
                "total_words": word_count,
                "total_chars": linguistic["char_count"],
                "burstiness_cv": linguistic["burstiness_cv"],
                "contractions_per_100": contractions_per_100,
                "first_person_pct": first_person_pct,
                "punctuation_diversity": special_punct,
                "reading_ease": linguistic["reading_ease"],
                "grade_level": linguistic["grade_level"],
                "lexical_diversity": linguistic["lexical_ttr"],
                "marker_count": marker_count
            }
        }
