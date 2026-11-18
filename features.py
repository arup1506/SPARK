"""
Stylometric and Linguistic Feature Extraction for AI Text Detection.
Comprehensive multi-signal feature extraction:
- Sentence Burstiness & Cadence (CV, std-dev, min/max ratio)
- Contraction frequency (Human conversational fingerprint)
- Pronoun perspective distribution (1st person vs 2nd vs 3rd/impersonal)
- Punctuation diversity & rhythm (dashes, semicolons, parentheses, colons)
- Expanded LLM Discourse Markers & Signature Lexicon (120+ tropes)
- Lexical Diversity (Type-Token Ratio, Hapax Legomena, Root TTR)
- Perplexity / Entropy approximation (Shannon Bigram Surprisal)
- Readability metrics (Flesch Reading Ease & Grade Level)
"""

import math
import re
from typing import List, Dict, Any, Tuple
from collections import Counter


# Comprehensive catalog of over-indexed LLM transition markers, filler phrases, and stylistic tropes
LLM_MARKERS = [
    # Transition & Discourse markers
    r"\bin conclusion\b",
    r"\bfurthermore\b",
    r"\bmoreover\b",
    r"\bconsequently\b",
    r"\badditionally\b",
    r"\bin summary\b",
    r"\bto summarize\b",
    r"\bultimately\b",
    r"\ball in all\b",
    r"\bin a nutshell\b",
    
    # Preachy / Cautionary phrases
    r"\bit is important to remember\b",
    r"\bit is important to note\b",
    r"\bit is worth noting that\b",
    r"\bit is essential to\b",
    r"\bit is crucial to\b",
    r"\bplays a crucial role\b",
    r"\bplays a vital role\b",
    r"\bplays a pivotal role\b",
    r"\bplays an indispensable role\b",
    r"\bparamount importance\b",
    r"\bof paramount\b",
    r"\bunderscores the need\b",
    r"\bunderscores the importance\b",
    r"\bserves as a reminder\b",
    r"\bserves as a testament\b",
    r"\btestament to\b",
    r"\bstands as a testament\b",
    
    # Overused LLM metaphors & buzzwords
    r"\bdelve into\b",
    r"\bdelves into\b",
    r"\bdelving into\b",
    r"\btapestry of\b",
    r"\brich tapestry\b",
    r"\bbeacon of\b",
    r"\bbeacon of hope\b",
    r"\bin today's fast-paced world\b",
    r"\bfast-paced world\b",
    r"\bever-evolving landscape\b",
    r"\bmodern landscape\b",
    r"\bnavigating the complexities\b",
    r"\bnavigating the nuances\b",
    r"\bnavigating the\b",
    r"\bfoster a sense of\b",
    r"\bfostering a culture\b",
    r"\bfosters collaboration\b",
    r"\bat its core\b",
    r"\bat the core of\b",
    r"\bnot only .*? but also\b",
    r"\bunveils\b",
    r"\bseamlessly integrates\b",
    r"\bseamless integration\b",
    r"\bseamlessly blends\b",
    r"\bvital component\b",
    r"\bpoised to\b",
    r"\bpaving the way\b",
    r"\bheralds a new era\b",
    r"\bholistic approach\b",
    r"\bmultifaceted nature\b",
    r"\bmyriad of\b",
    r"\bplethora of\b",
    r"\bembodies the essence\b",
    r"\ba profound impact\b",
    r"\bharnessing the power\b",
    r"\bleveraging cutting-edge\b",
    r"\bremains to be seen\b",
    r"\bcornerstone of\b",
    r"\bintricate dance\b",
    r"\bshed light on\b",
    r"\bculmination of\b",
    r"\bunravel the\b",
    r"\bvibrant tapestry\b",
    r"\bresonates deeply\b",
    r"\ba wide array of\b",
    r"\bdelicate balance\b"
]

# Common English contractions (humans use frequently; standard LLMs rarely produce unless instructed)
CONTRACTIONS = [
    r"\bdon't\b", r"\bcan't\b", r"\bit's\b", r"\bi'm\b", r"\bwe're\b", r"\bthey're\b",
    r"\bdidn't\b", r"\bwon't\b", r"\bcouldn't\b", r"\bwasn't\b", r"\bi've\b", r"\byou're\b",
    r"\bthere's\b", r"\bhasn't\b", r"\bhaven't\b", r"\bwouldn't\b", r"\bshouldn't\b",
    r"\bi'd\b", r"\bwe've\b", r"\blet's\b", r"\bthat's\b", r"\bwho's\b", r"\bwhat's\b",
    r"\baren't\b", r"\bisn't\b", r"\byou've\b", r"\bshe's\b", r"\bhe's\b", r"\bain't\b"
]

FIRST_PERSON_PRONOUNS = {
    "i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves"
}

SECOND_PERSON_PRONOUNS = {
    "you", "your", "yours", "yourself", "yourselves"
}

THIRD_PERSON_IMPERSONAL = {
    "it", "its", "itself", "one", "oneself", "this", "these", "that", "those",
    "they", "them", "their", "theirs", "themselves"
}


def split_sentences(text: str) -> List[str]:
    """
    Robust sentence boundary detection without external download dependencies.
    Handles abbreviations (Dr., e.g., i.e.), quotes, and line breaks.
    """
    if not text or not text.strip():
        return []
    
    # Normalize clean single lines within paragraphs
    text = re.sub(r'([a-z])\n([a-z])', r'\1 \2', text)
    
    # Split by standard sentence terminators (. ! ?) followed by whitespace or quote
    raw_sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9"\'\“\‘])', text)
    
    cleaned_sentences = []
    for s in raw_sentences:
        s = s.strip()
        if len(s) > 1:
            cleaned_sentences.append(s)
            
    return cleaned_sentences if cleaned_sentences else [text.strip()]


def tokenize_words(text: str) -> List[str]:
    """Extract clean lowercase alphabetic words."""
    return re.findall(r"\b[a-zA-Z']{2,}\b", text.lower())


def count_syllables(word: str) -> int:
    """Heuristic syllable counter for English words."""
    word = word.lower().strip()
    if len(word) <= 3:
        return 1
    # Remove silent e
    word = re.sub(r'(?:[^laeiouy]|ed|es|e)$', '', word)
    # Count vowel groups
    syllables = len(re.findall(r'[aeiouy]{1,2}', word))
    return max(1, syllables)


def calculate_burstiness(sentences: List[str]) -> Tuple[float, float, float, float]:
    """
    Calculate sentence length variation (Burstiness).
    Returns (mean_length, std_dev, coefficient_of_variation, min_max_ratio).
    AI text is notably uniform; human text has high variance and extreme min/max swings.
    """
    if not sentences:
        return 0.0, 0.0, 0.0, 1.0
    
    lengths = [len(tokenize_words(s)) for s in sentences if len(tokenize_words(s)) > 0]
    if not lengths:
        return 0.0, 0.0, 0.0, 1.0
    
    mean_len = sum(lengths) / len(lengths)
    if len(lengths) < 2:
        return float(mean_len), 0.0, 0.0, 1.0
        
    variance = sum((l - mean_len) ** 2 for l in lengths) / (len(lengths) - 1)
    std_dev = math.sqrt(variance)
    cv = (std_dev / mean_len) if mean_len > 0 else 0.0
    
    min_len = min(lengths)
    max_len = max(lengths)
    min_max_ratio = (min_len / max_len) if max_len > 0 else 1.0
    
    return float(mean_len), float(std_dev), float(cv), float(min_max_ratio)


def calculate_contractions(text: str, total_words: int) -> Dict[str, Any]:
    """
    Count contractions in text.
    High contractions indicate natural human conversational style.
    """
    text_lower = text.lower()
    total_contractions = 0
    matched = []
    
    for pattern in CONTRACTIONS:
        found = re.findall(pattern, text_lower)
        if found:
            total_contractions += len(found)
            matched.extend(found)
            
    ratio_per_100 = (total_contractions / max(1, total_words)) * 100.0
    return {
        "count": total_contractions,
        "per_100_words": round(ratio_per_100, 2),
        "examples": list(set(matched))[:5]
    }


def calculate_pronoun_distribution(words: List[str]) -> Dict[str, float]:
    """
    Measure pronoun perspective.
    Humans frequently use 1st-person ('I', 'we', 'my').
    AI heavily prefers impersonal 3rd-person ('it', 'this', 'one').
    """
    if not words:
        return {"first_person_pct": 0.0, "second_person_pct": 0.0, "third_person_pct": 0.0}
    
    total = len(words)
    c1 = sum(1 for w in words if w in FIRST_PERSON_PRONOUNS)
    c2 = sum(1 for w in words if w in SECOND_PERSON_PRONOUNS)
    c3 = sum(1 for w in words if w in THIRD_PERSON_IMPERSONAL)
    
    return {
        "first_person_pct": round((c1 / total) * 100.0, 2),
        "second_person_pct": round((c2 / total) * 100.0, 2),
        "third_person_pct": round((c3 / total) * 100.0, 2)
    }


def calculate_punctuation_richness(text: str) -> Dict[str, Any]:
    """
    Analyze punctuation diversity.
    Humans use dashes, semicolons, parentheses, ellipses, and quotes.
    AI almost exclusively uses periods, commas, and lists.
    """
    dashes = len(re.findall(r'[—–]|(?<!-)-(?!-)', text))
    semicolons = text.count(';')
    parentheses = text.count('(') + text.count(')')
    colons = text.count(':')
    quotes = len(re.findall(r'["\'\“\‘\”\’]', text))
    exclamations = text.count('!')
    questions = text.count('?')
    ellipses = len(re.findall(r'\.\.\.|…', text))
    
    special_punct_count = dashes + semicolons + parentheses + exclamations + ellipses
    
    return {
        "dashes": dashes,
        "semicolons": semicolons,
        "parentheses": parentheses,
        "colons": colons,
        "exclamations": exclamations,
        "questions": questions,
        "ellipses": ellipses,
        "special_punct_total": special_punct_count
    }


def calculate_lexical_diversity(words: List[str]) -> Dict[str, float]:
    """
    Calculate vocabulary richness:
    - Type-Token Ratio (TTR)
    - Root TTR (V / sqrt(N), independent of length)
    - Hapax Legomena Ratio (words appearing exactly once)
    """
    if not words:
        return {"ttr": 0.0, "root_ttr": 0.0, "hapax_ratio": 0.0, "unique_words": 0}
    
    total = len(words)
    counts = Counter(words)
    unique = len(counts)
    hapax = sum(1 for w, c in counts.items() if c == 1)
    root_ttr = unique / math.sqrt(total) if total > 0 else 0.0
    
    return {
        "ttr": round(unique / total, 4),
        "root_ttr": round(root_ttr, 2),
        "hapax_ratio": round(hapax / total, 4),
        "unique_words": unique
    }


def calculate_entropy_proxy(text: str) -> float:
    """
    Calculate Shannon character bigram entropy as a proxy for perplexity / predictability.
    Lower entropy indicates more predictable, standard model-generated text.
    """
    clean = re.sub(r'\s+', ' ', text.lower().strip())
    if len(clean) < 10:
        return 0.0
        
    bigrams = [clean[i:i+2] for i in range(len(clean) - 1)]
    if not bigrams:
        return 0.0
        
    total_bigrams = len(bigrams)
    counts = Counter(bigrams)
    
    entropy = 0.0
    for count in counts.values():
        p = count / total_bigrams
        entropy -= p * math.log2(p)
        
    return round(entropy, 4)


def calculate_readability(words: List[str], sentences: List[str]) -> Dict[str, float]:
    """
    Calculate Flesch Reading Ease and Flesch-Kincaid Grade Level.
    """
    num_words = max(1, len(words))
    num_sentences = max(1, len(sentences))
    num_syllables = sum(count_syllables(w) for w in words)
    
    asl = num_words / num_sentences
    asw = num_syllables / num_words
    
    fre = 206.835 - (1.015 * asl) - (84.6 * asw)
    fre = max(0.0, min(100.0, round(fre, 2)))
    
    fkgl = (0.39 * asl) + (11.8 * asw) - 15.59
    fkgl = max(1.0, min(20.0, round(fkgl, 1)))
    
    return {
        "reading_ease": fre,
        "grade_level": fkgl,
        "avg_sentence_length": round(asl, 1),
        "avg_syllables_per_word": round(asw, 2)
    }


def detect_llm_markers(text: str) -> List[str]:
    """Find matches of typical LLM discourse markers and clichés."""
    matches = []
    text_lower = text.lower()
    for pattern in LLM_MARKERS:
        found = re.findall(pattern, text_lower)
        if found:
            matches.extend(found)
    return list(set(matches))


def extract_all_features(text: str) -> Dict[str, Any]:
    """
    Aggregate all stylometric, statistical, and linguistic features into a comprehensive dictionary.
    """
    sentences = split_sentences(text)
    words = tokenize_words(text)
    total_words = len(words)
    
    mean_len, std_dev, cv, min_max = calculate_burstiness(sentences)
    contractions = calculate_contractions(text, total_words)
    pronouns = calculate_pronoun_distribution(words)
    punct = calculate_punctuation_richness(text)
    lex = calculate_lexical_diversity(words)
    entropy = calculate_entropy_proxy(text)
    readability = calculate_readability(words, sentences)
    markers = detect_llm_markers(text)
    
    # Calculate an intrinsic stylometric AI index (-1.0 to +1.0)
    # Positive indicates AI hallmarks; Negative indicates Human hallmarks
    ai_bias_score = 0.0
    
    # Burstiness factor: AI sentences are clustered around mean (low cv, high min/max ratio)
    if cv < 0.32:
        ai_bias_score += 0.22
    elif cv > 0.55:
        ai_bias_score -= 0.22
        
    # Contractions: humans use contractions frequently
    if contractions["per_100_words"] >= 1.5:
        ai_bias_score -= 0.25
    elif contractions["per_100_words"] == 0 and total_words > 80:
        ai_bias_score += 0.12
        
    # Perspective: 1st person is overwhelmingly human in personal/blog contexts
    if pronouns["first_person_pct"] >= 3.0:
        ai_bias_score -= 0.20
    elif pronouns["third_person_pct"] >= 8.0 and pronouns["first_person_pct"] == 0:
        ai_bias_score += 0.12
        
    # Punctuation richness: human texts use dashes, semicolons, parens
    if punct["special_punct_total"] >= 3:
        ai_bias_score -= 0.15
        
    # Cliché markers
    marker_count = len(markers)
    if marker_count >= 1:
        ai_bias_score += min(0.35, marker_count * 0.12)
        
    return {
        "sentence_count": len(sentences),
        "word_count": total_words,
        "char_count": len(text),
        "burstiness_mean_len": round(mean_len, 2),
        "burstiness_std_dev": round(std_dev, 2),
        "burstiness_cv": round(cv, 3),
        "min_max_ratio": round(min_max, 3),
        "contractions": contractions,
        "pronouns": pronouns,
        "punctuation": punct,
        "lexical_ttr": lex["ttr"],
        "lexical_root_ttr": lex["root_ttr"],
        "lexical_hapax": lex["hapax_ratio"],
        "unique_words": lex["unique_words"],
        "entropy": entropy,
        "reading_ease": readability["reading_ease"],
        "grade_level": readability["grade_level"],
        "avg_sentence_len": readability["avg_sentence_length"],
        "llm_markers_found": markers,
        "llm_marker_count": marker_count,
        "stylometric_bias": round(ai_bias_score, 3),
        "sentences": sentences
    }
