# universal_parser.py
import re
import spacy
from rapidfuzz import fuzz
from collections import defaultdict

nlp = spacy.load("en_core_web_sm")
STOPWORDS = spacy.lang.en.stop_words.STOP_WORDS

DEGREE_MAP = {
    "bachelor": "BTECH",
    "bsc": "BSC",
    "b.e": "BTECH",
    "btech": "BTECH",
    "master": "MTECH",
    "msc": "MSC",
    "mtech": "MTECH",
    "phd": "PHD",
    "doctor": "PHD"
}

BASE_SKILLS = [
    "python", "java", "c++", "machine learning", "deep learning",
    "nlp", "sql", "data analysis", "flask", "django",
    "pandas", "numpy", "tensorflow", "pytorch", "excel",
    "marketing", "seo", "content creation", "social media",
    "accounting", "financial analysis", "auditing"
]

DOMAIN_SYNONYMS = {
    "IT": ["python", "java", "c++", "sql", "flask", "django", "tensorflow", "pytorch", "machine learning", "deep learning", "nlp"],
    "Marketing": ["seo", "content creation", "social media", "branding", "campaign"],
    "Finance": ["accounting", "financial analysis", "auditing", "budgeting", "forecasting"],
}

DOMAIN_KEYWORDS = {
    "IT": ["software", "developer", "programmer", "engineer", "python", "java", "database"],
    "Marketing": ["marketing", "seo", "brand", "campaign", "advertising"],
    "Finance": ["finance", "accounting", "audit", "budget", "tax"]
}

def detect_domain(text):
    text_lower = text.lower()
    domain_scores = defaultdict(int)
    for domain, keywords in DOMAIN_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                domain_scores[domain] += 1
    if not domain_scores:
        return "General"
    return max(domain_scores, key=domain_scores.get)

def normalize_degree(text):
    text_lower = text.lower()
    best_match = None
    best_score = 0
    for key in DEGREE_MAP.keys():
        score = fuzz.partial_ratio(key, text_lower)
        if score > best_score and score > 70:
            best_score = score
            best_match = key
    return DEGREE_MAP.get(best_match, "UNKNOWN")

def extract_degree(text):
    return normalize_degree(text)

def extract_skills(text, domain=None):
    text_lower = text.lower()
    skill_candidates = set(BASE_SKILLS)
    if domain and domain in DOMAIN_SYNONYMS:
        skill_candidates.update(DOMAIN_SYNONYMS[domain])
    found_skills = set()
    for skill in skill_candidates:
        score = fuzz.partial_ratio(skill.lower(), text_lower)
        if score >= 70:
            found_skills.add(skill)
    return sorted(found_skills)

def extract_experience(text):
    matches = re.findall(r'(\d+)\s+(?:years|yrs|year)', text.lower())
    if matches:
        return max(int(x) for x in matches)
    return 0

def extract_keywords(text, top_n=20):
    doc = nlp(text.lower())
    keywords = []
    for chunk in doc.noun_chunks:
        if len(chunk.text.strip()) > 2 and not any(tok.is_stop for tok in chunk):
            keywords.append(chunk.text.strip())
    freq = defaultdict(int)
    for kw in keywords:
        freq[kw] += 1
    sorted_kw = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [kw for kw, _ in sorted_kw[:top_n]]

def parse_cv_text(filename, text):
    domain = detect_domain(text)
    degree = extract_degree(text)
    skills = extract_skills(text, domain)
    experience = extract_experience(text)
    return {
        'filename': filename,
        'domain': domain,
        'degree': degree,
        'skills': skills,
        'experience': experience,
        'text': text
    }

def extract_requirements(jd_text):
    domain = detect_domain(jd_text)
    keywords = extract_keywords(jd_text, top_n=30)
    skills = extract_skills(jd_text, domain)
    degrees = []
    for deg_key in DEGREE_MAP.keys():
        if fuzz.partial_ratio(deg_key, jd_text.lower()) > 70:
            degrees.append(DEGREE_MAP[deg_key])
    if not degrees:
        degrees = ["ANY"]
    return {
        "domain": domain,
        "degrees": sorted(set(degrees)),
        "skills": sorted(set(skills + keywords)),
        "text": jd_text
    }
