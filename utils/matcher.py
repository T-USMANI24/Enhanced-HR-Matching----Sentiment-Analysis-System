import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.embedding import compute_similarity
from utils.preprocess import clean_text

def rule_based_score(similarity, cv_text, jd_skills):
    """
    Enhances similarity score based on skill keyword matches.
    - Adds 0.05 per skill match (up to max 1.0)
    """
    bonus = 0.0
    for skill in jd_skills:
        if skill.lower() in cv_text.lower():
            bonus += 0.05
    final_score = min(similarity + bonus, 1.0)
    return final_score

import os
from utils.preprocess import prepare_document  # make sure this import is correct

def read_files_from_folder(folder_path):
    texts = []
    filenames = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".txt", ".pdf")):
            path = os.path.join(folder_path, filename)
            try:
                text = prepare_document(path)
                if text.strip():  # skip empty texts
                    texts.append(text)
                    filenames.append(filename)
            except Exception as e:
                print(f"Error reading file {filename}: {e}")
    return filenames, texts


def match_cvs_to_jd(cv_folder, jd_path):
    """
    Read CVs and a single JD, calculate similarity scores.
    """
    # Load data
    cv_filenames, cv_texts = read_files_from_folder(cv_folder)
    with open(jd_path, "r", encoding="utf-8") as f:
        jd_text = f.read()

    # Step 1: Compute base similarities
    scores = compute_similarity(cv_texts, jd_text)

    # Step 2: Define expected skills from JD manually
    jd_skills = ["Python", "Flask", "APIs", "NLP", "Machine Learning"]

    # Step 3: Apply rule-based enhancement
    adjusted_results = []
    for i in range(len(cv_texts)):
        enhanced_score = rule_based_score(scores[i], cv_texts[i], jd_skills)
        adjusted_results.append((cv_filenames[i], enhanced_score))

    # Step 4: Sort and return
    adjusted_results.sort(key=lambda x: x[1], reverse=True)
    return adjusted_results

# Test run
if __name__ == "__main__":
    cv_folder = "data/sample_cvs"
    jd_file = "data/sample_jds/jd1.txt"
    results = match_cvs_to_jd(cv_folder, jd_file)

    print("CV Matching Results:\n")
    for filename, score in results:
        print(f"{filename}: {score:.4f}")
