from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.preprocess import clean_text

def compute_similarity(cv_texts, jd_text):
    """
    Takes a list of CV texts and a single JD text, returns a list of cosine similarity scores.
    """
    # Clean JD and all CVs
    jd_clean = clean_text(jd_text)
    cv_clean_list = [clean_text(cv) for cv in cv_texts]

    # Combine all texts for vectorization
    corpus = [jd_clean] + cv_clean_list

    # Vectorize using TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)

    # JD is the first vector, compare with others
    jd_vector = tfidf_matrix[0]
    cv_vectors = tfidf_matrix[1:]

    # Compute cosine similarity
    similarities = cosine_similarity(jd_vector, cv_vectors).flatten()

    return similarities

# Sample usage (can delete after testing)
if __name__ == "__main__":
    cv_samples = [
        "Experienced software engineer with Python and machine learning skills.",
        "Sales executive with 5 years of experience in retail and CRM.",
        "Flask developer skilled in APIs, NLP, and backend systems."
    ]
    jd_sample = "Looking for a Python backend developer with NLP and API experience."

    scores = compute_similarity(cv_samples, jd_sample)
    for idx, score in enumerate(scores):
        print(f"CV {idx+1} Similarity Score: {score:.4f}")
  
