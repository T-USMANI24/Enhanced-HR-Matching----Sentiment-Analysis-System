import re
import string
import nltk
from nltk.corpus import stopwords
from PyPDF2 import PdfReader 
# Download stopwords once
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

def extract_text_from_pdf(pdf_path):
    """Extract all text from a PDF file."""
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text

def clean_text(text, remove_stopwords=False):
    """
    Clean text for embedding.
    - Lowercasing
    - Removing punctuation
    - Removing numbers & extra spaces
    - Optional stopword removal
    """
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    if remove_stopwords:
        words = text.split()
        words = [word for word in words if word not in stop_words]
        text = ' '.join(words)

    return text

def prepare_document(file_path, remove_stopwords=False):
    """Read file (PDF or TXT) and return cleaned text."""
    if file_path.lower().endswith('.pdf'):
        raw_text = extract_text_from_pdf(file_path)
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            raw_text = f.read()
    return clean_text(raw_text, remove_stopwords)
