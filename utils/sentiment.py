# utils/sentiment.py
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()
analyzer.lexicon.update({
    "poor": -2.0, "weak": -1.8, "unprepared": -2.0,
    "disorganized": -1.7, "hesitant": -1.5, "lacked": -1.5,
    "confusing": -1.6, "vague": -1.4, "defensive": -1.5,
    "uninterested": -1.8, "monotone": -1.6, "failed": -2.2,
    "confident": 1.8, "enthusiastic": 2.0, "articulate": 1.7,
    "prepared": 1.6, "engaging": 1.8, "insightful": 1.9,
    "impressive": 2.0, "strong": 1.5, "excellent": 2.2,
    "clear": 1.6, "thoughtful": 1.7, "professional": 1.5
})

def classify_sentiment(text):
    """
    Classify sentiment using VADER.
    Returns: (label, score)
    """
    score = analyzer.polarity_scores(text)['compound']
    if score >= 0.4:
        label = "Positive"
    elif score <= -0.2:
        label = "Negative"
    else:
        label = "Neutral"
    return label, round(score, 2)

def process_feedbacks(feedback_file="data/feedbacks.txt"):
    """
    Reads feedbacks from file, returns a list of (feedback, label, score).
    """
    if not os.path.exists(feedback_file):
        raise FileNotFoundError(f"Feedback file not found: {feedback_file}")

    results = []
    with open(feedback_file, "r", encoding="utf-8") as f:
        for line in f:
            feedback = line.strip()
            if feedback:
                label, score = classify_sentiment(feedback)
                results.append((feedback, label, score))
    return results

if __name__ == "__main__":
    feedback_results = process_feedbacks()
    for fb, label, score in feedback_results:
        print(f"Feedback: {fb}")
        print(f"→ Sentiment: {label} ({score})\n")
