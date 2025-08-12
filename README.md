Talah HR Matching & Sentiment Analysis AI System
Overview
This project is an end-to-end AI pipeline for:

Parsing CVs (PDF/Text) and Job Descriptions (JD).

Computing similarity scores using TF-IDF + Cosine Similarity.

Analyzing HR feedback sentiment using VADER.

Making automated hiring decisions using a Simple Reinforcement Learning (RL) Agent.

Visualizing results, similarity scores, and RL reward progression.

The system supports notebook-based execution (notebook.ipynb) and can be extended to a Streamlit app or Flask API.

Folder Structure
bash
Copy
Edit
project/
│
├── data/
│   ├── cvs/                  # PDF or text CV files
│   ├── jds/                  # Job description text files
│   └── feedbacks/            # Feedback text files
│
├── utils/
│   ├── universal_parser.py 
|   ├── preprocess.py
|   ├── rl_agent.py
|   ├── matcher.py 
│   ├── embedding.py          # TF-IDF + Cosine Similarity
│   ├── sentiment.py          # Sentiment analysis using TextBlob
│   ├── decision.py           # RL Agent + decision rules
│
├── notebook.ipynb            # Main demonstration notebook
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
Setup Instructions
1️⃣ Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
requirements.txt should include:

nginx
Copy
Edit
pandas
numpy
scikit-learn
VADER
matplotlib
seaborn
PyPDF2
2️⃣ Place Your Data
CVs → data/cvs/ (PDF or .txt)

JDs → data/jds/ (TXT format preferred)

Feedbacks → data/feedbacks/ (TXT format preferred)

Example:

kotlin
Copy
Edit
data/
  cvs/
    cv1.pdf
    cv2.pdf
  jds/
    jd1.txt
  feedbacks/
    feedback1.txt
3️⃣ Run the Notebook
bash
Copy
Edit
jupyter notebook notebook.ipynb
Inside the notebook:

Data Parsing – Extracts CV text, JD text, and feedback.

Similarity Calculation – TF-IDF + Cosine Similarity between CV and JD.

Sentiment Analysis – Classifies feedback as Positive, Neutral, or Negative.

RL Agent – Updates rewards and decisions based on matching rules.

Visualization – Plots:

Similarity score distribution

Sentiment counts

RL Agent Reward History Over Time

Decision Logic
If similarity score ≥ threshold and skill match ≥ threshold → Strong Hire or Consider.

If low similarity or missing degree → Reject or Needs Review.

RL Agent adjusts actions based on rewards (feedback).

Example Output
Sample similarity scores:

yaml
Copy
Edit
CV 1 Similarity Score: 0.1849
CV 2 Similarity Score: 0.1775
CV 3 Similarity Score: 0.3155
Sample sentiment distribution:

makefile
Copy
Edit
Positive: 12
Neutral: 5
Negative: 3
RL Reward History Graph:
📈 Shows how the agent learns better decisions over iterations.

Extending the Project
Streamlit UI → Multiple CV uploads, JD input, and interactive results.

Flask API → Integrate with HR systems.

Model Upgrade → Replace TF-IDF with OpenAI Embeddings or BERT.

Author
Talha usmani – AI HR Matching & Sentiment Analysis System (2025)