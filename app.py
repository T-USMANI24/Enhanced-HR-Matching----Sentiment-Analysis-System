import streamlit as st
import fitz  # PyMuPDF
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from utils.rl_agent import SimpleRLAgent
from utils.decision import make_decision, save_results_to_csv, save_results_to_json

st.set_page_config(page_title="Talha AI HR Matcher", layout="wide", page_icon="📄")

# Custom CSS styling
st.markdown("""
    <style>
        .main { background-color: #f9f9f9; }
        .stButton button {
            background-color: #4CAF50;
            color: white;
            padding: 0.5em 1.5em;
            border: none;
            border-radius: 8px;
            font-size: 1em;
        }
        .stDownloadButton button {
            background-color: #0E86D4;
            color: white;
            border-radius: 6px;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a6/Anonymous_emblem.svg/1200px-Anonymous_emblem.svg.png",
        width=100
    )
    st.title("📑 Talha HR Matcher")
    st.markdown("Upload CVs, enter Job Description and HR feedback, and get intelligent decisions.")
    st.markdown("---")
    st.info("💡 Built with RL + NLP")

# Tabs
tab1, tab2, tab3 = st.tabs(["📝 Input", "📊 Results", "⚙️ Logs"])

with tab1:
    with st.expander("📤 Upload CVs (PDF Format)", expanded=True):
        uploaded_cvs = st.file_uploader("Upload one or more CV PDFs", type="pdf", accept_multiple_files=True)

    with st.expander("📋 Paste Job Description", expanded=False):
        jd_text = st.text_area("Job Description", height=200)

    with st.expander("🗣️ Enter HR Feedback (one per candidate)", expanded=False):
        feedback_input = st.text_area("Enter HR feedbacks (each on a new line)", height=150)

    with st.expander("⚙️ Matching Settings", expanded=False):
        min_skill_threshold = st.slider(
            "Minimum Skill Match % Required", 0, 100, 20, step=5
        ) / 100.0
        min_similarity_threshold = st.slider(
            "Minimum Similarity Score Required", 0.0, 1.0, 0.5, step=0.05
        )

    run_button = st.button("🚀 Run Matching")

with tab2:
    if 'results' in st.session_state:
        st.success("✅ Decision Results")

        results_df = pd.DataFrame(st.session_state['results'])

        def color_decision(val):
            if val == "Hire":
                return 'background-color: #b6ffb3; color: black;'  # green
            elif val == "Reject":
                return 'background-color: #ffb3b3; color: black;'  # red
            elif val == "Reassign":
                return 'background-color: #fff5b3; color: black;'  # yellow
            return ''

        st.markdown("### 📄 Candidate Decisions (with explanations)")
        st.dataframe(results_df.style.applymap(color_decision, subset=['decision']))

        # Similarity Score Distribution
        st.markdown("### 🔍 Similarity Score Distribution")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(data=results_df, x="cv_name", y="similarity_score_%", palette="Blues_d", ax=ax)
        ax.set_title("Similarity Scores per CV")
        ax.set_xlabel("CV Filename")
        ax.set_ylabel("Similarity Score (%)")
        plt.xticks(rotation=45)
        st.pyplot(fig)

        # Sentiment Distribution Pie Chart
        st.markdown("### 🧠 Sentiment Distribution")
        sentiment_counts = results_df['sentiment_label'].value_counts()
        fig1, ax1 = plt.subplots()
        ax1.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90)
        ax1.axis('equal')
        st.pyplot(fig1)

        # Decision Actions Distribution
        st.markdown("### 🧾 Decision Actions Distribution")
        decision_counts = results_df['decision'].value_counts()
        fig2, ax2 = plt.subplots()
        sns.barplot(x=decision_counts.index, y=decision_counts.values, palette="Set2", ax=ax2)
        ax2.set_title("Actions Taken by RL Agent")
        ax2.set_ylabel("Count")
        st.pyplot(fig2)

        # Degree Match Pie Chart
        degree_counts = results_df['degree_match'].value_counts()
        degree_counts = degree_counts.reindex([True, False], fill_value=0)
        fig3, ax3 = plt.subplots()
        ax3.pie(
            degree_counts,
            labels=["Match", "Mismatch"],
            autopct='%1.1f%%',
            startangle=90,
            colors=["#4CAF50", "#F44336"]
        )
        ax3.set_title("Degree Match Distribution", fontsize=14, fontweight='bold')
        ax3.axis('equal')
        st.pyplot(fig3)

        # Skill Match Pie Chart
        def skill_category(pct):
            if pct >= 0.8:
                return "High Match"
            elif pct >= 0.5:
                return "Medium Match"
            else:
                return "Low Match"

        skill_cats = results_df['skill_match_%'].apply(skill_category)
        skill_counts = skill_cats.value_counts()
        fig4, ax4 = plt.subplots()
        ax4.pie(
            skill_counts,
            labels=skill_counts.index,
            autopct='%1.1f%%',
            startangle=90,
            colors=["#4CAF50", "#FFC107", "#F44336"]  # green, yellow, red
        )
        ax4.set_title(f"Candidate Skill Match Distribution (n = {len(results_df)})",
                      fontsize=14, fontweight='bold')
        ax4.axis('equal')
        st.pyplot(fig4)

        # Download Buttons
        try:
            with open("final_results.csv", "rb") as f_csv:
                csv_data = f_csv.read()
            with open("final_results.json", "rb") as f_json:
                json_data = f_json.read()
            st.download_button("⬇️ Download CSV", data=csv_data, file_name="results.csv")
            st.download_button("⬇️ Download JSON", data=json_data, file_name="results.json")
        except Exception as e:
            st.error(f"Error loading result files for download: {e}")

    else:
        st.info("Run the matching first in the Input tab.")

with tab3:
    st.text("Logs and system updates will appear here.")

def extract_text_from_pdf(file):
    try:
        file_bytes = file.read()
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    except Exception as e:
        st.error(f"Failed to extract text from PDF: {e}")
        return ""

if run_button:
    if uploaded_cvs and jd_text.strip() and feedback_input.strip():
        with st.spinner("Processing CVs and feedback..."):
            cv_texts = [extract_text_from_pdf(f) for f in uploaded_cvs]
            feedbacks = [line.strip() for line in feedback_input.strip().split("\n") if line.strip()]

            if len(cv_texts) != len(feedbacks):
                st.error("⚠️ Number of CVs and HR feedbacks must be the same!")
            else:
                agent = SimpleRLAgent(["Hire", "Reject", "Reassign"])

                training_data = [
                    (0.85, "Positive", 1, 1, "Hire", 10),
                    (0.20, "Negative", 0, 0, "Reject", 9),
                    (0.60, "Neutral", 1, 0, "Reassign", 6),
                ]

                for sim, sent, deg, skill, act, rew in training_data:
                    agent.update(sim, sent, deg, skill, act, rew)

                results = make_decision(
                    cv_texts,
                    jd_text,
                    feedbacks,
                    agent,
                    similarity_threshold=min_similarity_threshold,
                    skill_match_threshold=min_skill_threshold
                )

                for i, res in enumerate(results):
                    res['cv_name'] = uploaded_cvs[i].name

                st.session_state['results'] = results
                save_results_to_csv(results, "final_results.csv")
                save_results_to_json(results, "final_results.json")
    else:
        st.warning("Please upload CVs, paste JD, and enter feedbacks.")

