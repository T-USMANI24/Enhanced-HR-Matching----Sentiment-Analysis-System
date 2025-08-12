import json
import csv
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.embedding import compute_similarity
from utils.sentiment import classify_sentiment

# Replace these imports with the universal parser
from utils.universal_parser import parse_cv_text, extract_requirements


def evaluate_candidate(sim_score, sentiment_label, sentiment_score, degree_match, skill_pct,
                       similarity_threshold, skill_match_threshold, rl_agent):
    """
    Rule-based + RL decision combination
    """
    # Hard filters first
    if not degree_match:
        return "Reject", "❌ Degree mismatch", 0.0
    elif skill_pct < skill_match_threshold:
        return "Reject", f"❌ Skills match below {skill_match_threshold*100:.0f}%", 0.0
    elif sim_score < similarity_threshold:
        return "Reject", f"❌ Similarity score below {similarity_threshold:.2f}", 0.0

    # Sentiment → decision bias
    if sentiment_label.lower() == "positive":
        sentiment_bias = 1.0
    elif sentiment_label.lower() == "negative":
        sentiment_bias = -1.0
    else:
        sentiment_bias = 0.0

    # RL Agent decision
    action = rl_agent.choose_action(sim_score, sentiment_label, degree_match, skill_pct)
    q_values = rl_agent.get_q_values(sim_score, sentiment_label, degree_match, skill_pct)

    # Confidence = normalized Q-value for chosen action
    max_q = max(q_values.values()) if q_values else 1
    rl_conf = round((q_values.get(action, 0) / max_q) * 100, 1) if max_q != 0 else 0

    explanation = (
        f"Sim={sim_score*100:.1f}%, Sentiment={sentiment_label}({sentiment_score:.2f}), "
        f"DegreeMatch={degree_match}, Skills={skill_pct*100:.1f}%, "
        f"RL Action={action}, RL Confidence={rl_conf}%"
    )
    return action, explanation, rl_conf


def make_decision(cv_texts, jd_text, feedbacks, rl_agent,
                  similarity_threshold, skill_match_threshold):

    similarity_scores = compute_similarity(cv_texts, jd_text)
    sentiments = [classify_sentiment(fb) for fb in feedbacks]  # [(label, score), ...]

    jd_req = extract_requirements(jd_text)
    required_degrees = jd_req["degrees"]
    required_skills = jd_req["skills"]

    results = []
    for i, cv_text in enumerate(cv_texts):
        sim_score = similarity_scores[i]
        sent_label, sent_score = sentiments[i]

        # Parse CV using universal parser
        parsed_cv = parse_cv_text(f"cv_{i+1}", cv_text)
        cv_degree = parsed_cv["degree"]
        cv_skills = parsed_cv["skills"]

        degree_match = cv_degree in required_degrees or "ANY" in required_degrees
        skill_matches = set(cv_skills) & set(required_skills)
        skill_pct = len(skill_matches) / len(required_skills) if required_skills else 0.0

        action, explanation, rl_conf = evaluate_candidate(
            sim_score, sent_label, sent_score, degree_match, skill_pct,
            similarity_threshold, skill_match_threshold, rl_agent
        )

        # Simple reward logic for RL agent:
        # Reward +1 for "Strong Hire" or "Consider", else 0
        reward = 1.0 if action in ["Strong Hire", "Consider"] else 0.0

        # Update RL agent with this reward
        rl_agent.update(sim_score, sent_label, degree_match, skill_pct, action, reward)

        match_score = round((sim_score + skill_pct + (1 if degree_match else 0)) / 3 * 100, 1)

        result = {
            "cv_index": i + 1,
            "similarity_score_%": round(sim_score * 100, 1),
            "skill_match_%": round(skill_pct * 100, 1),
            "degree_match": degree_match,
            "match_score_%": match_score,
            "sentiment_label": sent_label,
            "sentiment_score": round(sent_score, 2),
            "rl_confidence_%": rl_conf,
            "decision": action,
            "explanation": explanation
        }

        results.append(result)
        log_decision(result)

    return results


def log_decision(result):
    print(f"[CV {result['cv_index']}] → {result['decision']} | {result['explanation']}")


def save_results_to_csv(results, filename="final_decisions.csv"):
    keys = results[0].keys()
    with open(filename, "w", newline='', encoding="utf-8") as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(results)


def save_results_to_json(results, filename="final_decisions.json"):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
