import pandas as pd
import json
import os
import numpy as np
import time
import requests
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from textblob import TextBlob

load_dotenv()

# === Configuration ===
EMBEDDINGS_FILE = "/Users/asus/Downloads/tweet_embeddings.json"
CSV_FILE = "Tweets_clean.csv"
MODEL_NAME = "gemini-2.5-flash"
API_KEY = os.getenv("API_KEY", "")
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={API_KEY}"

MAX_RETRIES = 5
TOP_N_TWEETS = 10
N_RANDOM_SAMPLES = 5

# ===============================
# JSON-safe helper
# ===============================
def to_json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): to_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_json_safe(x) for x in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    else:
        return obj

# ===============================
# Load Embeddings
# ===============================
def load_and_process_embeddings(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None

    with open(file_path, "r") as f:
        data = json.load(f)

    if "embeddings" not in data:
        print("JSON missing 'embeddings'")
        return None

    embeddings_dict = data["embeddings"]

    df = pd.DataFrame({
        "tweet_id": list(embeddings_dict.keys()),
        "embedding_vector": list(embeddings_dict.values())
    })

    print(f"Loaded {len(df)} embeddings")
    return df

# ===============================
# Load Tweets
# ===============================
def load_original_tweets(csv_path):
    if not os.path.exists(csv_path):
        print(f"CSV not found: {csv_path}")
        return None

    df = pd.read_csv(csv_path)
    df["tweet_id"] = df["tweet_id"].astype(str)
    return df[["tweet_id", "clean_text"]]

# ===============================
# GMM Clustering
# ===============================
def perform_gmm_clustering(df, n_components=7):
    X = np.array(df["embedding_vector"].tolist())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    gmm = GaussianMixture(n_components=n_components, random_state=42)
    df["cluster_label"] = gmm.fit_predict(X_scaled)

    cluster_means = []
    for i in range(n_components):
        cluster_vecs = X[df.cluster_label == i]
        if len(cluster_vecs) == 0:
            cluster_means.append(np.zeros_like(X[0]))
        else:
            cluster_means.append(cluster_vecs.mean(axis=0))

    return df, X, np.array(cluster_means)

# ===============================
# Sentiment Analysis
# ===============================
def get_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.1:
        return "Positive"
    elif polarity < -0.1:
        return "Negative"
    else:
        return "Neutral"

# ===============================
# Gemini API Caller
# ===============================
def call_gemini_api(payload):
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(
                API_URL,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=30
            )

            if response.status_code == 429:  # rate limit
                time.sleep(2 ** attempt)
                continue

            if response.status_code != 200:
                print("Gemini API error:", response.text)
                return "Labeling Failed"

            data = response.json()
            candidate = data.get("candidates", [{}])[0]
            parts = candidate.get("content", {}).get("parts", [{}])
            return parts[0].get("text", "").strip()

        except Exception:
            time.sleep(2 ** attempt)

    return "Labeling Failed"

# ===============================
# Label Clusters & Compute Stats
# ===============================
def label_clusters_with_gemini(clustered_df, original_tweets_df, cluster_means, n_components):

    df_merged = pd.merge(clustered_df, original_tweets_df, on="tweet_id")
    cluster_labels = {}
    cluster_summary = {}

    overall_positive = 0
    overall_negative = 0
    overall_neutral = 0

    for k in range(n_components):

        df_c = df_merged[df_merged.cluster_label == k]
        if df_c.empty:
            cluster_labels[k] = "Empty Cluster"
            continue

        # Compute similarities
        embeds = np.array(df_c["embedding_vector"].tolist())
        centroid = cluster_means[k].reshape(1, -1)
        sims = cosine_similarity(centroid, embeds)[0]
        df_c = df_c.copy()
        df_c["similarity"] = sims

        # Top tweets for labeling
        top_tweets = df_c.sort_values("similarity", ascending=False).head(TOP_N_TWEETS)
        tweet_list = "\n".join([f"- {t}" for t in top_tweets["clean_text"]])

        # Gemini prompt
        payload = {
            "system_instruction": {
                "parts": [{
                    "text": (
                        "Provide a short, general topic label for this group of tweets. "
                        "Do NOT reference any airline. "
                        "Max 5 words, generic theme only."
                    )
                }]
            },
            "contents": [
                {"parts": [{"text": tweet_list}]}
            ]
        }

        label = call_gemini_api(payload)
        cluster_labels[k] = label
        print(f"\nCluster {k} label → {label}")

        # Sentiment
        df_c["sentiment"] = df_c["clean_text"].apply(get_sentiment)

        pos = int((df_c["sentiment"] == "Positive").sum())
        neg = int((df_c["sentiment"] == "Negative").sum())
        neu = int((df_c["sentiment"] == "Neutral").sum())

        # Overall totals
        overall_positive += pos
        overall_negative += neg
        overall_neutral += neu

        cluster_sentiment_score = pos - neg

        # Random samples
        samples = df_c["clean_text"].sample(
            min(N_RANDOM_SAMPLES, len(df_c)), random_state=42
        ).tolist()

        # Store in summary
        cluster_summary[k] = {
            "topic_label": label,
            "count": int(len(df_c)),
            "sentiment_distribution": {
                "Positive": pos,
                "Negative": neg,
                "Neutral": neu
            },
            "cluster_sentiment_score": cluster_sentiment_score,
            "random_samples": samples
        }

    # JSON-safe
    cluster_summary_safe = to_json_safe(cluster_summary)
    overall_sentiment_safe = to_json_safe({
        "overall_positive": overall_positive,
        "overall_negative": overall_negative,
        "overall_neutral": overall_neutral
    })

    with open("cluster_summary.json", "w") as f:
        json.dump(cluster_summary_safe, f, indent=4)

    with open("overall_sentiment.json", "w") as f:
        json.dump(overall_sentiment_safe, f, indent=4)

    print("\n=== OVERALL SENTIMENT ===")
    print("Positive:", overall_positive)
    print("Negative:", overall_negative)
    print("Neutral:", overall_neutral)

    return cluster_labels

# ===============================
# Main Runner
# ===============================
def main():

    embeddings_df = load_and_process_embeddings(EMBEDDINGS_FILE)
    tweets_df = load_original_tweets(CSV_FILE)
    N = 10  # Number of clusters

    clustered_df, X, cluster_means = perform_gmm_clustering(embeddings_df, N)

    labels = label_clusters_with_gemini(clustered_df, tweets_df, cluster_means, N)

    clustered_df["topic_label"] = clustered_df["cluster_label"].map(labels)
    clustered_df.to_csv("clustered_tweets_with_labels.csv", index=False)

    print("\nFinal Labels:")
    for k, lbl in labels.items():
        print(f"Cluster {k}: {lbl}")

# ===============================
# Run
# ===============================
if __name__ == "__main__":
    main()
