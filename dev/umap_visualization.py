import pandas as pd
import json
import numpy as np
import os
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
import umap
import plotly.graph_objects as go
import plotly.express as px

# Configuration
EMBEDDINGS_FILE = "tweet_embeddings.json"
CSV_FILE = "Tweets_clean.csv"
CLUSTER_SUMMARY_FILE = "cluster_summary.json"
OUTPUT_HTML = "cluster_umap_visualization.html"
OUTPUT_METRICS = "cluster_metrics.json"
N_CLUSTERS = 10


def load_embeddings(file_path):
    """Load embeddings from JSON file."""
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None

    with open(file_path, "r") as f:
        data = json.load(f)

    if "embeddings" not in data:
        print("JSON missing 'embeddings'")
        return None

    embeddings_dict = data["embeddings"]

    df = pd.DataFrame(
        {
            "tweet_id": list(embeddings_dict.keys()),
            "embedding_vector": list(embeddings_dict.values()),
        }
    )

    print(f"Loaded {len(df)} embeddings")
    return df


def load_tweets(csv_path):
    """Load tweet data including sentiment labels."""
    if not os.path.exists(csv_path):
        print(f"CSV not found: {csv_path}")
        return None

    df = pd.read_csv(csv_path)
    df["tweet_id"] = df["tweet_id"].astype(str)
    return df[["tweet_id", "clean_text", "airline_sentiment"]]


def load_cluster_summary(file_path):
    """Load cluster summary to get topic labels."""
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None

    with open(file_path, "r") as f:
        cluster_summary = json.load(f)

    # Create mapping from cluster number to topic label
    cluster_labels = {int(k): v["topic_label"] for k, v in cluster_summary.items()}
    return cluster_labels


def recreate_clusters(embeddings_df, n_components=10):
    """Recreate cluster assignments using GMM clustering."""
    print(f"Performing GMM clustering with {n_components} clusters...")

    X = np.array(embeddings_df["embedding_vector"].tolist())

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit GMM
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    cluster_labels = gmm.fit_predict(X_scaled)

    embeddings_df = embeddings_df.copy()
    embeddings_df["cluster_label"] = cluster_labels

    print(f"Clustering complete. Cluster distribution:")
    print(embeddings_df["cluster_label"].value_counts().sort_index())

    return embeddings_df, X


def calculate_purity(cluster_labels, true_labels):
    """Calculate cluster purity score.

    Purity = (1/N) * Σ(max_j |C_i ∩ L_j|)
    where C_i are clusters and L_j are true label classes.
    """
    n_samples = len(cluster_labels)
    clusters = np.unique(cluster_labels)
    classes = np.unique(true_labels)

    purity_sum = 0

    for cluster_id in clusters:
        # Get all samples in this cluster
        cluster_mask = cluster_labels == cluster_id
        cluster_samples = true_labels[cluster_mask]

        if len(cluster_samples) == 0:
            continue

        # Count occurrences of each class in this cluster
        class_counts = {}
        for class_label in classes:
            class_counts[class_label] = np.sum(cluster_samples == class_label)

        # Find the most frequent class in this cluster
        max_count = max(class_counts.values())
        purity_sum += max_count

    purity = purity_sum / n_samples
    return purity


def calculate_metrics(embeddings, cluster_labels, true_labels):
    """Calculate all cluster quality metrics."""
    print("\nCalculating cluster quality metrics...")

    metrics = {}

    # Silhouette Score (higher is better, range: -1 to 1)
    try:
        silhouette = silhouette_score(embeddings, cluster_labels)
        metrics["silhouette_score"] = float(silhouette)
        print(f"Silhouette Score: {silhouette:.4f} (higher is better, range: -1 to 1)")
    except Exception as e:
        print(f"Error calculating Silhouette Score: {e}")
        metrics["silhouette_score"] = None

    # Davies-Bouldin Index (lower is better)
    try:
        db_index = davies_bouldin_score(embeddings, cluster_labels)
        metrics["davies_bouldin_index"] = float(db_index)
        print(f"Davies-Bouldin Index: {db_index:.4f} (lower is better)")
    except Exception as e:
        print(f"Error calculating Davies-Bouldin Index: {e}")
        metrics["davies_bouldin_index"] = None

    # Calinski-Harabasz Score (higher is better)
    try:
        ch_score = calinski_harabasz_score(embeddings, cluster_labels)
        metrics["calinski_harabasz_score"] = float(ch_score)
        print(f"Calinski-Harabasz Score: {ch_score:.4f} (higher is better)")
    except Exception as e:
        print(f"Error calculating Calinski-Harabasz Score: {e}")
        metrics["calinski_harabasz_score"] = None

    # Purity Score (higher is better, range: 0 to 1)
    try:
        purity = calculate_purity(cluster_labels, true_labels)
        metrics["purity_score"] = float(purity)
        print(f"Purity Score: {purity:.4f} (higher is better, range: 0 to 1)")
    except Exception as e:
        print(f"Error calculating Purity Score: {e}")
        metrics["purity_score"] = None

    return metrics


def create_umap_visualization(df_merged, cluster_labels_dict, output_path):
    """Create interactive Plotly visualization with UMAP coordinates."""
    print("\nCreating UMAP visualization...")

    # Prepare data for plotting
    df_plot = df_merged.copy()
    df_plot["cluster_topic"] = df_plot["cluster_label"].map(cluster_labels_dict)
    df_plot["cluster_label_str"] = df_plot["cluster_label"].astype(str)

    # Create color palette for clusters
    n_clusters = len(df_plot["cluster_label"].unique())
    colors = px.colors.qualitative.Set3[:n_clusters]
    if n_clusters > len(colors):
        # Extend color palette if needed
        import itertools

        colors = list(
            itertools.islice(itertools.cycle(px.colors.qualitative.Set3), n_clusters)
        )

    color_map = {i: colors[i % len(colors)] for i in range(n_clusters)}
    df_plot["color"] = df_plot["cluster_label"].map(color_map)

    # Create hover text
    df_plot["hover_text"] = (
        "Tweet ID: "
        + df_plot["tweet_id"]
        + "<br>"
        + "Cluster: "
        + df_plot["cluster_label_str"]
        + " - "
        + df_plot["cluster_topic"]
        + "<br>"
        + "Sentiment: "
        + df_plot["airline_sentiment"]
        + "<br>"
        + "Text: "
        + df_plot["clean_text"].str[:100]
        + "..."
    )

    # Create scatter plot
    fig = go.Figure()

    # Plot each cluster separately for better legend control
    for cluster_id in sorted(df_plot["cluster_label"].unique()):
        cluster_data = df_plot[df_plot["cluster_label"] == cluster_id]
        cluster_topic = cluster_labels_dict.get(cluster_id, f"Cluster {cluster_id}")

        fig.add_trace(
            go.Scatter(
                x=cluster_data["umap_x"],
                y=cluster_data["umap_y"],
                mode="markers",
                name=f"Cluster {cluster_id}: {cluster_topic}",
                marker=dict(
                    size=4,
                    color=color_map[cluster_id],
                    opacity=0.7,
                    line=dict(width=0.5, color="white"),
                ),
                text=cluster_data["hover_text"],
                hoverinfo="text",
                hovertemplate="%{text}<extra></extra>",
            )
        )

    # Update layout
    fig.update_layout(
        title={
            "text": "Tweet Clusters - UMAP 2D Visualization",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 20},
        },
        xaxis_title="UMAP Dimension 1",
        yaxis_title="UMAP Dimension 2",
        width=1200,
        height=800,
        hovermode="closest",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.01, font=dict(size=10)),
        template="plotly_white",
    )

    # Save to HTML
    fig.write_html(output_path)
    print(f"Visualization saved to {output_path}")


def main():
    """Main function to orchestrate the process."""
    print("=" * 60)
    print("UMAP Visualization and Cluster Quality Evaluation")
    print("=" * 60)

    # Get file paths relative to project root
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    embeddings_path = os.path.join(base_dir, EMBEDDINGS_FILE)
    csv_path = os.path.join(base_dir, CSV_FILE)
    cluster_summary_path = os.path.join(base_dir, CLUSTER_SUMMARY_FILE)
    output_html_path = os.path.join(base_dir, OUTPUT_HTML)
    output_metrics_path = os.path.join(base_dir, OUTPUT_METRICS)

    # Load data
    print("\n1. Loading data...")
    embeddings_df = load_embeddings(embeddings_path)
    if embeddings_df is None:
        return

    tweets_df = load_tweets(csv_path)
    if tweets_df is None:
        return

    cluster_labels_dict = load_cluster_summary(cluster_summary_path)
    if cluster_labels_dict is None:
        return

    # Recreate clusters
    print("\n2. Recreating cluster assignments...")
    clustered_df, embeddings_array = recreate_clusters(embeddings_df, N_CLUSTERS)

    # Merge with tweet data
    df_merged = pd.merge(clustered_df, tweets_df, on="tweet_id", how="inner")
    print(f"Merged data: {len(df_merged)} tweets")

    # Apply UMAP
    print("\n3. Applying UMAP dimensionality reduction...")
    reducer = umap.UMAP(
        n_neighbors=15, min_dist=0.1, n_components=2, metric="cosine", random_state=42
    )

    umap_embedding = reducer.fit_transform(embeddings_array)
    df_merged["umap_x"] = umap_embedding[:, 0]
    df_merged["umap_y"] = umap_embedding[:, 1]
    print("UMAP transformation complete")

    # Calculate metrics
    print("\n4. Calculating cluster quality metrics...")
    # Prepare true labels for purity (sentiment labels)
    true_labels = df_merged["airline_sentiment"].values

    metrics = calculate_metrics(
        embeddings_array, df_merged["cluster_label"].values, true_labels
    )

    # Save metrics
    with open(output_metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {output_metrics_path}")

    # Create visualization
    print("\n5. Creating interactive visualization...")
    create_umap_visualization(df_merged, cluster_labels_dict, output_html_path)

    print("\n" + "=" * 60)
    print("Process complete!")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"  - Visualization: {output_html_path}")
    print(f"  - Metrics: {output_metrics_path}")


if __name__ == "__main__":
    main()
