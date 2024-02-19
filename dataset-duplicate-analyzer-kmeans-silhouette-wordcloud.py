# Dataset Analyzer using a number of ML Tools - WIP
# (c) 2024 Ben Gorlick | ben@unifiedlearning.ai | MIT LICENSE | USE AT YOUR OWN RISK
import json
import numpy as np
import spacy
import argparse
import logging
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, pairwise_distances_argmin_min
from rainbow_tqdm import tqdm
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_samples
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import logging
import pickle
from pathlib import Path
import os

nlp = spacy.load("en_core_web_md", disable=["parser", "ner"])

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def batch_process_texts(input_texts, cache_dir="datagen-cache", batch_size=100):
    cache_file = os.path.join(cache_dir, "vectors_cache.pkl")
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            cache_data = pickle.load(f)
        logger.debug("Cached vectors loaded.")
        vectors = cache_data['vectors']
        vector_details = cache_data.get('vector_details', [])
    else:
        vectors = []
        vector_details = []
        for batch in tqdm([input_texts[i:i + batch_size] for i in range(0, len(input_texts), batch_size)], desc="Processing Batches"):
            docs = list(nlp.pipe(batch))
            batch_vectors = [doc.vector for doc in docs]  
            vectors.extend(batch_vectors)

            for i, doc in enumerate(docs):
                token_details = [{'text': token.text, 'vector': token.vector} for token in doc]
                vector_details.append({'text': batch[i], 'tokens': token_details})
        
        cache_data = {'vectors': vectors, 'vector_details': vector_details}
        with open(cache_file, "wb") as f:
            pickle.dump(cache_data, f)
        logger.debug("Vectors cached.")
    
    return vectors, vector_details

def find_optimal_clusters(X, max_k=10, method='silhouette'):
    scores = {'silhouette': [], 'davies_bouldin': [], 'inertia': []}
    K = range(2, max_k+1) 

    for k in K:
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(X)
        labels = km.labels_
        scores['inertia'].append(km.inertia_)

        if len(set(labels)) > 1:  # Ensure there are at least 2 clusters for valid silhouette calculation
            silhouette_avg = silhouette_score(X, labels)
            db_index = davies_bouldin_score(X, labels)
            scores['silhouette'].append(silhouette_avg)
            scores['davies_bouldin'].append(db_index)

    if method == 'elbow':
        optimal_k = K[scores['inertia'].index(min(scores['inertia']))]
    elif method == 'silhouette':
        optimal_k = K[scores['silhouette'].index(max(scores['silhouette']))]
    else:  # davies_bouldin
        optimal_k = K[scores['davies_bouldin'].index(min(scores['davies_bouldin']))]

    return optimal_k

def cluster_responses(input_file_path, field_name, show_full_line=False, use_elbow_method=False, use_silhouette_analysis=False, use_davies_bouldin=False, output_file=None):
    total_lines = 0
    read_lines = 0
    skipped_lines = 0
    lines = []
    responses = []

    label_response_map = {}

    with open(input_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            total_lines += 1
            try:
                data = json.loads(line)
                response = data.get(field_name, "")
                if response:
                    lines.append(line.strip() if show_full_line else "")
                    responses.append(response)
                    read_lines += 1
                else:
                    skipped_lines += 1
            except json.JSONDecodeError:
                skipped_lines += 1

    print(f"Total lines: {total_lines}, Read lines: {read_lines}, Skipped lines: {skipped_lines}")

    vectors, vector_details = batch_process_texts(responses)
    X = np.array(vectors)

    if use_elbow_method:
        optimal_clusters = find_optimal_clusters(X, method='elbow')
    elif use_silhouette_analysis:
        optimal_clusters = find_optimal_clusters(X, method='silhouette')
    elif use_davies_bouldin:
        optimal_clusters = find_optimal_clusters(X, method='davies_bouldin')
    else:
        optimal_clusters = find_optimal_clusters(X, method='silhouette')

    kmeans = KMeans(n_clusters=optimal_clusters, random_state=42).fit(X)
    labels = kmeans.labels_

    for label, response_id in zip(labels, range(len(responses))):
        label_response_map.setdefault(label, []).append(response_id)

    silhouette_avg = silhouette_score(X, kmeans.labels_) if use_silhouette_analysis else None
    db_index = davies_bouldin_score(X, kmeans.labels_) if use_davies_bouldin else None


    if output_file:
        generate_insightful_output(kmeans.labels_, X, kmeans.cluster_centers_, vectors, responses, lines, output_file, silhouette_avg, db_index, label_response_map, use_silhouette_analysis, use_davies_bouldin, show_full_line)

    all_labels = set(np.unique(kmeans.labels_))
    labels_in_map = set(label_response_map.keys())
    labels_not_in_map = all_labels - labels_in_map
    print("Labels not in label_response_map:", labels_not_in_map)
    print("Label Response Map: ", label_response_map)

    return label_response_map

def generate_silhouette_plots(X, labels, silhouette_avg):
    from matplotlib import cm
    import matplotlib.pyplot as plt
    import numpy as np

    print("Generating silhouette plots.")
    n_clusters = len(np.unique(labels))
    fig, ax1 = plt.subplots(1)
    fig.set_size_inches(18, 7)

    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    silhouette_vals = silhouette_samples(X, labels)
    y_lower = 10
    for i in range(n_clusters):
        ith_cluster_silhouette_values = silhouette_vals[labels == i]
        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster")

    plt.suptitle(f"Silhouette analysis for KMeans clustering on sample data with n_clusters = {n_clusters}", fontsize=14, fontweight='bold')
    plt.show()

def generate_word_clouds_for_clusters(labels, responses):
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt

    print("Generating word clouds for each cluster.")
    unique_labels = set(labels)
    for label in unique_labels:
        texts = [responses[i] for i, lbl in enumerate(labels) if lbl == label]
        wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(" ".join(texts))

        plt.figure(figsize=(10, 7))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.title(f"Cluster {label} Word Cloud")
        plt.show()

def generate_insightful_output(labels, X, cluster_centers, vectors, responses, lines, output_file, silhouette_avg, db_index, label_response_map, use_silhouette_analysis, use_davies_bouldin, show_full_line):
    # Validate matching lengths of labels and responses to ensure accurate analysis
    assert len(labels) == len(responses), f"Mismatch in labels and responses length: {len(labels)} labels vs {len(responses)} responses."

    # Log basic info about the clustering
    print("Generating detailed analysis output...")
    print(f"Number of clusters: {len(np.unique(labels))}")
    if silhouette_avg is not None:
        print(f"Silhouette average: {silhouette_avg}")
    if db_index is not None:
        print(f"Davies-Bouldin index: {db_index}")

    # Generate silhouette plots if requested and applicable
    if use_silhouette_analysis and silhouette_avg is not None:
        print("Generating silhouette plots for cluster analysis.")
        generate_silhouette_plots(X, labels, silhouette_avg)

    # Generate word clouds for each cluster
    print("Generating word clouds for each cluster.")
    generate_word_clouds_for_clusters(labels, responses)

    # Prepare DataFrame for detailed insights
    detailed_insights = []
    for i, label in enumerate(np.unique(labels)):
        cluster_indices = np.where(labels == label)[0]
        texts = [responses[idx] for idx in cluster_indices]
        for idx in cluster_indices:
            detailed_insights.append({
                "Cluster": label,
                "Text": lines[idx] if show_full_line else responses[idx],
                # Additional metrics can be included here, such as distances to centroid
            })

    # Convert insights to a DataFrame for easier manipulation and analysis
    df_insights = pd.DataFrame(detailed_insights)

    # Output to console or file as per the request
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            # Header
            f.write("Detailed Cluster Analysis Report\n\n")
            # Summary statistics
            f.write(f"Total Clusters: {len(np.unique(labels))}\n")
            f.write(f"Silhouette Score: {silhouette_avg if silhouette_avg is not None else 'N/A'}\n")
            f.write(f"Davies-Bouldin Index: {db_index if db_index is not None else 'N/A'}\n\n")
            # Detailed insights per cluster
            for label in np.unique(labels):
                f.write(f"Cluster {label} Details:\n")
                cluster_df = df_insights[df_insights['Cluster'] == label]
                for _, row in cluster_df.iterrows():
                    f.write(f"{row['Text']}\n")
                f.write("\n")
            print(f"Detailed analysis exported to {output_file}")
    else:
        # Optionally, print insights directly if no output file is specified
        print("Detailed Cluster Analysis:")
        print(df_insights.to_string(index=False))

    # Indicate completion of the function's execution
    print("Detailed cluster analysis generation complete.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Cluster responses in a JSONL file focusing on advanced clustering analysis.')
    parser.add_argument('input_file', type=str, help='Input JSONL file path')
    parser.add_argument('--field', type=str, default='response', help='JSON field to cluster (default: "response")')
    parser.add_argument('--show_full_line', action='store_true', help='Show full lines in output')
    parser.add_argument('--use_elbow_method', action='store_true', help='Use the elbow method to determine the optimal number of clusters')
    parser.add_argument('--use_silhouette_analysis', action='store_true', help='Perform silhouette analysis to evaluate clustering quality')
    parser.add_argument('--use_davies_bouldin', action='store_true', help='Use Davies-Bouldin index to evaluate clustering quality')
    parser.add_argument('--output_file', type=str, help='Output file path for exporting analysis', default=None)
    args = parser.parse_args()


cluster_responses(args.input_file, args.field, args.show_full_line, args.use_elbow_method, args.use_silhouette_analysis, args.use_davies_bouldin, args.output_file)
