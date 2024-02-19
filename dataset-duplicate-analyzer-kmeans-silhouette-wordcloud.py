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
    """Process texts in batches and return their vectors."""
    cache_file = os.path.join(cache_dir, "vectors_cache.pkl")
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            cache_data = pickle.load(f)
        print("Cached vectors loaded.")
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
        print("Vectors cached.")
    
    return vectors, vector_details



def find_optimal_clusters(X, max_k=10, method='silhouette'):
    """Determine the optimal number of clusters using specified method."""
    scores = {'silhouette': [], 'davies_bouldin': [], 'inertia': []}
    K = range(2, max_k+1) 

    for k in K:
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(X)
        labels = km.labels_
        scores['inertia'].append(km.inertia_)

        if k > 1:
            silhouette_avg = silhouette_score(X, labels)
            db_index = davies_bouldin_score(X, labels)
            scores['silhouette'].append(silhouette_avg)
            scores['davies_bouldin'].append(db_index)


    if method == 'elbow':
        optimal_k = 1 + scores['inertia'].index(min(scores['inertia']))
    elif method == 'silhouette':
        optimal_k = 2 + scores['silhouette'].index(max(scores['silhouette']))
    else: 
        optimal_k = 2 + scores['davies_bouldin'].index(min(scores['davies_bouldin']))

    return optimal_k


def analyze_clusters(X, labels):

    centroids = [np.mean(X[labels == i], axis=0) for i in range(max(labels) + 1)]
    closest, _ = pairwise_distances_argmin_min(centroids, X)
    duplicates = np.zeros(labels.shape[0], dtype=bool)
    for i, center in enumerate(closest):
        duplicates[center] = True  
    return duplicates.sum()


def cluster_responses(input_file_path, field_name, show_full_line=False, use_elbow_method=False, use_silhouette_analysis=False, use_davies_bouldin=False, output_file=None):
    total_lines = 0
    read_lines = 0
    skipped_lines = 0
    lines = []
    responses = []

    label_response_map = {}

    with open(input_file_path, 'r', encoding='utf-8') as file:
        for line in tqdm(file, desc="Reading File"):
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
            except json.JSONDecodeError as e:
                skipped_lines += 1
    
    vectors, vector_details = batch_process_texts(responses, batch_size=100)
    X = np.array(vectors)

    # very rebust to handle all cases
    if use_elbow_method:
        optimal_clusters = find_optimal_clusters(X, method='elbow')
    elif use_silhouette_analysis:
        optimal_clusters = find_optimal_clusters(X, method='silhouette')
    elif use_davies_bouldin:
        optimal_clusters = find_optimal_clusters(X, method='davies_bouldin')
    else:
        # If no method is specified, default to silhouette analysis or any preferred method
        optimal_clusters = find_optimal_clusters(X, method='silhouette')

    kmeans = KMeans(n_clusters=optimal_clusters, random_state=42).fit(X)

    for label, response_id in zip(kmeans.labels_, range(len(responses))):
        label_response_map.setdefault(label, []).append(response_id)

    silhouette_avg, db_index = None, None
    if use_silhouette_analysis and optimal_clusters > 1:
        silhouette_avg = silhouette_score(X, kmeans.labels_)
    if use_davies_bouldin and optimal_clusters > 1:
        db_index = davies_bouldin_score(X, kmeans.labels_)

    if output_file:
        generate_insightful_output(kmeans.labels_, X, kmeans.cluster_centers_, vectors, responses, lines, output_file, silhouette_avg, db_index, label_response_map, use_silhouette_analysis, use_davies_bouldin, show_full_line)

    all_labels = set(np.unique(kmeans.labels_))
    labels_in_map = set(label_response_map.keys())
    labels_not_in_map = all_labels - labels_in_map
    print("Labels not in label_response_map:", labels_not_in_map)
    print("Label Response Map: ", label_response_map)

    return label_response_map

def generate_silhouette_plots(X, labels, silhouette_avg):
    import matplotlib.cm as cm
    n_clusters = len(set(labels))
    fig, ax1 = plt.subplots(1, 1)
    fig.set_size_inches(18, 7)
    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])
    y_lower = 10
    for i in range(n_clusters):
        ith_cluster_silhouette_values = silhouette_samples(X, labels)[labels == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, facecolor=color, edgecolor=color, alpha=0.7)
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10
    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax1.set_yticks([])
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.suptitle("Silhouette analysis for KMeans clustering on sample data with n_clusters = %d" % n_clusters, fontsize=14, fontweight='bold')
    plt.show()

def generate_word_clouds_for_clusters(labels, responses):
    unique_labels = set(labels)
    for label in unique_labels:
        cluster_responses = [responses[i] for i, lbl in enumerate(labels) if lbl == label]
        text = ' '.join(cluster_responses)
        wordcloud = WordCloud(max_font_size=50, max_words=100, background_color='white').generate(text)
        plt.figure(figsize=(10, 7))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Word Cloud for Cluster {label + 1}')
        plt.show()

def generate_insightful_output(labels, X, cluster_centers, vectors, responses, lines, output_file, silhouette_avg, db_index, label_response_map, use_silhouette_analysis, use_davies_bouldin, show_full_line):
    assert len(labels) == len(responses), f"Mismatch in labels and responses length: {len(labels)} labels vs {len(responses)} responses."

    print("Labels (KMeans):", labels, "Type:", type(labels))
    print("X:", X, "Type:", type(X))
    print("Cluster Centers:", cluster_centers, "Type:", type(cluster_centers))
    print("Vectors:", vectors, "Type:", type(vectors))
    print("Output File:", output_file, "Type:", type(output_file))
    print("Silhouette Avg:", silhouette_avg, "Type:", type(silhouette_avg))
    print("DB Index:", db_index, "Type:", type(db_index))

    # Silhouette Plot
    if use_silhouette_analysis and silhouette_avg is not None:
        plt.figure(figsize=(10, 7))
        silhouette_vals = silhouette_samples(X, labels)
        y_lower = 0
        for i, cluster in enumerate(np.unique(labels)):
            cluster_silhouette_vals = silhouette_vals[labels == cluster]
            cluster_silhouette_vals.sort()
            y_upper = y_lower + len(cluster_silhouette_vals)
            color = plt.cm.nipy_spectral(float(i) / len(np.unique(labels)))
            plt.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_silhouette_vals, facecolor=color, edgecolor=color, alpha=0.7)
            plt.text(-0.05, y_lower + 0.5 * len(cluster_silhouette_vals), str(i))
            y_lower = y_upper + 10  # 10 for the 0 samples
        plt.axvline(x=silhouette_avg, color="red", linestyle="--")
        plt.xlabel('Silhouette coefficient values')
        plt.ylabel('Cluster labels')
        plt.title('Silhouette plot for the various clusters')
        plt.show()

    # Word Clouds 
    for i, cluster in enumerate(np.unique(labels)):
        cluster_responses = np.array(responses)[labels == cluster]
        text = " ".join(cluster_responses)
        wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(text)
        plt.figure(figsize=(10, 7))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title(f'Word Cloud for Cluster {i + 1}')
        plt.show()


    detailed_df = pd.DataFrame(columns=["Line Number", "Cluster ID", "Distance to Centroid", "Potential Duplicate", "Text"])
    for label in np.unique(labels):
        cluster_indices = np.where(labels == label)[0]
        cluster_vectors = X[cluster_indices]
        distances_to_centroid = np.linalg.norm(cluster_vectors - cluster_centers[label], axis=1)
        new_rows = [{
            "Line Number": idx + 1,
            "Cluster ID": f"Cluster {label}",
            "Distance to Centroid": distances_to_centroid[i],
            "Potential Duplicate": "Yes" if idx == np.argmin(distances_to_centroid) else "No",
            "Text": responses[idx] if not show_full_line else lines[idx]  
        } for i, idx in enumerate(cluster_indices)]
        detailed_df = pd.concat([detailed_df, pd.DataFrame(new_rows)], ignore_index=True)
    
    detailed_df.sort_values(by=["Cluster ID", "Distance to Centroid"], inplace=True)

    # Writing to Output File
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("Detailed Cluster Analysis Report\n")
            f.write("=================================\n\n")
            f.write(f"Total Clusters: {len(np.unique(labels))}\n")
            f.write(f"Silhouette Score: {silhouette_avg if silhouette_avg is not None else 'N/A'}\n")
            if use_davies_bouldin and db_index is not None:
                f.write(f"Davies-Bouldin Index: {db_index if db_index is not None else 'N/A'}\n\n")
            f.write("Cluster Summaries\n")
            f.write("-----------------\n")
            for label in np.unique(labels):
                cluster_summary = detailed_df[detailed_df["Cluster ID"] == f"Cluster {label}"]
                f.write(f"{cluster_summary['Cluster ID'].iloc[0]} - Items: {len(cluster_summary)}\n")
                if len(cluster_summary[cluster_summary["Potential Duplicate"] == "Yes"]) > 0:
                    f.write("  Potential Duplicates: Yes\n")
                f.write("\n")
            f.write("Detailed Insights\n")
            f.write("-----------------\n")
            f.write(detailed_df.to_string(index=False))
            f.write("\n\nEnd of Report\n")
    print(f"Detailed analysis exported to {output_file}")



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
