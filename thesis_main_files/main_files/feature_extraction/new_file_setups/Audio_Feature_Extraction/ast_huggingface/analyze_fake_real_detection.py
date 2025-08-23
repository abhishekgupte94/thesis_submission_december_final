"""
AST Feature Analysis Script for Fake vs Real Audio Detection
Test the meaningfulness of extracted AST features for distinguishing fake/real audio
Using cosine similarity, t-SNE, clustering, and fake/real classification analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from pathlib import Path
from scipy.spatial.distance import cosine, pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd

def load_separate_fake_real_features(fake_path, real_path):
    """Load fake and real features separately and combine them"""
    # Load fake features
    fake_data = np.load(fake_path + ".npz")
    with open(fake_path + ".meta.json", 'r') as f:
        fake_metadata = json.load(f)

    fake_features = fake_data["features"]
    fake_time_axis = fake_data["time_axis"] if "time_axis" in fake_data else None
    fake_paths = fake_data["paths"]
    fake_filenames = [Path(p).name for p in fake_paths]

    # Load real features
    real_data = np.load(real_path + ".npz")
    with open(real_path + ".meta.json", 'r') as f:
        real_metadata = json.load(f)

    real_features = real_data["features"]
    real_time_axis = real_data["time_axis"] if "time_axis" in real_data else None
    real_paths = real_data["paths"]
    real_filenames = [Path(p).name for p in real_paths]

    # Combine features
    combined_features = np.concatenate([fake_features, real_features], axis=0)
    combined_filenames = fake_filenames + real_filenames
    combined_paths = list(fake_paths) + list(real_paths)

    # Use time axis from fake (should be identical to real)
    time_axis = fake_time_axis if fake_time_axis is not None else real_time_axis

    # Create combined metadata
    combined_metadata = fake_metadata.copy()
    combined_metadata["combined_from"] = {
        "evaluate_files": len(fake_filenames),
        "real_files": len(real_filenames),
        "fake_source": fake_path,
        "real_source": real_path
    }

    return combined_features, time_axis, combined_filenames, combined_metadata

def load_ast_features(base_path):
    """Load AST features and metadata from base path (without extensions)"""
    npz_path = base_path + ".npz"
    meta_path = base_path + ".meta.json"

    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Features file not found: {npz_path}")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    data = np.load(npz_path)
    with open(meta_path, 'r') as f:
        metadata = json.load(f)

    features = data["features"]  # (n_files, n_time_patches, n_features)
    time_axis = data["time_axis"] if "time_axis" in data else None
    file_paths = data["paths"]

    # Extract just filenames for cleaner display
    filenames = [Path(p).name for p in file_paths]

    return features, time_axis, filenames, metadata

def parse_fake_real_labels(filenames):
    """Parse fake/real labels from filenames"""
    labels = []
    label_names = []

    for filename in filenames:
        if 'fake' in filename.lower():
            labels.append(0)  # 0 for fake
            label_names.append('fake')
        elif 'real' in filename.lower():
            labels.append(1)  # 1 for real
            label_names.append('real')
        else:
            # If unclear, try to infer from context or position
            # Since we load fake files first, then real files
            labels.append(-1)  # Unknown
            label_names.append('unknown')

    return np.array(labels), label_names

def compute_global_features(features):
    """Compute global (file-level) features from temporal features"""
    global_features = {}

    # Statistical aggregations over time dimension
    global_features['mean'] = np.mean(features, axis=1)  # (n_files, n_features)
    global_features['std'] = np.std(features, axis=1)
    global_features['max'] = np.max(features, axis=1)
    global_features['min'] = np.min(features, axis=1)
    global_features['median'] = np.median(features, axis=1)

    # Temporal dynamics - these might be key for fake detection
    global_features['range'] = global_features['max'] - global_features['min']
    global_features['temporal_variance'] = np.var(np.mean(features, axis=2), axis=1, keepdims=True)

    # Spectral stability measures
    temporal_means = np.mean(features, axis=2)  # (n_files, n_time)
    global_features['temporal_smoothness'] = np.array([
        np.mean([np.corrcoef(temporal_means[i, :-1], temporal_means[i, 1:])[0,1]])
        for i in range(features.shape[0])
    ]).reshape(-1, 1)

    return global_features

def fake_real_classification_test(features, labels, filenames):
    """Test how well features can distinguish fake from real audio"""
    print("=" * 60)
    print("🤖 FAKE vs REAL CLASSIFICATION TEST")
    print("=" * 60)

    # Filter out unknown labels
    valid_mask = labels >= 0
    features_clean = features[valid_mask]
    labels_clean = labels[valid_mask]
    filenames_clean = [f for f, v in zip(filenames, valid_mask) if v]

    if len(np.unique(labels_clean)) < 2:
        print("❌ Need both fake and real samples for classification test")
        return None

    print(f"📊 Dataset composition:")
    fake_count = np.sum(labels_clean == 0)
    real_count = np.sum(labels_clean == 1)
    print(f"   Fake samples: {fake_count}")
    print(f"   Real samples: {real_count}")
    print(f"   Total samples: {len(labels_clean)}")

    # Check if we have enough samples for cross-validation
    min_class_size = min(fake_count, real_count)

    if min_class_size < 2:
        print("❌ Need at least 2 samples per class for meaningful analysis")
        return None

    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_clean)

    # For small datasets, use simpler analysis
    if len(labels_clean) <= 6:
        print(f"\n🔍 Small dataset detected - using simple train/test analysis:")

        # Simple classifiers for small datasets
        classifiers = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=10, random_state=42)
        }

        results = {}
        for name, clf in classifiers.items():
            try:
                # Fit on all data and test on same data (overfitting check)
                clf.fit(features_scaled, labels_clean)
                predictions = clf.predict(features_scaled)
                accuracy = np.mean(predictions == labels_clean)
                results[name] = [accuracy]  # Single score instead of CV scores
                print(f"   {name}: {accuracy:.4f} (training accuracy)")
            except Exception as e:
                print(f"   {name}: Failed ({str(e)})")

        # Show individual predictions
        if results:
            best_classifier_name = max(results.keys(), key=lambda k: results[k][0])
            best_classifier = classifiers[best_classifier_name]
            best_classifier.fit(features_scaled, labels_clean)
            predictions = best_classifier.predict(features_scaled)

            print(f"\n🏆 Best classifier: {best_classifier_name}")
            print(f"   Training accuracy: {np.mean(predictions == labels_clean):.4f}")

            # Detailed predictions
            print(f"\n📋 Individual predictions:")
            for i, (filename, true_label, pred_label) in enumerate(zip(filenames_clean, labels_clean, predictions)):
                true_str = "real" if true_label == 1 else "fake"
                pred_str = "real" if pred_label == 1 else "fake"
                correct = "✅" if true_label == pred_label else "❌"
                print(f"   {correct} {filename}: True={true_str}, Predicted={pred_str}")

    else:
        # Original cross-validation approach for larger datasets
        classifiers = {
            'SVM (RBF)': SVC(kernel='rbf', random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42)
        }

        results = {}
        cv_folds = min(min_class_size, 3)  # Use fewer folds for small datasets
        print(f"\n🎯 Cross-validation results ({cv_folds}-fold):")

        for name, clf in classifiers.items():
            try:
                scores = cross_val_score(clf, features_scaled, labels_clean, cv=cv_folds, scoring='accuracy')
                results[name] = scores
                print(f"   {name}: {scores.mean():.4f} (±{scores.std()*2:.4f})")
            except Exception as e:
                print(f"   {name}: Failed ({str(e)})")

        # Fit best classifier and analyze predictions
        if results:
            best_classifier_name = max(results.keys(), key=lambda k: results[k].mean())
            best_classifier = classifiers[best_classifier_name]
            best_classifier.fit(features_scaled, labels_clean)
            predictions = best_classifier.predict(features_scaled)

            print(f"\n🏆 Best classifier: {best_classifier_name}")
            print(f"   Training accuracy: {np.mean(predictions == labels_clean):.4f}")

            # Detailed predictions
            print(f"\n📋 Individual predictions:")
            for i, (filename, true_label, pred_label) in enumerate(zip(filenames_clean, labels_clean, predictions)):
                true_str = "real" if true_label == 1 else "fake"
                pred_str = "real" if pred_label == 1 else "fake"
                correct = "✅" if true_label == pred_label else "❌"
                print(f"   {correct} {filename}: True={true_str}, Predicted={pred_str}")

    return results

def fake_real_similarity_analysis(features, labels, filenames):
    """Analyze similarities within and between fake/real groups"""
    print("\n" + "=" * 60)
    print("🔍 FAKE vs REAL SIMILARITY ANALYSIS")
    print("=" * 60)

    # Filter valid labels
    valid_mask = labels >= 0
    features_clean = features[valid_mask]
    labels_clean = labels[valid_mask]
    filenames_clean = [f for f, v in zip(filenames, valid_mask) if v]

    if len(np.unique(labels_clean)) < 2:
        print("❌ Need both fake and real samples for similarity analysis")
        return

    # Compute pairwise similarities
    n_files = len(features_clean)
    similarity_matrix = np.zeros((n_files, n_files))

    for i in range(n_files):
        for j in range(n_files):
            similarity_matrix[i, j] = 1 - cosine(features_clean[i], features_clean[j])

    # Create labels for plotting
    label_colors = ['red' if l == 0 else 'blue' for l in labels_clean]
    label_names = ['fake' if l == 0 else 'real' for l in labels_clean]

    # Plot similarity matrix with fake/real coloring
    plt.figure(figsize=(12, 10))

    # Custom colormap for the matrix
    mask = np.zeros_like(similarity_matrix, dtype=bool)
    mask[np.triu_indices_from(mask)] = True

    # Create subplot for similarity matrix
    plt.subplot(2, 2, 1)
    sns.heatmap(similarity_matrix,
                xticklabels=filenames_clean,
                yticklabels=filenames_clean,
                annot=True,
                fmt='.3f',
                cmap='viridis',
                mask=mask,
                square=True)
    plt.title('Similarity Matrix\n(Fake=Red labels, Real=Blue labels)')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    # Color the tick labels
    ax = plt.gca()
    for i, (ticklabel, color) in enumerate(zip(ax.get_xticklabels(), label_colors)):
        ticklabel.set_color(color)
    for i, (ticklabel, color) in enumerate(zip(ax.get_yticklabels(), label_colors)):
        ticklabel.set_color(color)

    # Analyze within-group vs between-group similarities
    fake_indices = np.where(labels_clean == 0)[0]
    real_indices = np.where(labels_clean == 1)[0]

    # Within-group similarities
    if len(fake_indices) > 1:
        fake_similarities = []
        for i in range(len(fake_indices)):
            for j in range(i+1, len(fake_indices)):
                fake_similarities.append(similarity_matrix[fake_indices[i], fake_indices[j]])
        fake_within_sim = np.mean(fake_similarities) if fake_similarities else 0
    else:
        fake_within_sim = 0

    if len(real_indices) > 1:
        real_similarities = []
        for i in range(len(real_indices)):
            for j in range(i+1, len(real_indices)):
                real_similarities.append(similarity_matrix[real_indices[i], real_indices[j]])
        real_within_sim = np.mean(real_similarities) if real_similarities else 0
    else:
        real_within_sim = 0

    # Between-group similarities
    between_similarities = []
    for i in fake_indices:
        for j in real_indices:
            between_similarities.append(similarity_matrix[i, j])
    between_sim = np.mean(between_similarities) if between_similarities else 0

    print(f"📊 Similarity Analysis Results:")
    print(f"   Fake-to-fake similarity: {fake_within_sim:.4f}")
    print(f"   Real-to-real similarity: {real_within_sim:.4f}")
    print(f"   Fake-to-real similarity: {between_sim:.4f}")

    # Interpretation
    if fake_within_sim > between_sim and real_within_sim > between_sim:
        print(f"   ✅ Good separation: Within-group > Between-group similarity")
    else:
        print(f"   ⚠️  Poor separation: Groups not well distinguished")

    # Plot similarity distributions
    plt.subplot(2, 2, 2)
    if fake_similarities:
        plt.hist(fake_similarities, bins=10, alpha=0.7, label='Fake-Fake', color='red')
    if real_similarities:
        plt.hist(real_similarities, bins=10, alpha=0.7, label='Real-Real', color='blue')
    if between_similarities:
        plt.hist(between_similarities, bins=10, alpha=0.7, label='Fake-Real', color='gray')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Count')
    plt.title('Similarity Distributions')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return {
        'fake_within': fake_within_sim,
        'real_within': real_within_sim,
        'between': between_sim
    }

def fake_real_clustering_analysis(features, labels, filenames):
    """Analyze clustering of fake vs real samples"""
    print("\n" + "=" * 60)
    print("🌳 FAKE vs REAL CLUSTERING ANALYSIS")
    print("=" * 60)

    # Filter valid labels
    valid_mask = labels >= 0
    features_clean = features[valid_mask]
    labels_clean = labels[valid_mask]
    filenames_clean = [f for f, v in zip(filenames, valid_mask) if v]

    if len(np.unique(labels_clean)) < 2:
        print("❌ Need both fake and real samples for clustering analysis")
        return

    # Hierarchical clustering
    distances = pdist(features_clean, metric='cosine')
    linkage_matrix = linkage(distances, method='ward')

    # Plot dendrogram with fake/real coloring
    plt.figure(figsize=(15, 8))

    # Create color mapping
    label_colors = ['red' if l == 0 else 'blue' for l in labels_clean]

    plt.subplot(1, 2, 1)
    dendrogram(linkage_matrix,
               labels=filenames_clean,
               leaf_rotation=45,
               leaf_font_size=10)
    plt.title('Hierarchical Clustering Dendrogram\n(Red=Fake, Blue=Real)')
    plt.ylabel('Distance')

    # Color the leaf labels
    ax = plt.gca()
    xlbls = ax.get_xmajorticklabels()
    for lbl in xlbls:
        filename = lbl.get_text()
        if filename in filenames_clean:
            idx = filenames_clean.index(filename)
            lbl.set_color(label_colors[idx])

    # Test different numbers of clusters
    silhouette_scores = []
    n_clusters_range = range(2, min(6, len(features_clean)))

    for n_clusters in n_clusters_range:
        clusters = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
        score = silhouette_score(features_clean, clusters)
        silhouette_scores.append(score)

    plt.subplot(1, 2, 2)
    plt.plot(n_clusters_range, silhouette_scores, 'bo-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Clustering Quality vs Number of Clusters')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Analyze clustering purity with 2 clusters
    clusters_2 = fcluster(linkage_matrix, 2, criterion='maxclust')

    print(f"📂 2-Cluster Analysis:")
    cluster_purity = []
    for cluster_id in [1, 2]:
        cluster_mask = clusters_2 == cluster_id
        cluster_labels = labels_clean[cluster_mask]
        cluster_files = [f for f, m in zip(filenames_clean, cluster_mask) if m]

        if len(cluster_labels) > 0:
            fake_count = np.sum(cluster_labels == 0)
            real_count = np.sum(cluster_labels == 1)
            purity = max(fake_count, real_count) / len(cluster_labels)
            cluster_purity.append(purity)

            print(f"   Cluster {cluster_id}: {len(cluster_labels)} files")
            print(f"     Fake: {fake_count}, Real: {real_count}")
            print(f"     Purity: {purity:.3f}")
            print(f"     Files: {cluster_files}")

    overall_purity = np.mean(cluster_purity) if cluster_purity else 0
    print(f"   Overall clustering purity: {overall_purity:.3f}")

    if overall_purity > 0.8:
        print(f"   ✅ Excellent clustering: Features clearly separate fake/real")
    elif overall_purity > 0.6:
        print(f"   ✅ Good clustering: Features partially separate fake/real")
    else:
        print(f"   ⚠️  Poor clustering: Features don't separate fake/real well")

    return overall_purity

def fake_real_tsne_analysis(features, labels, filenames):
    """t-SNE visualization colored by fake/real labels"""
    print("\n" + "=" * 60)
    print("🎨 FAKE vs REAL t-SNE VISUALIZATION")
    print("=" * 60)

    # Filter valid labels
    valid_mask = labels >= 0
    features_clean = features[valid_mask]
    labels_clean = labels[valid_mask]
    filenames_clean = [f for f, v in zip(filenames, valid_mask) if v]

    if len(np.unique(labels_clean)) < 2:
        print("❌ Need both fake and real samples for t-SNE analysis")
        return

    # Normalize features
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(features_clean)

    # Run t-SNE
    n_files = len(features_clean)
    perplexity = min(30, max(2, n_files - 1))

    print(f"Running t-SNE with perplexity={perplexity}...")

    tsne = TSNE(n_components=2,
                perplexity=perplexity,
                random_state=42,
                max_iter=1000)
    features_2d = tsne.fit_transform(features_normalized)

    # Plot t-SNE with fake/real coloring
    plt.figure(figsize=(12, 8))

    # Color mapping
    colors = ['red' if l == 0 else 'blue' for l in labels_clean]
    labels_text = ['fake' if l == 0 else 'real' for l in labels_clean]

    # Scatter plot
    for label_val, color, label_text in [(0, 'red', 'fake'), (1, 'blue', 'real')]:
        mask = labels_clean == label_val
        plt.scatter(features_2d[mask, 0], features_2d[mask, 1],
                   c=color, label=label_text, s=100, alpha=0.7)

    # Add file labels
    for i, filename in enumerate(filenames_clean):
        plt.annotate(filename,
                    (features_2d[i, 0], features_2d[i, 1]),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=9,
                    alpha=0.8,
                    color=colors[i])

    plt.title('t-SNE Visualization: Fake vs Real Audio\n(Red=Fake, Blue=Real)')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Analyze separation in t-SNE space
    fake_indices = np.where(labels_clean == 0)[0]
    real_indices = np.where(labels_clean == 1)[0]

    if len(fake_indices) > 0 and len(real_indices) > 0:
        fake_centroid = np.mean(features_2d[fake_indices], axis=0)
        real_centroid = np.mean(features_2d[real_indices], axis=0)
        centroid_distance = np.linalg.norm(fake_centroid - real_centroid)

        # Average within-group distances
        fake_distances = [np.linalg.norm(features_2d[i] - fake_centroid) for i in fake_indices]
        real_distances = [np.linalg.norm(features_2d[i] - real_centroid) for i in real_indices]

        avg_fake_spread = np.mean(fake_distances) if fake_distances else 0
        avg_real_spread = np.mean(real_distances) if real_distances else 0
        avg_within_spread = (avg_fake_spread + avg_real_spread) / 2

        separation_ratio = centroid_distance / (avg_within_spread + 1e-6)

        print(f"📊 t-SNE Separation Analysis:")
        print(f"   Distance between fake/real centroids: {centroid_distance:.3f}")
        print(f"   Average within-group spread: {avg_within_spread:.3f}")
        print(f"   Separation ratio: {separation_ratio:.3f}")

        if separation_ratio > 2.0:
            print(f"   ✅ Excellent separation in t-SNE space")
        elif separation_ratio > 1.0:
            print(f"   ✅ Good separation in t-SNE space")
        else:
            print(f"   ⚠️  Poor separation in t-SNE space")

    return features_2d

def main():
    """Main analysis function for fake vs real audio detection"""
    print("🎵 AST FEATURE ANALYSIS: FAKE vs REAL AUDIO DETECTION")
    print("=" * 80)

    print("🔄 Step 1: Looking for extracted features...")
    print()

    # Try to load separate fake and real features first
    fake_real_loaded = False
    features, time_axis, filenames, metadata = None, None, None, None

    try:
        print("🔍 Searching for separate fake and real feature files...")
        if os.path.exists("output_features_fake.npz") and os.path.exists("output_features_real.npz"):
            print("✅ Found separate fake and real feature files!")
            features, time_axis, filenames, metadata = load_separate_fake_real_features(
                "output_features_fake",
                "output_features_real"
            )
            print(f"✅ Successfully loaded and combined features:")
            print(f"   - Fake files: {metadata['combined_from']['evaluate_files']}")
            print(f"   - Real files: {metadata['combined_from']['real_files']}")
            fake_real_loaded = True
        else:
            print("❌ Separate fake/real files not found")
            print("🔍 Looking for combined feature files...")
    except FileNotFoundError as e:
        print(f"❌ Error loading separate files: {e}")
        print("🔍 Looking for combined feature files...")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        print("🔍 Looking for combined feature files...")

    # Fallback: try to load combined features
    if not fake_real_loaded:
        possible_paths = [
            "output_features_all",
            "output_features",
        ]

        for base_path in possible_paths:
            try:
                print(f"🔍 Trying to load {base_path}.npz...")
                features, time_axis, filenames, metadata = load_ast_features(base_path)
                print(f"✅ Loaded features from {base_path}.npz")
                break
            except FileNotFoundError:
                print(f"❌ {base_path}.npz not found")
                continue
            except Exception as e:
                print(f"❌ Error loading {base_path}.npz: {e}")
                continue

    if features is None:
        print()
        print("❌ No feature files found!")
        print()
        print("📂 Current directory files:")
        current_files = [f for f in os.listdir('.') if f.endswith('.npz') or f.endswith('.meta.json')]
        if current_files:
            for f in sorted(current_files):
                print(f"   {f}")
        else:
            print("   No .npz or .meta.json files found")

        print()
        print("🔧 Please run feature extraction first:")
        print()
        print("# Extract fake features:")
        print("python extract_audio_features_from_AST.py \\")
        print("    --data-dir video_dir_test/eval_fake \\")
        print("    --recursive no \\")
        print("    --time-series yes \\")
        print("    --return-time-axis yes \\")
        print("    --center-time yes \\")
        print("    --token-pool none \\")
        print("    --out output_features_fake")
        print()
        print("# Extract real features:")
        print("python extract_audio_features_from_AST.py \\")
        print("    --data-dir video_dir_test/eval_real \\")
        print("    --recursive no \\")
        print("    --time-series yes \\")
        print("    --return-time-axis yes \\")
        print("    --center-time yes \\")
        print("    --token-pool none \\")
        print("    --out output_features_real")
        return

    print(f"📁 Loaded features from {len(filenames)} files")
    print(f"📊 Feature shape: {features.shape}")
    print(f"📂 Files: {filenames}")

    # Parse fake/real labels
    labels, label_names = parse_fake_real_labels(filenames)
    print(f"\n🏷️  Labels: {dict(zip(filenames, label_names))}")

    # Check if we have both fake and real samples
    unique_labels = np.unique(labels[labels >= 0])
    if len(unique_labels) < 2:
        print("❌ Need both fake and real samples for analysis!")
        print("   Make sure your filenames contain 'fake' and 'real'")
        return

    # Compute global features for analysis
    global_features = compute_global_features(features)

    # Test different feature aggregations
    feature_types = {
        'mean': global_features['mean'],
        'std': global_features['std'],
        'max': global_features['max'],
        'temporal_variance': global_features['temporal_variance'],
        'temporal_smoothness': global_features['temporal_smoothness']
    }

    print(f"\n🧪 Testing multiple feature aggregations...")

    best_accuracy = 0
    best_features = None
    best_type = None

    for feature_type, file_features in feature_types.items():
        print(f"\n" + "="*60)
        print(f"🎯 ANALYZING {feature_type.upper()} FEATURES")
        print(f"Feature shape: {file_features.shape}")

        # Classification test
        results = fake_real_classification_test(file_features, labels, filenames)
        if results:
            max_acc = max([np.mean(scores) for scores in results.values()])
            if max_acc > best_accuracy:
                best_accuracy = max_acc
                best_features = file_features
                best_type = feature_type

    # Detailed analysis with best features
    if best_features is not None:
        print(f"\n" + "="*80)
        print(f"🏆 DETAILED ANALYSIS WITH BEST FEATURES: {best_type.upper()}")
        print(f"Best classification accuracy: {best_accuracy:.4f}")
        print(f"="*80)

        # Run all analyses with best features
        similarity_results = fake_real_similarity_analysis(best_features, labels, filenames)
        clustering_purity = fake_real_clustering_analysis(best_features, labels, filenames)
        tsne_features = fake_real_tsne_analysis(best_features, labels, filenames)

        # Final assessment
        print(f"\n" + "="*80)
        print(f"📋 FINAL ASSESSMENT: FEATURE MEANINGFULNESS FOR FAKE DETECTION")
        print(f"="*80)

        meaningful_indicators = 0
        total_indicators = 0

        # Classification accuracy
        total_indicators += 1
        if best_accuracy > 0.8:
            print(f"✅ Classification accuracy: {best_accuracy:.3f} (Excellent)")
            meaningful_indicators += 1
        elif best_accuracy > 0.6:
            print(f"✅ Classification accuracy: {best_accuracy:.3f} (Good)")
            meaningful_indicators += 1
        else:
            print(f"❌ Classification accuracy: {best_accuracy:.3f} (Poor)")

        # Clustering purity
        if clustering_purity:
            total_indicators += 1
            if clustering_purity > 0.8:
                print(f"✅ Clustering purity: {clustering_purity:.3f} (Excellent)")
                meaningful_indicators += 1
            elif clustering_purity > 0.6:
                print(f"✅ Clustering purity: {clustering_purity:.3f} (Good)")
                meaningful_indicators += 1
            else:
                print(f"❌ Clustering purity: {clustering_purity:.3f} (Poor)")

        # Similarity separation
        if similarity_results:
            total_indicators += 1
            within_sim = (similarity_results['fake_within'] + similarity_results['real_within']) / 2
            between_sim = similarity_results['between']
            separation = within_sim - between_sim

            if separation > 0.2:
                print(f"✅ Similarity separation: {separation:.3f} (Excellent)")
                meaningful_indicators += 1
            elif separation > 0.1:
                print(f"✅ Similarity separation: {separation:.3f} (Good)")
                meaningful_indicators += 1
            else:
                print(f"❌ Similarity separation: {separation:.3f} (Poor)")

        # Overall assessment
        meaningfulness_score = meaningful_indicators / total_indicators if total_indicators > 0 else 0

        print(f"\n🎯 OVERALL MEANINGFULNESS SCORE: {meaningfulness_score:.2f}")
        print(f"   ({meaningful_indicators}/{total_indicators} indicators positive)")

        if meaningfulness_score >= 0.75:
            print(f"   🎉 EXCELLENT: Features are highly meaningful for fake detection!")
        elif meaningfulness_score >= 0.5:
            print(f"   ✅ GOOD: Features show promise for fake detection")
        else:
            print(f"   ⚠️  POOR: Features may not be suitable for fake detection")

        print(f"\n💡 INTERPRETATION:")
        print(f"   🔍 High classification accuracy → Features capture fake/real differences")
        print(f"   🌳 High clustering purity → Features naturally group fake/real separately")
        print(f"   📊 Good similarity separation → Fakes similar to fakes, reals to reals")
        print(f"   🎨 Clear t-SNE separation → Features form distinct clusters in low-D space")

if __name__ == "__main__":
    main()