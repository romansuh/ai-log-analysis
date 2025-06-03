#!/usr/bin/env python3
"""
Log Entropy Analysis for Anomaly Detection
==========================================

This script implements multiple entropy metrics to measure uncertainty in log messages
and correlates them with anomaly likelihood for automated log monitoring.
"""

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import re
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, classification_report
from transformers import AutoTokenizer, AutoModel
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

from src.bert_pipeline import BERTConfig, LogBERTPipeline

class LogEntropyAnalyzer:
    """
    Comprehensive entropy analysis for log messages.
    
    Implements multiple entropy metrics:
    1. Token-level Shannon entropy
    2. BERT embedding entropy  
    3. Contextual uncertainty from attention
    4. Cluster-based entropy
    5. Pattern deviation entropy
    """
    
    def __init__(self, model_name="distilbert-base-uncased"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        
        # For pattern analysis
        self.token_frequencies = defaultdict(int)
        self.pattern_frequencies = defaultdict(int)
        self.cluster_centers = None
        self.fitted = False
        
    def fit(self, log_messages, log_patterns=None):
        """
        Fit the entropy analyzer on training log data.
        
        Args:
            log_messages: List of log message strings
            log_patterns: Optional list of log patterns for pattern-based entropy
        """
        print("🔧 Fitting entropy analyzer on training data...")
        
        # Build token frequency distribution
        all_tokens = []
        for message in log_messages:
            tokens = self.tokenizer.tokenize(message.lower())
            all_tokens.extend(tokens)
            for token in tokens:
                self.token_frequencies[token] += 1
        
        # Calculate total tokens for probability estimation
        self.total_tokens = len(all_tokens)
        
        # Build pattern frequencies if provided
        if log_patterns:
            for pattern in log_patterns:
                self.pattern_frequencies[pattern] += 1
        
        # Extract embeddings for cluster-based entropy
        print("Extracting embeddings for cluster analysis...")
        embeddings = self._extract_embeddings(log_messages)
        
        # Fit clustering model for cluster-based entropy
        n_clusters = min(10, len(log_messages) // 20)  # Adaptive cluster count
        if n_clusters >= 2:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            kmeans.fit(embeddings)
            self.cluster_centers = kmeans.cluster_centers_
        
        self.fitted = True
        print(f"✅ Entropy analyzer fitted with {len(self.token_frequencies)} unique tokens")
        
    def calculate_entropy_metrics(self, log_messages, log_patterns=None):
        """
        Calculate comprehensive entropy metrics for log messages.
        
        Returns:
            DataFrame with entropy metrics for each log message
        """
        if not self.fitted:
            raise ValueError("Analyzer must be fitted before calculating entropy")
            
        print("📊 Calculating entropy metrics...")
        
        results = []
        embeddings = self._extract_embeddings(log_messages)
        
        for i, message in enumerate(log_messages):
            entropy_data = {
                'message_index': i,
                'message': message,
                'message_length': len(message)
            }
            
            # 1. Token-level Shannon entropy
            entropy_data.update(self._calculate_token_entropy(message))
            
            # 2. BERT embedding entropy
            entropy_data.update(self._calculate_embedding_entropy(embeddings[i]))
            
            # 3. Cluster-based entropy
            if self.cluster_centers is not None:
                entropy_data.update(self._calculate_cluster_entropy(embeddings[i]))
            
            # 4. Pattern-based entropy
            if log_patterns and i < len(log_patterns):
                entropy_data.update(self._calculate_pattern_entropy(log_patterns[i]))
            
            # 5. Contextual uncertainty from BERT attention
            entropy_data.update(self._calculate_contextual_entropy(message))
            
            results.append(entropy_data)
        
        df = pd.DataFrame(results)
        
        # Calculate composite entropy score
        df['composite_entropy'] = self._calculate_composite_entropy(df)
        
        return df
    
    def _extract_embeddings(self, texts, batch_size=8):
        """Extract BERT embeddings efficiently."""
        embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                encoded = self.tokenizer(
                    batch_texts,
                    add_special_tokens=True,
                    max_length=256,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                
                outputs = self.model(**encoded)
                cls_embeddings = outputs.last_hidden_state[:, 0, :]
                embeddings.append(cls_embeddings.cpu().numpy())
        
        return np.vstack(embeddings)
    
    def _calculate_token_entropy(self, message):
        """Calculate Shannon entropy based on token frequencies."""
        tokens = self.tokenizer.tokenize(message.lower())
        
        if not tokens:
            return {
                'token_entropy': 0.0,
                'token_perplexity': 1.0,
                'rare_token_ratio': 0.0,
                'unique_token_ratio': 0.0
            }
        
        # Token probabilities based on training data
        token_probs = []
        rare_tokens = 0
        
        for token in tokens:
            freq = self.token_frequencies.get(token, 1)  # Laplace smoothing
            prob = freq / (self.total_tokens + len(self.token_frequencies))
            token_probs.append(prob)
            
            # Count rare tokens (frequency < 5)
            if freq < 5:
                rare_tokens += 1
        
        # Shannon entropy: H = -Σ p(x) * log2(p(x))
        entropy = -sum(p * np.log2(p) for p in token_probs if p > 0)
        
        # Perplexity: 2^H
        perplexity = 2 ** entropy if entropy > 0 else 1.0
        
        # Additional metrics
        rare_token_ratio = rare_tokens / len(tokens)
        unique_token_ratio = len(set(tokens)) / len(tokens)
        
        return {
            'token_entropy': entropy,
            'token_perplexity': perplexity,
            'rare_token_ratio': rare_token_ratio,
            'unique_token_ratio': unique_token_ratio
        }
    
    def _calculate_embedding_entropy(self, embedding):
        """Calculate entropy based on BERT embedding distribution."""
        # Normalize embedding
        embedding_norm = embedding / (np.linalg.norm(embedding) + 1e-8)
        
        # Calculate entropy based on activation distribution
        # Use softmax to convert to probability distribution
        softmax_weights = np.exp(embedding_norm) / np.sum(np.exp(embedding_norm))
        
        # Shannon entropy of the embedding distribution
        embedding_entropy = -np.sum(softmax_weights * np.log2(softmax_weights + 1e-8))
        
        # Additional metrics
        embedding_variance = np.var(embedding)
        embedding_sparsity = np.sum(np.abs(embedding) < 0.01) / len(embedding)
        embedding_magnitude = np.linalg.norm(embedding)
        
        return {
            'embedding_entropy': embedding_entropy,
            'embedding_variance': embedding_variance,
            'embedding_sparsity': embedding_sparsity,
            'embedding_magnitude': embedding_magnitude
        }
    
    def _calculate_cluster_entropy(self, embedding):
        """Calculate entropy based on distance to cluster centers."""
        if self.cluster_centers is None:
            return {'cluster_entropy': 0.0, 'min_cluster_distance': 0.0}
        
        # Calculate distances to all cluster centers
        distances = cdist([embedding], self.cluster_centers)[0]
        
        # Convert distances to probabilities (softmax)
        exp_neg_distances = np.exp(-distances / np.std(distances))
        cluster_probs = exp_neg_distances / np.sum(exp_neg_distances)
        
        # Shannon entropy of cluster assignment probabilities
        cluster_entropy = -np.sum(cluster_probs * np.log2(cluster_probs + 1e-8))
        
        # Minimum distance to any cluster center (uncertainty measure)
        min_distance = np.min(distances)
        
        return {
            'cluster_entropy': cluster_entropy,
            'min_cluster_distance': min_distance
        }
    
    def _calculate_pattern_entropy(self, pattern):
        """Calculate entropy based on pattern frequency."""
        if not self.pattern_frequencies:
            return {'pattern_entropy': 0.0, 'pattern_rarity': 0.0}
        
        pattern_freq = self.pattern_frequencies.get(pattern, 1)
        total_patterns = sum(self.pattern_frequencies.values())
        
        # Pattern probability
        pattern_prob = pattern_freq / total_patterns
        
        # Pattern entropy (negative log probability)
        pattern_entropy = -np.log2(pattern_prob) if pattern_prob > 0 else 0
        
        # Pattern rarity (inverse of normalized frequency)
        pattern_rarity = 1.0 / (pattern_freq / total_patterns)
        
        return {
            'pattern_entropy': pattern_entropy,
            'pattern_rarity': pattern_rarity
        }
    
    def _calculate_contextual_entropy(self, message):
        """Calculate contextual uncertainty using BERT attention weights."""
        try:
            # Tokenize and get attention weights
            encoded = self.tokenizer(
                message,
                add_special_tokens=True,
                max_length=256,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            with torch.no_grad():
                # Get model outputs with attention weights
                outputs = self.model(**encoded, output_attentions=True)
                
                # Extract attention weights from last layer
                attention_weights = outputs.attentions[-1][0]  # [num_heads, seq_len, seq_len]
                
                # Average across heads and compute entropy
                avg_attention = torch.mean(attention_weights, dim=0)  # [seq_len, seq_len]
                
                # Calculate entropy of attention distribution
                attention_entropy = 0.0
                for i in range(avg_attention.shape[0]):
                    attention_dist = avg_attention[i].cpu().numpy()
                    attention_dist = attention_dist / (np.sum(attention_dist) + 1e-8)
                    entropy = -np.sum(attention_dist * np.log2(attention_dist + 1e-8))
                    attention_entropy += entropy
                
                attention_entropy /= avg_attention.shape[0]  # Average across positions
                
                # Attention concentration (inverse of entropy)
                attention_concentration = 1.0 / (attention_entropy + 1.0)
                
        except Exception as e:
            # Fallback values if attention calculation fails
            attention_entropy = 0.0
            attention_concentration = 1.0
        
        return {
            'attention_entropy': attention_entropy,
            'attention_concentration': attention_concentration
        }
    
    def _calculate_composite_entropy(self, df):
        """Calculate composite entropy score combining all metrics."""
        # Normalize all entropy metrics to [0, 1] range
        entropy_cols = [col for col in df.columns if 'entropy' in col and col != 'composite_entropy']
        
        normalized_entropies = []
        for col in entropy_cols:
            if df[col].std() > 0:
                normalized = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
                normalized_entropies.append(normalized)
        
        if normalized_entropies:
            # Weighted average of normalized entropies
            weights = [0.3, 0.25, 0.2, 0.15, 0.1]  # Adjust based on importance
            weights = weights[:len(normalized_entropies)]
            weights = np.array(weights) / np.sum(weights)  # Normalize weights
            
            composite = np.average(normalized_entropies, axis=0, weights=weights)
            return composite
        else:
            return np.zeros(len(df))


def detect_anomalies_with_entropy(entropy_df, contamination=0.1):
    """
    Detect anomalies using entropy metrics and Isolation Forest.
    
    Args:
        entropy_df: DataFrame with entropy metrics
        contamination: Expected proportion of anomalies
        
    Returns:
        DataFrame with anomaly predictions and scores
    """
    print(f"🔍 Detecting anomalies using entropy metrics...")
    
    # Select entropy features for anomaly detection
    entropy_features = [
        'token_entropy', 'embedding_entropy', 'cluster_entropy',
        'pattern_entropy', 'attention_entropy', 'composite_entropy',
        'rare_token_ratio', 'embedding_variance', 'min_cluster_distance'
    ]
    
    # Filter available features
    available_features = [f for f in entropy_features if f in entropy_df.columns]
    
    if not available_features:
        print("⚠️  No entropy features available for anomaly detection")
        return entropy_df
    
    # Prepare feature matrix
    X = entropy_df[available_features].fillna(0)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Isolation Forest for anomaly detection
    iso_forest = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_estimators=100
    )
    
    anomaly_labels = iso_forest.fit_predict(X_scaled)
    anomaly_scores = iso_forest.score_samples(X_scaled)
    
    # Add results to dataframe
    result_df = entropy_df.copy()
    result_df['is_anomaly'] = (anomaly_labels == -1)
    result_df['anomaly_score'] = -anomaly_scores  # Higher score = more anomalous
    result_df['anomaly_percentile'] = stats.rankdata(result_df['anomaly_score']) / len(result_df)
    
    # Entropy-based anomaly score
    if 'composite_entropy' in result_df.columns:
        result_df['entropy_anomaly_score'] = result_df['composite_entropy']
    
    print(f"✅ Detected {result_df['is_anomaly'].sum()} anomalies out of {len(result_df)} messages")
    
    return result_df


def create_entropy_visualizations(entropy_df):
    """Create comprehensive entropy analysis visualizations."""
    
    print("📊 Creating entropy analysis visualizations...")
    
    plt.style.use('default')
    sns.set_palette("husl")
    
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Entropy distribution histograms
    ax1 = plt.subplot(3, 4, 1)
    entropy_cols = [col for col in entropy_df.columns if 'entropy' in col][:4]
    for col in entropy_cols:
        if col in entropy_df.columns:
            plt.hist(entropy_df[col], alpha=0.6, label=col.replace('_entropy', ''), bins=30)
    plt.xlabel('Entropy Value')
    plt.ylabel('Frequency')
    plt.title('Entropy Distributions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Composite entropy vs anomaly scores
    ax2 = plt.subplot(3, 4, 2)
    if 'composite_entropy' in entropy_df.columns and 'anomaly_score' in entropy_df.columns:
        colors = ['red' if x else 'blue' for x in entropy_df['is_anomaly']]
        plt.scatter(entropy_df['composite_entropy'], entropy_df['anomaly_score'], 
                   c=colors, alpha=0.6, s=20)
        plt.xlabel('Composite Entropy')
        plt.ylabel('Anomaly Score')
        plt.title('Entropy vs Anomaly Score')
        
        # Add correlation coefficient
        corr = entropy_df['composite_entropy'].corr(entropy_df['anomaly_score'])
        plt.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=ax2.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    plt.grid(True, alpha=0.3)
    
    # 3. Entropy correlation heatmap
    ax3 = plt.subplot(3, 4, 3)
    entropy_numeric_cols = entropy_df.select_dtypes(include=[np.number]).columns
    entropy_subset = [col for col in entropy_numeric_cols if 'entropy' in col or 'anomaly' in col]
    if len(entropy_subset) > 1:
        corr_matrix = entropy_df[entropy_subset].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, ax=ax3, fmt='.2f')
        plt.title('Entropy Correlation Matrix')
    
    # 4. Token entropy vs message length
    ax4 = plt.subplot(3, 4, 4)
    if 'token_entropy' in entropy_df.columns and 'message_length' in entropy_df.columns:
        colors = ['red' if x else 'blue' for x in entropy_df['is_anomaly']]
        plt.scatter(entropy_df['message_length'], entropy_df['token_entropy'], 
                   c=colors, alpha=0.6, s=20)
        plt.xlabel('Message Length')
        plt.ylabel('Token Entropy')
        plt.title('Token Entropy vs Message Length')
    plt.grid(True, alpha=0.3)
    
    # 5. Embedding entropy distribution by anomaly
    ax5 = plt.subplot(3, 4, 5)
    if 'embedding_entropy' in entropy_df.columns and 'is_anomaly' in entropy_df.columns:
        normal_entropy = entropy_df[~entropy_df['is_anomaly']]['embedding_entropy']
        anomaly_entropy = entropy_df[entropy_df['is_anomaly']]['embedding_entropy']
        
        plt.hist(normal_entropy, alpha=0.7, label='Normal', bins=20, color='blue')
        plt.hist(anomaly_entropy, alpha=0.7, label='Anomaly', bins=20, color='red')
        plt.xlabel('Embedding Entropy')
        plt.ylabel('Frequency')
        plt.title('Embedding Entropy by Anomaly Type')
        plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 6. Cluster distance vs entropy
    ax6 = plt.subplot(3, 4, 6)
    if 'min_cluster_distance' in entropy_df.columns and 'cluster_entropy' in entropy_df.columns:
        colors = ['red' if x else 'blue' for x in entropy_df['is_anomaly']]
        plt.scatter(entropy_df['min_cluster_distance'], entropy_df['cluster_entropy'],
                   c=colors, alpha=0.6, s=20)
        plt.xlabel('Min Cluster Distance')
        plt.ylabel('Cluster Entropy')
        plt.title('Cluster-based Metrics')
    plt.grid(True, alpha=0.3)
    
    # 7. Anomaly score distribution
    ax7 = plt.subplot(3, 4, 7)
    if 'anomaly_score' in entropy_df.columns:
        normal_scores = entropy_df[~entropy_df['is_anomaly']]['anomaly_score']
        anomaly_scores = entropy_df[entropy_df['is_anomaly']]['anomaly_score']
        
        plt.hist(normal_scores, alpha=0.7, label='Normal', bins=20, color='blue')
        plt.hist(anomaly_scores, alpha=0.7, label='Anomaly', bins=20, color='red')
        plt.xlabel('Anomaly Score')
        plt.ylabel('Frequency')
        plt.title('Anomaly Score Distribution')
        plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 8. Top anomalous messages
    ax8 = plt.subplot(3, 4, (8, 12))
    ax8.axis('off')
    
    if 'anomaly_score' in entropy_df.columns:
        top_anomalies = entropy_df.nlargest(10, 'anomaly_score')
        
        anomaly_text = "Top 10 Most Anomalous Log Messages:\n\n"
        for i, (_, row) in enumerate(top_anomalies.iterrows()):
            score = row['anomaly_score']
            entropy = row.get('composite_entropy', 0)
            message = row['message'][:80] + "..." if len(row['message']) > 80 else row['message']
            
            anomaly_text += f"{i+1}. Score: {score:.3f}, Entropy: {entropy:.3f}\n"
            anomaly_text += f"   {message}\n\n"
        
        ax8.text(0.02, 0.98, anomaly_text, transform=ax8.transAxes, fontsize=9,
                verticalalignment='top', fontfamily='monospace')
    
    # 9-11. Individual entropy metric plots
    subplot_positions = [9, 10, 11]
    individual_entropies = ['token_entropy', 'embedding_entropy', 'attention_entropy']
    
    for pos, entropy_col in zip(subplot_positions, individual_entropies):
        if entropy_col in entropy_df.columns and 'is_anomaly' in entropy_df.columns:
            ax = plt.subplot(3, 4, pos)
            
            normal_data = entropy_df[~entropy_df['is_anomaly']][entropy_col]
            anomaly_data = entropy_df[entropy_df['is_anomaly']][entropy_col]
            
            plt.boxplot([normal_data, anomaly_data], labels=['Normal', 'Anomaly'])
            plt.ylabel(entropy_col.replace('_', ' ').title())
            plt.title(f'{entropy_col.replace("_", " ").title()} by Type')
            plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('log_entropy_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig


def analyze_entropy_correlations(entropy_df):
    """Analyze correlations between entropy metrics and anomaly likelihood."""
    
    print("\n📈 ENTROPY-ANOMALY CORRELATION ANALYSIS")
    print("=" * 50)
    
    if 'is_anomaly' not in entropy_df.columns:
        print("⚠️  No anomaly labels available for correlation analysis")
        return
    
    # Calculate correlations with anomaly indicator
    numeric_cols = entropy_df.select_dtypes(include=[np.number]).columns
    correlations = {}
    
    for col in numeric_cols:
        if col != 'is_anomaly' and not col.startswith('message_'):
            corr = entropy_df[col].corr(entropy_df['is_anomaly'].astype(int))
            correlations[col] = corr
    
    # Sort by absolute correlation
    sorted_correlations = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    
    print("🎯 Entropy Metrics Ranked by Anomaly Correlation:")
    print("-" * 50)
    for metric, corr in sorted_correlations[:10]:
        direction = "↑" if corr > 0 else "↓"
        strength = "Strong" if abs(corr) > 0.5 else "Moderate" if abs(corr) > 0.3 else "Weak"
        print(f"   {metric:25s}: {corr:6.3f} {direction} ({strength})")
    
    # Statistical significance tests
    print(f"\n📊 STATISTICAL ANALYSIS:")
    print("-" * 30)
    
    # Compare entropy distributions between normal and anomalous logs
    entropy_metrics = ['token_entropy', 'embedding_entropy', 'cluster_entropy', 'composite_entropy']
    
    for metric in entropy_metrics:
        if metric in entropy_df.columns:
            normal_values = entropy_df[~entropy_df['is_anomaly']][metric]
            anomaly_values = entropy_df[entropy_df['is_anomaly']][metric]
            
            if len(normal_values) > 0 and len(anomaly_values) > 0:
                # Welch's t-test (unequal variances)
                t_stat, p_value = stats.ttest_ind(anomaly_values, normal_values, equal_var=False)
                
                # Effect size (Cohen's d)
                pooled_std = np.sqrt(((len(anomaly_values) - 1) * np.var(anomaly_values, ddof=1) +
                                    (len(normal_values) - 1) * np.var(normal_values, ddof=1)) /
                                   (len(anomaly_values) + len(normal_values) - 2))
                cohens_d = (np.mean(anomaly_values) - np.mean(normal_values)) / pooled_std
                
                significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
                
                print(f"{metric:20s}: t={t_stat:6.3f}, p={p_value:.6f}{significance}, d={cohens_d:5.3f}")
    
    # Summary insights
    print(f"\n💡 KEY INSIGHTS:")
    print("-" * 20)
    
    best_metric = sorted_correlations[0] if sorted_correlations else None
    if best_metric:
        metric_name, correlation = best_metric
        print(f"• Best anomaly predictor: {metric_name} (r={correlation:.3f})")
    
    high_corr_metrics = [m for m, c in sorted_correlations if abs(c) > 0.4]
    if high_corr_metrics:
        print(f"• Strong predictors ({len(high_corr_metrics)}): {', '.join(high_corr_metrics[:3])}")
    
    if 'composite_entropy' in correlations:
        comp_corr = correlations['composite_entropy']
        print(f"• Composite entropy correlation: {comp_corr:.3f}")
    
    return correlations


def main():
    """Main function to demonstrate entropy analysis on real logs."""
    
    print("🔍 LOG ENTROPY ANALYSIS FOR ANOMALY DETECTION")
    print("=" * 60)
    
    # Parse the Zookeeper log file (reusing from previous analysis)
    from real_log_clustering import ZookeeperLogParser
    
    parser = ZookeeperLogParser()
    log_data = parser.parse_log_file('Zookeeper_2k.log', sample_size=500)  # Smaller sample for speed
    
    print(f"\n📊 LOG DATA OVERVIEW:")
    print(f"   Total entries: {len(log_data)}")
    print(f"   Log levels: {log_data['log_level'].value_counts().to_dict()}")
    
    # Initialize entropy analyzer
    print(f"\n🔧 INITIALIZING ENTROPY ANALYZER")
    print("-" * 40)
    
    analyzer = LogEntropyAnalyzer(model_name="distilbert-base-uncased")
    
    # Prepare data
    log_messages = log_data['message'].tolist()
    log_patterns = log_data['message_pattern'].tolist() if 'message_pattern' in log_data.columns else None
    
    # Fit analyzer on first 80% of data
    train_size = int(len(log_messages) * 0.8)
    train_messages = log_messages[:train_size]
    train_patterns = log_patterns[:train_size] if log_patterns else None
    
    analyzer.fit(train_messages, train_patterns)
    
    # Calculate entropy metrics for all data
    print(f"\n📊 CALCULATING ENTROPY METRICS")
    print("-" * 40)
    
    entropy_df = analyzer.calculate_entropy_metrics(log_messages, log_patterns)
    
    # Detect anomalies using entropy
    print(f"\n🔍 ANOMALY DETECTION")
    print("-" * 25)
    
    entropy_with_anomalies = detect_anomalies_with_entropy(entropy_df, contamination=0.05)
    
    # Create visualizations
    print(f"\n📊 CREATING VISUALIZATIONS")
    print("-" * 30)
    
    create_entropy_visualizations(entropy_with_anomalies)
    
    # Analyze correlations
    correlations = analyze_entropy_correlations(entropy_with_anomalies)
    
    # Print summary
    print(f"\n✅ ENTROPY ANALYSIS COMPLETE!")
    print("=" * 40)
    print(f"📁 Visualization saved as 'log_entropy_analysis.png'")
    print(f"🎯 Analyzed {len(entropy_df)} log messages")
    print(f"📊 Calculated {len([c for c in entropy_df.columns if 'entropy' in c])} entropy metrics")
    print(f"🔍 Detected {entropy_with_anomalies['is_anomaly'].sum()} anomalies")
    
    # Show top entropy insights
    print(f"\n📋 ENTROPY INSIGHTS:")
    high_entropy_msgs = entropy_with_anomalies.nlargest(5, 'composite_entropy')
    for i, (_, row) in enumerate(high_entropy_msgs.iterrows()):
        msg = row['message'][:60] + "..." if len(row['message']) > 60 else row['message']
        entropy = row['composite_entropy']
        anomaly = "🚨 ANOMALY" if row['is_anomaly'] else "✅ Normal"
        print(f"   {i+1}. Entropy: {entropy:.3f} | {anomaly}")
        print(f"      {msg}")


if __name__ == "__main__":
    main() 