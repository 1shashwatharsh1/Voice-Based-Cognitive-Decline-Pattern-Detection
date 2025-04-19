import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple

class MLProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=3)  # Increased to 3 components for better separation
        self.kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        
    def prepare_features(self, features_list):
        """Convert features list to DataFrame and prepare for analysis."""
        df = pd.DataFrame(features_list)
        
        # Define columns to drop if they exist
        columns_to_drop = ['pause_durations', 'file_name']
        existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
        
        if existing_columns_to_drop:
            df = df.drop(columns=existing_columns_to_drop)
            
        # Fill missing values with mean for numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
        
        return df
    
    def analyze_features(self, df):
        """Analyze features using PCA and clustering."""
        # Scale features
        scaled_features = self.scaler.fit_transform(df)
        
        # Apply PCA
        pca_result = self.pca.fit_transform(scaled_features)
        
        # Perform clustering
        clusters = self.kmeans.fit_predict(scaled_features)
        
        # Calculate feature importance
        feature_importance = pd.DataFrame(
            self.pca.components_.T,
            columns=[f'PC{i+1}' for i in range(self.pca.n_components_)],
            index=df.columns
        )
        
        # Calculate cluster statistics
        cluster_stats = {}
        for cluster_id in range(self.kmeans.n_clusters):
            cluster_mask = clusters == cluster_id
            cluster_stats[cluster_id] = {
                'size': sum(cluster_mask),
                'mean_features': df[cluster_mask].mean().to_dict(),
                'std_features': df[cluster_mask].std().to_dict()
            }
        
        return {
            'pca_result': pca_result,
            'clusters': clusters,
            'explained_variance_ratio': self.pca.explained_variance_ratio_,
            'feature_importance': feature_importance,
            'cluster_stats': cluster_stats
        }
    
    def visualize_results(self, df, results, output_file):
        """Generate visualizations of the analysis results."""
        plt.figure(figsize=(20, 15))
        
        # Plot 1: 3D PCA scatter plot
        ax1 = plt.subplot(2, 2, 1, projection='3d')
        scatter = ax1.scatter(
            results['pca_result'][:, 0],
            results['pca_result'][:, 1],
            results['pca_result'][:, 2],
            c=results['clusters'],
            cmap='viridis'
        )
        ax1.set_title('3D PCA of Speech Features')
        ax1.set_xlabel(f'PC1 ({results["explained_variance_ratio"][0]:.2%} variance)')
        ax1.set_ylabel(f'PC2 ({results["explained_variance_ratio"][1]:.2%} variance)')
        ax1.set_zlabel(f'PC3 ({results["explained_variance_ratio"][2]:.2%} variance)')
        plt.colorbar(scatter, label='Cluster')
        
        # Plot 2: Feature importance heatmap
        plt.subplot(2, 2, 2)
        sns.heatmap(
            results['feature_importance'],
            cmap='RdBu',
            center=0,
            annot=True,
            fmt='.2f'
        )
        plt.title('Feature Importance in Principal Components')
        
        # Plot 3: Feature distributions by cluster
        plt.subplot(2, 2, 3)
        cluster_means = pd.DataFrame({
            'Cluster': results['clusters'],
            'PC1': results['pca_result'][:, 0],
            'PC2': results['pca_result'][:, 1],
            'PC3': results['pca_result'][:, 2]
        }).groupby('Cluster').mean()
        cluster_means.plot(kind='bar')
        plt.title('Average PC Values by Cluster')
        plt.xlabel('Cluster')
        plt.ylabel('Principal Component Value')
        
        # Plot 4: Feature correlations
        plt.subplot(2, 2, 4)
        sns.heatmap(
            df.corr(),
            cmap='coolwarm',
            center=0,
            annot=True,
            fmt='.2f'
        )
        plt.title('Feature Correlations')
        
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()
    
    def generate_report(self, df, results):
        """Generate a text report summarizing the analysis findings."""
        report = []
        
        # Overall statistics
        report.append("=== Cognitive Pattern Analysis Report ===\n")
        report.append(f"Number of samples analyzed: {len(df)}")
        report.append(f"Number of features: {df.shape[1]}\n")
        
        # PCA analysis
        report.append("Principal Component Analysis:")
        for i, ratio in enumerate(results['explained_variance_ratio']):
            report.append(f"- PC{i+1} explains {ratio:.2%} of variance")
        report.append(f"- Total variance explained: {sum(results['explained_variance_ratio']):.2%}\n")
        
        # Cluster analysis
        report.append("Cluster Analysis:")
        for cluster_id, stats in results['cluster_stats'].items():
            report.append(f"\nCluster {cluster_id} ({stats['size']} samples):")
            report.append("Key Features:")
            # Sort features by importance in PC1
            important_features = results['feature_importance']['PC1'].abs().sort_values(ascending=False)
            for feature in important_features.index[:5]:
                mean = stats['mean_features'][feature]
                std = stats['std_features'][feature]
                report.append(f"- {feature}: {mean:.3f} ± {std:.3f}")
        
        # Feature importance
        report.append("\nTop Features by Importance:")
        feature_importance = abs(results['feature_importance']).mean(axis=1).sort_values(ascending=False)
        for feature, importance in feature_importance.head().items():
            report.append(f"- {feature}: {importance:.3f}")
        
        return "\n".join(report) 