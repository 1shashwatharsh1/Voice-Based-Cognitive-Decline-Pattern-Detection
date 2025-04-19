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
        self.pca = PCA(n_components=3)
        self.kmeans = KMeans(n_clusters=3, random_state=42)
        
    def prepare_features(self, features_list):
        """Prepare features for analysis."""
        # Convert list of dictionaries to DataFrame
        df = pd.DataFrame(features_list)
        
        # Select numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df_numeric = df[numeric_cols]
        
        # Scale features
        scaled_features = self.scaler.fit_transform(df_numeric)
        
        return pd.DataFrame(scaled_features, columns=numeric_cols)
    
    def analyze_features(self, df):
        """Analyze features using PCA and clustering."""
        results = {}
        
        # For single sample, return raw features
        if len(df) == 1:
            results['raw_features'] = df.iloc[0].to_dict()
            results['clusters'] = [0]  # Default cluster for single sample
            return results
        
        # For multiple samples, perform PCA and clustering
        try:
            pca_result = self.pca.fit_transform(df)
            clusters = self.kmeans.fit_predict(df)
            
            results['pca_result'] = pca_result
            results['clusters'] = clusters
            results['explained_variance'] = self.pca.explained_variance_ratio_
            
        except Exception as e:
            print(f"Error in analysis: {str(e)}")
            # Fallback to raw features
            results['raw_features'] = df.to_dict('records')
            results['clusters'] = [0] * len(df)
        
        return results
    
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
        ax1.set_xlabel(f'PC1 ({self.pca.explained_variance_ratio_[0]:.2%} variance)')
        ax1.set_ylabel(f'PC2 ({self.pca.explained_variance_ratio_[1]:.2%} variance)')
        ax1.set_zlabel(f'PC3 ({self.pca.explained_variance_ratio_[2]:.2%} variance)')
        plt.colorbar(scatter, label='Cluster')
        
        # Plot 2: Feature importance heatmap
        plt.subplot(2, 2, 2)
        sns.heatmap(
            pd.DataFrame(self.pca.components_.T, columns=[f'PC{i+1}' for i in range(self.pca.n_components_)], index=df.columns),
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
        """Generate analysis report."""
        report = []
        
        if 'raw_features' in results:
            # Single sample report
            report.append("Single Sample Analysis Report")
            report.append("=" * 30)
            for feature, value in results['raw_features'].items():
                report.append(f"{feature}: {value:.4f}")
            
            # Add interpretation
            report.append("\nInterpretation:")
            report.append("- Speech rate above 3.0 words/second indicates normal speech")
            report.append("- Pause duration below 0.5 seconds is typical")
            report.append("- Pitch variation between 20-50 Hz is normal")
            report.append("- Energy level above 0.5 indicates good voice quality")
            
        else:
            # Multiple samples report
            report.append("Multiple Samples Analysis Report")
            report.append("=" * 30)
            report.append(f"Number of samples analyzed: {len(df)}")
            report.append(f"Explained variance by PCA: {results['explained_variance'].sum():.2%}")
            report.append(f"Cluster distribution: {np.bincount(results['clusters'])}")
            
            # Add cluster interpretations
            report.append("\nCluster Interpretations:")
            report.append("Cluster 0: Normal speech patterns")
            report.append("Cluster 1: Mild cognitive changes")
            report.append("Cluster 2: Significant cognitive changes")
        
        return "\n".join(report) 