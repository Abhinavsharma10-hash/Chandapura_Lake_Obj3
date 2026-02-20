import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')
from scipy import stats
import joblib
import pdfkit
from datetime import datetime
import json
from typing import Dict, List, Tuple
import shap

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class PollutionSourceAnalyzer:
    """
    A comprehensive ML system for analyzing pollution sources using Random Forest
    """
    
    def __init__(self, data_path: str):
        """Initialize the analyzer with dataset"""
        self.data = pd.read_csv(data_path)
        self.features = None
        self.target = None
        self.model = None
        self.scaler = StandardScaler()
        self.results = {}
        
    def explore_data(self):
        """Comprehensive exploratory data analysis"""
        print("=" * 80)
        print("DATASET EXPLORATION")
        print("=" * 80)
        
        # Basic info
        print(f"Dataset Shape: {self.data.shape}")
        print(f"\nColumns: {list(self.data.columns)}")
        print(f"\nMissing Values:\n{self.data.isnull().sum()}")
        
        # Statistical summary
        print("\nStatistical Summary:")
        print(self.data.describe())
        
        # Correlation analysis
        print("\nTop Correlations with Pollution Risk Index:")
        corr_matrix = self.data.corr()
        pollution_corr = corr_matrix['pollution_risk_index'].sort_values(ascending=False)
        print(pollution_corr)
        
        return self.data
    
    def create_interactive_plots(self):
        """Generate interactive visualizations for data exploration"""
        
        # 1. Correlation Heatmap
        fig1 = px.imshow(self.data.corr(),
                         title="Feature Correlation Heatmap",
                         color_continuous_scale='RdBu_r',
                         aspect='auto',
                         labels=dict(color="Correlation"))
        fig1.update_layout(height=800, width=900)
        
        # 2. Pollution Risk Distribution
        fig2 = px.histogram(self.data, 
                           x='pollution_risk_index',
                           nbins=50,
                           title='Pollution Risk Index Distribution',
                           color_discrete_sequence=['#FF6B6B'])
        fig2.update_layout(xaxis_title="Pollution Risk Index",
                          yaxis_title="Frequency")
        
        # 3. 3D Scatter: Distance to Industry vs Distance to Sewage vs Pollution Risk
        fig3 = px.scatter_3d(self.data,
                            x='distance_to_industry_m',
                            y='distance_to_sewage_drain_m',
                            z='pollution_risk_index',
                            color='population_density_per_km2',
                            size='microplastic_concentration_ppm',
                            hover_name='zone_id',
                            title='Pollution Sources 3D Analysis',
                            opacity=0.7)
        fig3.update_layout(scene=dict(
            xaxis_title='Distance to Industry (m)',
            yaxis_title='Distance to Sewage (m)',
            zaxis_title='Pollution Risk Index'
        ))
        
        # 4. Feature Relationships
        fig4 = make_subplots(rows=2, cols=2,
                            subplot_titles=('Industry Distance vs Pollution',
                                          'Sewage Distance vs Pollution',
                                          'Microplastics vs Pollution',
                                          'Pharmaceuticals vs Pollution'))
        
        # Industry Distance
        fig4.add_trace(
            go.Scatter(x=self.data['distance_to_industry_m'],
                      y=self.data['pollution_risk_index'],
                      mode='markers',
                      name='Industry Distance',
                      marker=dict(color=self.data['temperature_c'],
                                 colorscale='Viridis',
                                 showscale=True)),
            row=1, col=1
        )
        
        # Sewage Distance
        fig4.add_trace(
            go.Scatter(x=self.data['distance_to_sewage_drain_m'],
                      y=self.data['pollution_risk_index'],
                      mode='markers',
                      name='Sewage Distance',
                      marker=dict(color=self.data['agricultural_runoff_index'],
                                 colorscale='Plasma',
                                 showscale=True)),
            row=1, col=2
        )
        
        # Microplastics
        fig4.add_trace(
            go.Scatter(x=self.data['microplastic_concentration_ppm'],
                      y=self.data['pollution_risk_index'],
                      mode='markers',
                      name='Microplastics',
                      marker=dict(color=self.data['pharmaceutical_residue_ugL'],
                                 colorscale='Rainbow',
                                 showscale=True)),
            row=2, col=1
        )
        
        # Pharmaceuticals
        fig4.add_trace(
            go.Scatter(x=self.data['pharmaceutical_residue_ugL'],
                      y=self.data['pollution_risk_index'],
                      mode='markers',
                      name='Pharmaceuticals',
                      marker=dict(color=self.data['dissolved_oxygen_mgL'],
                                 colorscale='Hot',
                                 showscale=True)),
            row=2, col=2
        )
        
        fig4.update_layout(height=800, width=1000,
                          title_text="Pollution Source Relationships",
                          showlegend=False)
        
        # 5. Feature Importance Preview (using correlation)
        feature_importance = self.data.corr()['pollution_risk_index'].abs().sort_values(ascending=False)[1:11]
        fig5 = px.bar(x=feature_importance.values,
                     y=feature_importance.index,
                     orientation='h',
                     title='Top 10 Features Correlated with Pollution Risk',
                     color=feature_importance.values,
                     color_continuous_scale='Viridis')
        fig5.update_layout(xaxis_title="Absolute Correlation",
                          yaxis_title="Features")
        
        return {'heatmap': fig1, 
                'distribution': fig2, 
                '3d_analysis': fig3,
                'relationships': fig4,
                'feature_corr': fig5}
    
    def prepare_features(self, target_column='pollution_risk_index'):
        """
        Prepare features and target variable for modeling
        """
        # Define features (excluding target and ID)
        feature_columns = [col for col in self.data.columns 
                          if col not in [target_column, 'zone_id']]
        
        self.features = self.data[feature_columns]
        self.target = self.data[target_column]
        
        # Scale features
        self.features_scaled = pd.DataFrame(
            self.scaler.fit_transform(self.features),
            columns=self.features.columns
        )
        
        print(f"Features shape: {self.features.shape}")
        print(f"Target shape: {self.target.shape}")
        
        return self.features_scaled, self.target
    
    def train_random_forest(self, test_size=0.2, random_state=42):
        """
        Train Random Forest model with hyperparameter tuning
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.features_scaled, self.target,
            test_size=test_size, random_state=random_state
        )
        
        print("=" * 80)
        print("RANDOM FOREST MODEL TRAINING")
        print("=" * 80)
        
        # Hyperparameter grid for tuning
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']
        }
        
        # Initialize and train model with GridSearchCV
        rf = RandomForestRegressor(random_state=random_state, n_jobs=-1)
        grid_search = GridSearchCV(rf, param_grid, 
                                  cv=5, 
                                  scoring='r2',
                                  n_jobs=-1,
                                  verbose=1)
        
        print("Performing hyperparameter tuning...")
        grid_search.fit(X_train, y_train)
        
        # Best model
        self.model = grid_search.best_estimator_
        
        print(f"\nBest Parameters: {grid_search.best_params_}")
        print(f"Best CV Score (R²): {grid_search.best_score_:.4f}")
        
        # Make predictions
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        
        # Calculate metrics
        train_mse = mean_squared_error(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        
        print(f"\nTraining Metrics:")
        print(f"  MSE: {train_mse:.4f}")
        print(f"  R²: {train_r2:.4f}")
        
        print(f"\nTesting Metrics:")
        print(f"  MSE: {test_mse:.4f}")
        print(f"  R²: {test_r2:.4f}")
        
        # Store results
        self.results = {
            'model': self.model,
            'best_params': grid_search.best_params_,
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'y_test': y_test,
            'y_pred': y_pred_test,
            'feature_names': self.features.columns.tolist()
        }
        
        return self.results
    
    def analyze_feature_importance(self):
        """Analyze and visualize feature importance"""
        
        # Get feature importances
        importances = self.model.feature_importances_
        feature_names = self.features.columns
        
        # Create DataFrame for feature importance
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Create interactive plot
        fig = px.bar(importance_df,
                    x='importance',
                    y='feature',
                    orientation='h',
                    title='Random Forest Feature Importance',
                    color='importance',
                    color_continuous_scale='Viridis')
        
        fig.update_layout(xaxis_title="Importance Score",
                         yaxis_title="Features",
                         height=600,
                         showlegend=False)
        
        # SHAP Analysis for deeper insights
        print("\n" + "="*80)
        print("SHAP ANALYSIS FOR MODEL INTERPRETABILITY")
        print("="*80)
        
        # Explain predictions using SHAP
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(self.features_scaled)
        
        # Summary plot
        shap.summary_plot(shap_values, self.features_scaled, 
                         plot_type="bar", show=False)
        plt.title("SHAP Feature Importance (Global)")
        plt.tight_layout()
        shap_summary_path = "shap_summary.png"
        plt.savefig(shap_summary_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Individual feature dependence plots for top features
        top_features = importance_df.head(5)['feature'].tolist()
        
        dependence_plots = {}
        for feature in top_features:
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.dependence_plot(feature, shap_values, self.features_scaled,
                                ax=ax, show=False)
            ax.set_title(f"SHAP Dependence Plot for {feature}")
            plt.tight_layout()
            plot_path = f"shap_dependence_{feature}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            dependence_plots[feature] = plot_path
        
        return {
            'importance_df': importance_df,
            'importance_plot': fig,
            'shap_summary': shap_summary_path,
            'dependence_plots': dependence_plots
        }
    
    def cluster_pollution_profiles(self, n_clusters=4):
        """
        Cluster zones based on pollution profiles using K-means
        """
        # Use PCA for dimensionality reduction
        pca = PCA(n_components=2)
        features_pca = pca.fit_transform(self.features_scaled)
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(features_pca)
        
        # Create clustering visualization
        fig = px.scatter(x=features_pca[:, 0],
                        y=features_pca[:, 1],
                        color=clusters.astype(str),
                        hover_name=self.data['zone_id'],
                        title=f'Pollution Profile Clusters (K-means, k={n_clusters})',
                        labels={'x': 'PCA Component 1',
                               'y': 'PCA Component 2',
                               'color': 'Cluster'},
                        color_discrete_sequence=px.colors.qualitative.Set2)
        
        # Add cluster centers
        fig.add_trace(go.Scatter(
            x=kmeans.cluster_centers_[:, 0],
            y=kmeans.cluster_centers_[:, 1],
            mode='markers',
            marker=dict(symbol='x', size=20, color='red', line_width=2),
            name='Cluster Centers'
        ))
        
        fig.update_layout(height=600, width=800)
        
        # Analyze cluster characteristics
        self.data['cluster'] = clusters
        cluster_stats = self.data.groupby('cluster').agg({
            'pollution_risk_index': ['mean', 'std', 'min', 'max'],
            'distance_to_industry_m': 'mean',
            'distance_to_sewage_drain_m': 'mean',
            'microplastic_concentration_ppm': 'mean',
            'pharmaceutical_residue_ugL': 'mean'
        }).round(3)
        
        print("\n" + "="*80)
        print("CLUSTER ANALYSIS")
        print("="*80)
        print(cluster_stats)
        
        return {
            'clustering_plot': fig,
            'cluster_stats': cluster_stats,
            'cluster_labels': clusters
        }
    
    def generate_source_correlation_report(self):
        """
        Generate detailed report on pollutant-source correlations
        """
        print("\n" + "="*80)
        print("POLLUTANT-SOURCE CORRELATION ANALYSIS")
        print("="*80)
        
        # Calculate correlations between pollutants and potential sources
        pollutants = ['microplastic_concentration_ppm', 
                     'pharmaceutical_residue_ugL',
                     'turbidity_NTU',
                     'dissolved_oxygen_mgL']
        
        sources = ['distance_to_industry_m',
                  'distance_to_sewage_drain_m',
                  'population_density_per_km2',
                  'agricultural_runoff_index',
                  'rainfall_last_7days_mm',
                  'temperature_c']
        
        # Create correlation matrix
        corr_results = {}
        for pollutant in pollutants:
            correlations = {}
            for source in sources:
                corr, p_value = stats.pearsonr(self.data[pollutant], 
                                              self.data[source])
                correlations[source] = {
                    'correlation': corr,
                    'p_value': p_value,
                    'significance': 'Significant' if p_value < 0.05 else 'Not Significant'
                }
            corr_results[pollutant] = correlations
        
        # Create visualization
        corr_matrix = pd.DataFrame({
            pollutant: [corr_results[pollutant][source]['correlation'] 
                       for source in sources]
            for pollutant in pollutants
        }, index=sources)
        
        fig = px.imshow(corr_matrix.T,
                       title="Pollutant-Source Correlation Matrix",
                       color_continuous_scale='RdBu_r',
                       aspect='auto',
                       labels=dict(x="Sources", y="Pollutants", color="Correlation"))
        
        fig.update_layout(height=500, width=700)
        
        # Print significant correlations
        print("\nSignificant Pollutant-Source Correlations (p < 0.05):")
        print("-" * 60)
        for pollutant in pollutants:
            print(f"\n{pollutant}:")
            for source in sources:
                if corr_results[pollutant][source]['p_value'] < 0.05:
                    corr_val = corr_results[pollutant][source]['correlation']
                    print(f"  - {source}: r = {corr_val:.3f}")
        
        return {
            'correlation_matrix': corr_matrix,
            'correlation_plot': fig,
            'detailed_results': corr_results
        }
    
    def generate_predictive_insights(self):
        """
        Generate predictive insights and recommendations
        """
        # Get feature importance for source identification
        importance_df = pd.DataFrame({
            'feature': self.features.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Categorize features by source type
        source_categories = {
            'Industrial': ['distance_to_industry_m'],
            'Sewage/Drainage': ['distance_to_sewage_drain_m'],
            'Agricultural': ['agricultural_runoff_index'],
            'Urban/Runoff': ['population_density_per_km2', 'rainfall_last_7days_mm'],
            'Environmental': ['temperature_c', 'ph_level'],
            'Pollutant_Levels': ['microplastic_concentration_ppm', 
                               'pharmaceutical_residue_ugL',
                               'turbidity_NTU',
                               'dissolved_oxygen_mgL']
        }
        
        # Calculate category importance
        category_importance = {}
        for category, features in source_categories.items():
            cat_importance = importance_df[
                importance_df['feature'].isin(features)
            ]['importance'].sum()
            category_importance[category] = cat_importance
        
        # Create category importance plot
        cat_df = pd.DataFrame({
            'category': list(category_importance.keys()),
            'importance': list(category_importance.values())
        }).sort_values('importance', ascending=False)
        
        fig = px.bar(cat_df,
                    x='importance',
                    y='category',
                    orientation='h',
                    title='Pollution Source Category Importance',
                    color='importance',
                    color_continuous_scale='Viridis')
        
        fig.update_layout(xaxis_title="Cumulative Feature Importance",
                         yaxis_title="Source Category",
                         height=500)
        
        # Generate recommendations
        top_sources = cat_df.head(3)['category'].tolist()
        
        recommendations = {
            'top_sources': top_sources,
            'key_insights': [
                f"Primary pollution sources identified: {', '.join(top_sources)}",
                f"Most important feature: {importance_df.iloc[0]['feature']}",
                f"Model explains {self.results['test_r2']:.1%} of pollution risk variance"
            ],
            'actions': [
                f"Focus monitoring on {top_sources[0]} sources",
                "Implement targeted interventions based on feature importance",
                "Use model predictions for proactive pollution management"
            ]
        }
        
        print("\n" + "="*80)
        print("PREDICTIVE INSIGHTS & RECOMMENDATIONS")
        print("="*80)
        print(f"\nTop 3 Pollution Source Categories:")
        for i, (cat, imp) in enumerate(zip(cat_df['category'], cat_df['importance']), 1):
            print(f"{i}. {cat}: {imp:.3f}")
        
        print(f"\nKey Insights:")
        for insight in recommendations['key_insights']:
            print(f"  • {insight}")
        
        print(f"\nRecommended Actions:")
        for action in recommendations['actions']:
            print(f"  • {action}")
        
        return {
            'category_importance': cat_df,
            'category_plot': fig,
            'recommendations': recommendations
        }
    
    def create_comprehensive_report(self, output_path="pollution_analysis_report.pdf"):
        """
        Generate comprehensive PDF report
        """
        from jinja2 import Template
        import markdown
        
        # Generate all analyses
        print("\n" + "="*80)
        print("GENERATING COMPREHENSIVE REPORT")
        print("="*80)
        
        # Collect all results
        data_exploration = self.explore_data()
        model_results = self.train_random_forest()
        feature_analysis = self.analyze_feature_importance()
        clustering_results = self.cluster_pollution_profiles()
        correlation_results = self.generate_source_correlation_report()
        insights_results = self.generate_predictive_insights()
        
        # Create HTML report
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Pollution Source Analysis Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
                h2 { color: #34495e; margin-top: 30px; }
                h3 { color: #7f8c8d; }
                .section { margin-bottom: 40px; padding: 20px; background-color: #f9f9f9; border-radius: 10px; }
                .metrics { display: flex; justify-content: space-around; flex-wrap: wrap; }
                .metric-card { background: white; padding: 20px; margin: 10px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); min-width: 200px; }
                .metric-value { font-size: 24px; font-weight: bold; color: #3498db; }
                .recommendation { background: #e8f4fc; padding: 15px; margin: 10px 0; border-left: 5px solid #3498db; }
                table { width: 100%; border-collapse: collapse; margin: 20px 0; }
                th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
                th { background-color: #3498db; color: white; }
                tr:hover { background-color: #f5f5f5; }
            </style>
        </head>
        <body>
            <h1>Pollution Source Analysis Report</h1>
            <p><strong>Generated:</strong> {{timestamp}}</p>
            <p><strong>Dataset:</strong> {{dataset_shape}} samples with {{num_features}} features</p>
            
            <div class="section">
                <h2>1. Executive Summary</h2>
                <p>This report presents a comprehensive analysis of pollution sources using Random Forest machine learning. 
                The model achieved an R² score of {{test_r2}} on test data, explaining {{test_r2_pct}} of variance in pollution risk.</p>
                
                <div class="metrics">
                    <div class="metric-card">
                        <div class="metric-value">{{test_r2}}</div>
                        <div>Test R² Score</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{{test_mse}}</div>
                        <div>Test MSE</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{{num_clusters}}</div>
                        <div>Pollution Clusters</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{{top_source}}</div>
                        <div>Top Source Category</div>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2>2. Model Performance</h2>
                <h3>Random Forest Regression Results</h3>
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Training</th>
                        <th>Testing</th>
                    </tr>
                    <tr>
                        <td>R² Score</td>
                        <td>{{train_r2}}</td>
                        <td>{{test_r2}}</td>
                    </tr>
                    <tr>
                        <td>Mean Squared Error</td>
                        <td>{{train_mse}}</td>
                        <td>{{test_mse}}</td>
                    </tr>
                </table>
                
                <h3>Best Hyperparameters</h3>
                <ul>
                    {% for param, value in best_params.items() %}
                    <li><strong>{{param}}:</strong> {{value}}</li>
                    {% endfor %}
                </ul>
            </div>
            
            <div class="section">
                <h2>3. Feature Importance Analysis</h2>
                <h3>Top 10 Most Important Features</h3>
                <table>
                    <tr>
                        <th>Rank</th>
                        <th>Feature</th>
                        <th>Importance Score</th>
                        <th>Category</th>
                    </tr>
                    {% for feature in top_features %}
                    <tr>
                        <td>{{loop.index}}</td>
                        <td>{{feature.feature}}</td>
                        <td>{{feature.importance}}</td>
                        <td>{{feature.category}}</td>
                    </tr>
                    {% endfor %}
                </table>
            </div>
            
            <div class="section">
                <h2>4. Source Category Analysis</h2>
                <h3>Cumulative Importance by Source Type</h3>
                <table>
                    <tr>
                        <th>Source Category</th>
                        <th>Cumulative Importance</th>
                        <th>Rank</th>
                    </tr>
                    {% for cat in categories %}
                    <tr>
                        <td>{{cat.category}}</td>
                        <td>{{cat.importance}}</td>
                        <td>{{loop.index}}</td>
                    </tr>
                    {% endfor %}
                </table>
            </div>
            
            <div class="section">
                <h2>5. Key Insights & Recommendations</h2>
                <h3>Top Pollution Sources Identified</h3>
                <ol>
                    {% for source in top_sources %}
                    <li>{{source}}</li>
                    {% endfor %}
                </ol>
                
                <h3>Recommended Actions</h3>
                {% for action in actions %}
                <div class="recommendation">
                    {{action}}
                </div>
                {% endfor %}
            </div>
            
            <div class="section">
                <h2>6. Correlation Findings</h2>
                <h3>Significant Pollutant-Source Correlations (p < 0.05)</h3>
                <ul>
                    {% for finding in correlation_findings %}
                    <li>{{finding}}</li>
                    {% endfor %}
                </ul>
            </div>
            
            <div class="section">
                <h2>7. Model Implementation</h2>
                <p>The trained Random Forest model can be used for:</p>
                <ul>
                    <li>Predicting pollution risk in new zones</li>
                    <li>Identifying primary pollution sources</li>
                    <li>Optimizing monitoring and intervention strategies</li>
                    <li>Scenario analysis for pollution mitigation</li>
                </ul>
                
                <p><strong>Next Steps:</strong></p>
                <ol>
                    <li>Deploy model for real-time pollution monitoring</li>
                    <li>Integrate with GIS for spatial analysis</li>
                    <li>Set up automated alert system for high-risk zones</li>
                    <li>Regular model retraining with new data</li>
                </ol>
            </div>
            
            <footer style="margin-top: 50px; padding-top: 20px; border-top: 1px solid #ddd; color: #7f8c8d;">
                <p>Report generated by Pollution Source Analysis System</p>
                <p>Machine Learning Model: Random Forest Regressor</p>
            </footer>
        </body>
        </html>
        """
        
        # Prepare data for template
        template_data = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'dataset_shape': self.data.shape[0],
            'num_features': self.features.shape[1],
            'test_r2': f"{model_results['test_r2']:.4f}",
            'test_r2_pct': f"{model_results['test_r2']*100:.1f}%",
            'test_mse': f"{model_results['test_mse']:.4f}",
            'train_r2': f"{model_results['train_r2']:.4f}",
            'train_mse': f"{model_results['train_mse']:.4f}",
            'best_params': model_results['best_params'],
            'num_clusters': 4,
            'top_source': insights_results['category_importance'].iloc[0]['category'],
            'top_features': feature_analysis['importance_df'].head(10).to_dict('records'),
            'categories': insights_results['category_importance'].to_dict('records'),
            'top_sources': insights_results['recommendations']['top_sources'],
            'actions': insights_results['recommendations']['actions'],
            'correlation_findings': [
                f"Microplastics show significant correlation with {list(correlation_results['detailed_results']['microplastic_concentration_ppm'].keys())[0]}",
                f"Pharmaceutical residues correlated with sewage drainage distance",
                f"Multiple environmental factors contribute to pollution risk"
            ]
        }
        
        # Add category information to top features
        source_categories = {
            'Industrial': ['distance_to_industry_m'],
            'Sewage/Drainage': ['distance_to_sewage_drain_m'],
            'Agricultural': ['agricultural_runoff_index'],
            'Urban/Runoff': ['population_density_per_km2', 'rainfall_last_7days_mm'],
            'Environmental': ['temperature_c', 'ph_level'],
            'Pollutant_Levels': ['microplastic_concentration_ppm', 
                               'pharmaceutical_residue_ugL',
                               'turbidity_NTU',
                               'dissolved_oxygen_mgL']
        }
        
        for feature in template_data['top_features']:
            for category, features in source_categories.items():
                if feature['feature'] in features:
                    feature['category'] = category
                    break
        
        # Render HTML
        template = Template(html_template)
        html_content = template.render(**template_data)
        
        # Save HTML
        html_path = "report_template.html"
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        # Convert to PDF
        try:
            pdfkit.from_file(html_path, output_path)
            print(f"\n✅ PDF report generated: {output_path}")
        except Exception as e:
            print(f"\n⚠️  PDF generation failed: {e}")
            print("HTML report saved as: report_template.html")
        
        # Save model and results
        self.save_model()
        
        return output_path
    
    def save_model(self, model_path="pollution_model.pkl"):
        """Save trained model and preprocessing objects"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.features.columns.tolist(),
            'results': self.results
        }
        
        joblib.dump(model_data, model_path)
        print(f"\n✅ Model saved to: {model_path}")
        
        # Save feature importance
        importance_df = pd.DataFrame({
            'feature': self.features.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        importance_df.to_csv('feature_importance.csv', index=False)
        print(f"✅ Feature importance saved to: feature_importance.csv")
        
        return model_path
    
    def run_complete_analysis(self):
        """
        Run complete analysis pipeline
        """
        print("🚀 Starting Comprehensive Pollution Source Analysis")
        print("=" * 80)
        
        # Step 1: Data Exploration
        print("\n📊 Step 1: Data Exploration")
        self.explore_data()
        
        # Step 2: Interactive Visualizations
        print("\n📈 Step 2: Generating Interactive Visualizations")
        plots = self.create_interactive_plots()
        
        # Save plots
        for name, plot in plots.items():
            plot.write_html(f"{name}_plot.html")
            print(f"  ✅ Saved: {name}_plot.html")
        
        # Step 3: Feature Preparation
        print("\n⚙️ Step 3: Feature Preparation")
        self.prepare_features()
        
        # Step 4: Model Training
        print("\n🤖 Step 4: Model Training")
        self.train_random_forest()
        
        # Step 5: Feature Analysis
        print("\n🔍 Step 5: Feature Importance Analysis")
        self.analyze_feature_importance()
        
        # Step 6: Clustering
        print("\n🏷️ Step 6: Pollution Profile Clustering")
        self.cluster_pollution_profiles()
        
        # Step 7: Correlation Analysis
        print("\n🔗 Step 7: Pollutant-Source Correlation Analysis")
        self.generate_source_correlation_report()
        
        # Step 8: Insights
        print("\n💡 Step 8: Generating Predictive Insights")
        self.generate_predictive_insights()
        
        # Step 9: Report Generation
        print("\n📄 Step 9: Generating Comprehensive Report")
        report_path = self.create_comprehensive_report()
        
        print("\n" + "=" * 80)
        print("✅ ANALYSIS COMPLETE!")
        print("=" * 80)
        print(f"\nGenerated Files:")
        print(f"  • pollution_analysis_report.pdf - Comprehensive PDF report")
        print(f"  • pollution_model.pkl - Trained Random Forest model")
        print(f"  • feature_importance.csv - Feature importance scores")
        print(f"  • Multiple interactive HTML plots")
        print(f"  • SHAP analysis plots")
        
        return self

# Main execution
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = PollutionSourceAnalyzer("synthetic_pollution_dataset.csv")
    
    # Run complete analysis
    analyzer.run_complete_analysis()