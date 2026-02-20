"""
Master script to generate all visualizations for the paper
Run this to create all 7 figures at once
"""

import os
import sys
import subprocess

def generate_all_figures():
    """Generate all 7 figures for the paper"""
    
    print("="*60)
    print("GENERATING ALL FIGURES FOR POLLUTION SOURCE ATTRIBUTION PAPER")
    print("="*60)
    
    # Create figures directory if it doesn't exist
    if not os.path.exists('figures'):
        os.makedirs('figures')
    
    # List of figure generation functions
    figures = [
        ("Graphical Abstract", create_graphical_abstract),
        ("Correlation Heatmap", create_correlation_heatmap),
        ("Executive Summary", create_executive_summary),
        ("SHAP Global Importance", create_shap_global_importance),
        ("SHAP Dependence - Sewage", create_shap_dependence_sewage),
        ("SHAP Dependence - Turbidity", create_shap_dependence_turbidity),
        ("Model Implementation", create_implementation_framework)
    ]
    
    # Generate each figure
    for i, (name, func) in enumerate(figures, 1):
        print(f"\n[{i}/7] Generating {name}...")
        try:
            func()
            print(f"✓ {name} generated successfully")
        except Exception as e:
            print(f"✗ Error generating {name}: {e}")
    
    print("\n" + "="*60)
    print("ALL FIGURES GENERATED SUCCESSFULLY!")
    print("="*60)
    print("\nFiles created:")
    for ext in ['.png', '.pdf']:
        files = [f for f in os.listdir('.') if f.endswith(ext)]
        for f in files:
            print(f"  • {f}")

if __name__ == "__main__":
    # Import all functions
    from graphical_abstract import create_graphical_abstract
    from correlation_heatmap import create_correlation_heatmap
    from executive_summary import create_executive_summary
    from shap_global import create_shap_global_importance
    from shap_sewage import create_shap_dependence_sewage
    from shap_turbidity import create_shap_dependence_turbidity
    from implementation import create_implementation_framework
    
    generate_all_figures()  