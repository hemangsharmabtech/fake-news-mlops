#!/usr/bin/env python3
import json
import pandas as pd
from tabulate import tabulate

print("=" * 70)
print("FAKE NEWS DETECTION - MODEL COMPARISON REPORT")
print("=" * 70)

# Load metrics from both experiments
try:
    with open('model/metrics.json', 'r') as f:
        lr_metrics = json.load(f)
    print("✓ Loaded Linear Regression metrics")
except Exception as e:
    print(f"✗ Failed to load Linear Regression metrics: {e}")
    lr_metrics = {}

try:
    with open('metrics/rf/metrics.json', 'r') as f:
        rf_metrics = json.load(f)
    print("✓ Loaded Random Forest metrics")
except Exception as e:
    print(f"✗ Failed to load Random Forest metrics: {e}")
    rf_metrics = {}

# Create comparison table
comparison_data = []
metrics_to_compare = [
    'accuracy', 'precision_fake', 'recall_fake', 'f1_fake', 
    'precision_true', 'recall_true', 'f1_true'
]

for metric in metrics_to_compare:
    if metric in lr_metrics and metric in rf_metrics:
        lr_val = lr_metrics.get(metric, 0)
        rf_val = rf_metrics.get(metric, 0)
        diff = rf_val - lr_val
        improvement = "✓" if diff > 0 else "✗" if diff < 0 else "="
        
        comparison_data.append({
            'Metric': metric.replace('_', ' ').title(),
            'Linear Regression': f"{lr_val:.4f}",
            'Random Forest': f"{rf_val:.4f}",
            'Difference': f"{diff:+.4f}",
            'Improvement': improvement
        })

# Display comparison table
print("\n" + "=" * 70)
print("PERFORMANCE COMPARISON")
print("=" * 70)
print(tabulate(comparison_data, headers='keys', tablefmt='grid', showindex=False))

# Summary
print("\n" + "=" * 70)
print("EXPERIMENT SUMMARY")
print("=" * 70)
print(f"Linear Regression Accuracy: {lr_metrics.get('accuracy', 'N/A')}")
print(f"Random Forest Accuracy:     {rf_metrics.get('accuracy', 'N/A')}")

if 'accuracy' in lr_metrics and 'accuracy' in rf_metrics:
    accuracy_diff = rf_metrics['accuracy'] - lr_metrics['accuracy']
    print(f"Accuracy Improvement:       {accuracy_diff:+.4f} ({accuracy_diff*100:+.2f}%)")
    
    if accuracy_diff > 0:
        print("🎯 CONCLUSION: Random Forest performs BETTER than Linear Regression")
    elif accuracy_diff < 0:
        print("🎯 CONCLUSION: Linear Regression performs BETTER than Random Forest")
    else:
        print("🎯 CONCLUSION: Both models perform EQUALLY")

print(f"\nTraining Samples: {rf_metrics.get('training_samples', 'N/A')}")
print(f"Test Samples:     {rf_metrics.get('test_samples', 'N/A')}")
print("=" * 70)
