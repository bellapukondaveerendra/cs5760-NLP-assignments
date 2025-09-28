import numpy as np

def calculate_metrics_from_confusion_matrix(confusion_matrix, class_names):
    
    # Convert to numpy array for easier calculation
    cm = np.array(confusion_matrix)
    n_classes = len(class_names)
    
    # Calculate per-class metrics
    per_class_metrics = {}
    
    for i, class_name in enumerate(class_names):
        # True Positives: diagonal element
        tp = cm[i, i]
        
        # False Positives: sum of row i excluding diagonal
        fp = np.sum(cm[i, :]) - tp
        
        # False Negatives: sum of column i excluding diagonal  
        fn = np.sum(cm[:, i]) - tp
        
        # True Negatives: everything else
        tn = np.sum(cm) - tp - fp - fn
        
        # Calculate precision and recall
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        per_class_metrics[class_name] = {
            'precision': precision,
            'recall': recall,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn
        }
    
    # Calculate macro-averaged metrics
    macro_precision = np.mean([metrics['precision'] for metrics in per_class_metrics.values()])
    macro_recall = np.mean([metrics['recall'] for metrics in per_class_metrics.values()])
    
    # Calculate micro-averaged metrics
    total_tp = sum([metrics['tp'] for metrics in per_class_metrics.values()])
    total_fp = sum([metrics['fp'] for metrics in per_class_metrics.values()])
    total_fn = sum([metrics['fn'] for metrics in per_class_metrics.values()])
    
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    
    return {
        'per_class': per_class_metrics,
        'macro': {'precision': macro_precision, 'recall': macro_recall},
        'micro': {'precision': micro_precision, 'recall': micro_recall}
    }

def print_results(results, class_names):
    """Print all results in a clear format"""
    
    print("=== PER-CLASS METRICS ===")
    print(f"{'Class':<8} {'Precision':<10} {'Recall':<8}")
    print("-" * 30)
    
    for class_name in class_names:
        metrics = results['per_class'][class_name]
        print(f"{class_name:<8} {metrics['precision']:<10.3f} {metrics['recall']:<8.3f}")
    
    print("\n=== MACRO vs MICRO AVERAGING ===")
    print(f"Macro-averaged Precision: {results['macro']['precision']:.3f}")
    print(f"Macro-averaged Recall:    {results['macro']['recall']:.3f}")
    print()
    print(f"Micro-averaged Precision: {results['micro']['precision']:.3f}")
    print(f"Micro-averaged Recall:    {results['micro']['recall']:.3f}")
    
    print("\n=== INTERPRETATION ===")
    print("Macro averaging: Treats each class equally (unweighted average)")
    print("Micro averaging: Treats each instance equally (weighted by class frequency)")
    print("Micro-averaging gives more weight to classes with more instances.")

# Define the confusion matrix from the problem
# Rows: System predictions, Columns: Gold standard
confusion_matrix = [
    [5, 10, 5],    # System predicted Cat
    [15, 20, 10],  # System predicted Dog  
    [0, 15, 10]    # System predicted Rabbit
]

class_names = ['Cat', 'Dog', 'Rabbit']

print("CONFUSION MATRIX:")
print("System \\ Gold    Cat   Dog   Rabbit")
print("Cat              5     10    5")
print("Dog              15    20    10") 
print("Rabbit           0     15    10")
print()

# Calculate and print results
results = calculate_metrics_from_confusion_matrix(confusion_matrix, class_names)
print_results(results, class_names)