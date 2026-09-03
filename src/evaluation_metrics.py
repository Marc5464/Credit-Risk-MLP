import numpy as np
import matplotlib.pyplot as plt
from src.training_model import forward_pass, compute_loss, train_network
from src.optimisers import MomentumGD

def calculate_pr_roc_data(Yhat, Y, num_thresholds=100):
    thresholds = np.linspace(0.0, 1.0, num_thresholds)
    recalls, precisions, fpr_list = [], [], []
    
    for t in thresholds:
        Yhat_converted = (Yhat >= t).astype(int)
        tp = np.sum((Y + Yhat_converted) == 2)
        fp = np.sum((Yhat_converted - Y) == 1)
        fn = np.sum((Yhat_converted - Y) == -1)
        tn = np.sum((Yhat_converted + Y) == 0)
        
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        recalls.append(recall)
        precisions.append(precision)
        fpr_list.append(fpr)
        
    pr_recalls = np.array(recalls + [0.0])
    pr_precisions = np.array(precisions + [1.0])
    
    pr_sort_idx = np.argsort(pr_recalls)
    pr_recalls_sorted = pr_recalls[pr_sort_idx]
    pr_precisions_sorted = pr_precisions[pr_sort_idx]
    
    roc_fpr = np.array(fpr_list)
    roc_tpr = np.array(recalls)  
    
    roc_sort_idx = np.argsort(roc_fpr)
    roc_fpr_sorted = roc_fpr[roc_sort_idx]
    roc_tpr_sorted = roc_tpr[roc_sort_idx]
    
    return pr_recalls_sorted, pr_precisions_sorted, roc_fpr_sorted, roc_tpr_sorted

def train_and_eval_single(X_tr, y_tr, X_va, y_va, pos_w, H=16):
    input_dim = X_tr.shape[1]
    
    W1 = np.random.randn(input_dim, H) * np.sqrt(2.0 / input_dim)
    W2 = np.random.randn(H + 1, 1) * np.sqrt(2.0 / (H + 1))
    
    momentum = MomentumGD(0.01, 0.9)
    bestW1, bestW2, _, _ = train_network(
        X_tr, y_tr, X_va, y_va, W1, W2, momentum, pos_w, 
        epochs=1000, batch=512, use_early_stopping=True
    )
    
    _, _, Yhat_val = forward_pass(X_va, bestW1, bestW2)
    recalls_pr, precisions, fpr, tpr = calculate_pr_roc_data(Yhat_val, y_va)
    
    pr_auc = np.abs(np.trapezoid(precisions, recalls_pr))
    std_auc = np.abs(np.trapezoid(tpr, fpr))
    val_loss = compute_loss(X_va, y_va, bestW1, bestW2, pos_w)
    
    return val_loss, pr_auc, std_auc, bestW1, bestW2

def find_best_hidden_feats(X_train, y_train, X_val, y_val, pos_weight=1, hidden_features=[8, 12, 16, 20, 32]):
    results = {}
    
    for H in hidden_features:
        print(f"\n--- Training Hidden Layer Size: {H} ---")
        input_dim = X_train.shape[1]
        W1 = np.random.randn(input_dim, H) * np.sqrt(2.0 / input_dim)
        W2 = np.random.randn(H + 1, 1) * np.sqrt(2.0 / (H + 1))
        
        momentum = MomentumGD(0.01, 0.9)
        bestW1, bestW2, train_loss_history, val_loss_history = train_network(
            X_train, y_train, X_val, y_val, W1, W2, momentum, pos_weight, epochs=1000, batch=512, use_early_stopping=True
        )

        _, _, Yhat_val = forward_pass(X_val, bestW1, bestW2)
        recalls_pr, precisions, fpr, tpr = calculate_pr_roc_data(Yhat_val, y_val)

        pr_auc = np.abs(np.trapezoid(precisions, recalls_pr))
        std_auc = np.abs(np.trapezoid(tpr, fpr))
        final_val_loss = compute_loss(X_val, y_val, bestW1, bestW2, pos_weight)
        
        print(f"H={H} Complete | Final Val Loss: {final_val_loss:.4f} | PR-AUC: {pr_auc:.4f} | AUC: {std_auc:.4f} ")

        results[H] = {
            'val_loss': final_val_loss,
            'AUC PR': pr_auc,
            'recalls': recalls_pr,
            'precisions': precisions,
            'train_loss_history': train_loss_history,
            'val_loss_history': val_loss_history,
            'AUC STD': std_auc,
            'true_pos_rate': tpr,
            'false_pos_rate': fpr
        }

    plt.figure(figsize=(9, 6))
    for H in hidden_features:
        plt.plot(results[H]['recalls'], results[H]['precisions'], label=f'H={H} (PR-AUC = {results[H]["AUC PR"]:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves Across Hidden Units')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.show()

    plt.figure(figsize=(9, 6))
    for H in hidden_features:
        plt.plot(results[H]['false_pos_rate'], results[H]['true_pos_rate'], label=f'H={H} (ROC-AUC = {results[H]["AUC STD"]:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random Classifier (AUC = 0.5000)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Across Hidden Units')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.show()

def run_lofo_feature_pruning(X_train_aug, y_train_vec, X_val_aug, y_val_vec, pos_weight, feature_names, H=16):
    print(f"=== Starting Leave-One-Out Feature Selection (H={H}) ===")
    base_loss, base_pr_auc, base_roc_auc, _, _ = train_and_eval_single(
        X_train_aug, y_train_vec, X_val_aug, y_val_vec, pos_weight, H=H
    )
    
    print(f"\nBASELINE -> Val Loss: {base_loss:.4f} | PR-AUC: {base_pr_auc:.4f} | ROC-AUC: {base_roc_auc:.4f}\n")
    print(f"{'Dropped Feature':<40} | {'Val Loss':<10} | {'PR-AUC':<10} | {'ROC-AUC':<10} | {'Δ ROC-AUC':<10}")
    print("-" * 90)
    
    lofo_results = []
    
    for idx, f_name in enumerate(feature_names):
        aug_col_idx = idx + 1
        X_tr_popped = np.delete(X_train_aug, aug_col_idx, axis=1)
        X_va_popped = np.delete(X_val_aug, aug_col_idx, axis=1)
        
        l_loss, l_pr_auc, l_roc_auc, _, _ = train_and_eval_single(
            X_tr_popped, y_train_vec, X_va_popped, y_val_vec, pos_weight, H=H
        )
        
        delta_roc = l_roc_auc - base_roc_auc
        lofo_results.append({
            'feature': f_name,
            'col_idx': aug_col_idx,
            'val_loss': l_loss,
            'pr_auc': l_pr_auc,
            'roc_auc': l_roc_auc,
            'delta_roc': delta_roc
        })
        
        print(f"{f_name:<40} | {l_loss:<10.4f} | {l_pr_auc:<10.4f} | {l_roc_auc:<10.4f} | {delta_roc:<+10.4f}")
        
    return lofo_results

def find_optimal_fixed_threshold(Yhat, Y, FNcost, FPcost):
    thresholds = np.linspace(0.0, 1.0, 50)
    min_cost = np.inf
    best_t = 0 
    
    for t in thresholds:
        Yhat_converted = (Yhat >= t).astype(int)
        fp = np.sum((Yhat_converted - Y) == 1)
        fn = np.sum((Yhat_converted - Y) == -1)
        cost = fp * FPcost + fn * FNcost

        if cost < min_cost:
            min_cost = cost
            best_t = t
            
    return best_t
