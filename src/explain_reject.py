import numpy as np

def get_top_3_rejection_reasons(x_vector, W1, W2, feature_names, alpha=0.01):
    x_features = x_vector[1:]
    Z1 = x_vector @ W1
    d = np.where(Z1 > 0, 1.0, alpha)
    
    W1_no_bias = W1[1:, :]
    W2_no_bias = W2[1:, :]
    
    M = (W1_no_bias * d) @ W2_no_bias
    M_vector = M.flatten()
    
    contributions = M_vector * x_features
    top_3_indices = np.argsort(contributions)[::-1][:3]
    
    top_3_reasons = []
    for idx in top_3_indices:
        top_3_reasons.append({
            'feature': feature_names[idx],
            'M_i (effective_weight)': M_vector[idx],
            'x_i (normalized_val)': x_features[idx],
            'total_contribution': contributions[idx]
        })
        
    return top_3_reasons
