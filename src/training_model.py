import numpy as np

def leaky_relu(Z, alpha=0.01):
    return np.where(Z > 0, Z, Z * alpha)

def leaky_relu_prime(Z, alpha=0.01):
    return np.where(Z > 0, 1.0, alpha)

def sigmoid(X):
    return 1 / (1 + np.exp(-X))

def forward_pass(X_batch, W1, W2):
    Z1 = X_batch @ W1
    A1 = leaky_relu(Z1)
    
    ones = np.ones((len(A1), 1))
    A1_aug = np.hstack([ones, A1])
    
    Z2 = A1_aug @ W2
    Yhat = sigmoid(Z2)
    
    return Z1, A1_aug, Yhat

def compute_loss(X, Y, W1, W2, pos_weight=1.0):
    _, _, Yhat = forward_pass(X, W1, W2)
    eps = 1e-15
    Yhat_clipped = np.clip(Yhat, eps, 1 - eps)
    
    loss_vector = -(pos_weight * Y * np.log(Yhat_clipped) + (1 - Y) * np.log(1 - Yhat_clipped))
    return np.mean(loss_vector)

def compute_gradients(X_batch, Y_batch, W1, W2, Z1, A1_aug, Yhat, pos_weight=1.0):
    m = len(X_batch)
    dZ2 = Yhat * (1 + Y_batch * (pos_weight - 1)) - Y_batch * pos_weight
    
    W2_no_bias = W2[1:, :] 
    dZ1 = (dZ2 @ np.transpose(W2_no_bias)) * leaky_relu_prime(Z1)
    
    nablaW1 = (1 / m) * (np.transpose(X_batch) @ dZ1)
    nablaW2 = (1 / m) * (np.transpose(A1_aug) @ dZ2)
    return nablaW1, nablaW2

def train_network(X, Y, X_val, Y_val, W1, W2, optimizer, pos_weight, epochs=1000, batch=64, use_early_stopping=True):
    m = len(X)
    train_loss_history = []
    val_loss_history = []

    min_delta = 1e-4
    patience = 20
    best_val_loss = float('inf')
    patience_counter = 0

    bestW1 = W1.copy()
    bestW2 = W2.copy()
    
    for epoch in range(1, epochs + 1):
        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        Y_shuffled = Y[indices]

        for start_idx in range(0, m, batch):
            end_idx = start_idx + batch
            X_batch = X_shuffled[start_idx:end_idx]
            Y_batch = Y_shuffled[start_idx:end_idx]

            Z1, A1_aug, Yhat = forward_pass(X_batch, W1, W2)
            nablaW1, nablaW2 = compute_gradients(X_batch, Y_batch, W1, W2, Z1, A1_aug, Yhat, pos_weight)
            W1, W2 = optimizer.update_weights(W1, W2, nablaW1, nablaW2)

        val_loss = compute_loss(X_val, Y_val, W1, W2, pos_weight)
        val_loss_history.append(val_loss)
        train_loss_history.append(compute_loss(X, Y, W1, W2, pos_weight))

        if use_early_stopping:
            if val_loss < (best_val_loss - min_delta):
                best_val_loss = val_loss
                patience_counter = 0
                bestW1 = W1.copy()
                bestW2 = W2.copy()
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"Stopped early at epoch {epoch}")
                break
        else:
            bestW1, bestW2 = W1, W2

    return bestW1, bestW2, train_loss_history, val_loss_history
