from src.data_processing import load_and_preprocess_data
from src.evaluation_metrics import find_best_hidden_feats

def main():
    print("Load and process data...")
    X_train_aug, y_train_vec, X_val_aug, y_val_vec, pos_weight, feature_names = load_and_preprocess_data(
        "cs-training.csv"
    )
    
    print(f"Train shape: {X_train_aug.shape}, Val shape: {X_val_aug.shape}")
    print(f"Positive class weight: {pos_weight:.2f}\n")

    print("Compare training for different hidden layer architectures...")
    find_best_hidden_feats(
        X_train_aug, 
        y_train_vec, 
        X_val_aug, 
        y_val_vec, 
        pos_weight, 
        hidden_features=[8, 16, 32]
    )
if __name__ == "__main__":
    main()
