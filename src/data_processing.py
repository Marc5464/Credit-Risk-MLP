import pandas as pd
import numpy as np

def load_and_preprocess_data("cs-training.csv", train_ratio=0.75):
    df = pd.read_csv(filepath, sep=';', index_col=0)

    df['NumberOfDependents'] = df['NumberOfDependents'].fillna(df['NumberOfDependents'].mode()[0])
    df['MonthlyIncome'] = df['MonthlyIncome'].fillna(df['MonthlyIncome'].median())

    delinquency_cols = [
        'NumberOfTime30-59DaysPastDueNotWorse',
        'NumberOfTime60-89DaysPastDueNotWorse',
        'NumberOfTimes90DaysLate'
    ]

    for col in delinquency_cols:
        df[f'{col}_has_delinquency'] = (df[col] > 0).astype(float)

    df['serious_delinquent'] = (
        (df['NumberOfTime30-59DaysPastDueNotWorse'] > 6) |
        (df['NumberOfTime60-89DaysPastDueNotWorse'] > 6) |
        (df['NumberOfTimes90DaysLate'] > 6)
    ).astype(float)

    df['TotalDelinquencyCount'] = (
        df['NumberOfTime30-59DaysPastDueNotWorse'] +
        df['NumberOfTime60-89DaysPastDueNotWorse'] +
        df['NumberOfTimes90DaysLate']
    )
    df['IncomePerPerson'] = df['MonthlyIncome'] / (df['NumberOfDependents'] + 1)
    df['EstimatedMonthlyDebt'] = df['DebtRatio'] * df['MonthlyIncome']
    df['UtilizationBurden'] = df['RevolvingUtilizationOfUnsecuredLines'] * df['NumberOfOpenCreditLinesAndLoans']

    working_years = np.maximum(df['age'] - 18, 1)
    df['LinesPerWorkingYear'] = df['NumberOfOpenCreditLinesAndLoans'] / working_years
    df['RealEstatePerWorkingYear'] = df['NumberRealEstateLoansOrLines'] / working_years

    log_util_temp = np.log1p(np.clip(df['RevolvingUtilizationOfUnsecuredLines'], 0, 10))
    df['Util_x_SeriousDelinquent'] = log_util_temp * df['serious_delinquent']
    df['Debt_x_TotalDelinquency'] = np.log1p(df['EstimatedMonthlyDebt']) * df['TotalDelinquencyCount']

    q_cols = ['RevolvingUtilizationOfUnsecuredLines', 'DebtRatio', 'MonthlyIncome']
    for col in q_cols:
        df[f'{col}_decile'] = pd.qcut(df[col], q=10, labels=False, duplicates='drop').astype(float)

    feature_names = df.drop(columns=['SeriousDlqin2yrs']).columns.tolist()

    y = df['SeriousDlqin2yrs'].to_numpy()
    X = df.drop(columns=['SeriousDlqin2yrs']).to_numpy()

    split_row = int(len(X) * train_ratio)
    X_train, X_val = X[:split_row].copy(), X[split_row:].copy()
    y_train, y_val = y[:split_row], y[split_row:]

    delinquency_indices = [feature_names.index(col) for col in delinquency_cols]
    for idx in delinquency_indices:
        X_train[:, idx] = np.clip(X_train[:, idx], 0, 6)
        X_val[:, idx]   = np.clip(X_val[:, idx], 0, 6)

    target_cols = [
        'RevolvingUtilizationOfUnsecuredLines',
        'DebtRatio',
        'MonthlyIncome',
        'IncomePerPerson',
        'EstimatedMonthlyDebt',
        'UtilizationBurden'
    ]
    target_indices = [feature_names.index(col) for col in target_cols]

    for idx in target_indices:
        cap_val = np.percentile(X_train[:, idx], 99)
        X_train[:, idx] = np.log1p(np.clip(X_train[:, idx], 0, cap_val))
        X_val[:, idx]   = np.log1p(np.clip(X_val[:, idx], 0, cap_val))

    binary_flag_cols = [f'{col}_has_delinquency' for col in delinquency_cols] + ['serious_delinquent']
    binary_indices = [feature_names.index(col) for col in binary_flag_cols]
    continuous_indices = [i for i in range(X_train.shape[1]) if i not in binary_indices]

    mean_cont = np.mean(X_train[:, continuous_indices], axis=0)
    std_cont = np.std(X_train[:, continuous_indices], axis=0)
    std_cont[std_cont == 0] = 1e-8

    X_train_cont_norm = (X_train[:, continuous_indices] - mean_cont) / std_cont
    X_val_cont_norm = (X_val[:, continuous_indices] - mean_cont) / std_cont

    X_train_aug = np.hstack((
        np.ones((X_train.shape[0], 1)),
        X_train_cont_norm,
        X_train[:, binary_indices]
    ))

    X_val_aug = np.hstack((
        np.ones((X_val.shape[0], 1)),
        X_val_cont_norm,
        X_val[:, binary_indices]
    ))

    y_train_vec = y_train.reshape(-1, 1)
    y_val_vec = y_val.reshape(-1, 1)

    num_negatives = np.sum(y_train_vec == 0)
    num_positives = np.sum(y_train_vec == 1)
    pos_weight = num_negatives / num_positives

    return X_train_aug, y_train_vec, X_val_aug, y_val_vec, pos_weight, feature_names
