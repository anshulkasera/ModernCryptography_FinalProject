import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data_financial():
    df = pd.read_excel(
        "data/default_of_credit_card_clients.xls",
        header=1,
        engine="xlrd"
    )

    features = [
        "LIMIT_BAL",
        "AGE",
        "PAY_0",
        "PAY_2",
        "BILL_AMT1",
        "PAY_AMT1",
    ]

    target = "default payment next month"

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Fit scaling only on training data to prevent test-data leakage.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)

    # BCEWithLogitsLoss expects labels shaped like (number_of_rows, 1).
    y_train = torch.tensor(y_train.to_numpy(), dtype=torch.float32).reshape(-1, 1)
    y_test = torch.tensor(y_test.to_numpy(), dtype=torch.float32).reshape(-1, 1)

    return X_train, X_test, y_train, y_test, scaler