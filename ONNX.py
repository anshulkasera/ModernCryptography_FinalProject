import json
import torch
from models import LogisticRegressionModel, CreditRiskMLP
from load_data import load_data_financial

X_train, X_test, y_train, y_test, scaler = load_data_financial()
num_features = X_train.shape[1]

# Pick which model to prove
model = CreditRiskMLP(num_features)
model.load_state_dict(torch.load("credit_risk_mlp.pth"))
model.eval()

sample = X_test[0:1]  # shape (1, 6), already scaled

torch.onnx.export(
    model,
    sample,
    "network.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=14,
)

with open("input.json", "w") as f:
    json.dump({"input_data": [sample.flatten().tolist()]}, f)