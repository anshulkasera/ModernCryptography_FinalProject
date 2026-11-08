import json
import torch
from models import LogisticRegressionModel, CreditRiskMLP
from load_data import load_data_financial
import sys

X_train, X_test, y_train, y_test, scaler = load_data_financial()
num_features = X_train.shape[1]

# Pick which model to prove
try:
    if sys.argv[1] == "credit":
        model = CreditRiskMLP(num_features)
        model.load_state_dict(torch.load("credit_risk_mlp.pth"))
        model.eval()
    elif sys.argv[1] == "logistic":
        model = LogisticRegressionModel(num_features)
        model.load_state_dict(torch.load("logistic_model.pth"))
        model.eval()
except:
    print("Please enter correct parameters")
    sys.exit()


sample = X_test[0:1] 

torch.onnx.export(model,sample,"network.onnx",input_names=["input"],output_names=["output"],opset_version=14,)

with open("input.json", "w") as f:
    json.dump({"input_data": [sample.flatten().tolist()]}, f)
