from models import LogisticRegressionModel, CreditRiskMLP
from load_data import load_data_financial
from train import train_model, evaluate_model
import torch

X_train, X_test, y_train, y_test, scaler = load_data_financial()

logistic_model = LogisticRegressionModel(X_train.shape[1])
train_model(logistic_model, X_train, y_train)
evaluate_model(logistic_model, X_test, y_test)

mlp_model = CreditRiskMLP(X_train.shape[1])
train_model(mlp_model, X_train, y_train)
evaluate_model(mlp_model, X_test, y_test)

torch.save(logistic_model.state_dict(), "logistic_model.pth")
torch.save(mlp_model.state_dict(), "credit_risk_mlp.pth")
