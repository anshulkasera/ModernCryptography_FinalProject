from load_data import load_data_financial
from models import LogisticRegressionModel

df = load_data_financial()
model = LogisticRegressionModel(num_features=6)



