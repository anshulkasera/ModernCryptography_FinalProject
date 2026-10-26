import pandas as pd
import numpy as np

def load_data_financial():
    df = pd.read_excel("data/default_of_credit_card_clients.xls")
    return df
