
import pandas as pd

def my_metric(prices, **kwargs):
    return pd.DataFrame({'ticker': ['A'], 'my_metric': [1.23]})

METRICS = {'my_metric': my_metric}
