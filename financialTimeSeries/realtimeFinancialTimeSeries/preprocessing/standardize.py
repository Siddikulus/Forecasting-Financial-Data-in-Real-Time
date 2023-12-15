from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

def standard(df):
    X = StandardScaler().fit_transform(df.values)
    standard_df = pd.DataFrame(X, index=df.index, columns=df.columns)
    return standard_df