import pandas as pd
from functools import lru_cache


@lru_cache(maxsize=1)
def load_survey_data():
    df=pd.read_csv("data/nps_survey.csv")
    df["score"]=pd.to_numeric(df["score"],errors="coerce")
    df=df[df["score"].between(0,10) & (df["score"]%1==0)].copy()
    return df