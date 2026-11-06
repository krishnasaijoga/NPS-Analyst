import pandas as pd
from functools import lru_cache

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "nps_survey.csv"


@lru_cache(maxsize=1)
def load_survey_data():
    df=pd.read_csv(DATA_PATH)
    df["score"]=pd.to_numeric(df["score"],errors="coerce")
    df=df[df["score"].between(0,10) & (df["score"]%1==0)].copy()
    return df