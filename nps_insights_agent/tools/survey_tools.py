from strands import tool

from nps_insights_agent.data_loader import load_survey_data

import numpy as np
import pandas as pd


@tool
def get_nps_overview()->dict:
    """
    Retrieve nps details from dataset

    Args:
        Doesn't take any inputs

    Returns:
        Dictionary containing nps details: total_responses, overall_nps, promoters count, passives count, detractors count, monthly nps, margin of error
    """
    df=load_survey_data().copy()
    
    promoters=len(df[df["score"]>=9])
    passives=len(df[df["score"].between(7,8)])
    detractors=len(df[df["score"]<7])
    total_responses=len(df)

    monthly_data=df.groupby("month").agg(
        total_count=("score","size"),
        promoters=("score",lambda x: (x>=9).sum()),
        detractors=("score", lambda x: (x<=6).sum())
    )

    monthly_data["nps_score"]=(
        (monthly_data["promoters"]-monthly_data["detractors"])*100/monthly_data["total_count"]
    )

    nps_values=np.where(
        df["score"]>=9,
        1,
        np.where(df["score"]<=6,-1,0)
    )

    n=len(nps_values)

    moe_95=(1.96*np.std(nps_values,ddof=1)*100/np.sqrt(n)) if n>1 else None

    overall_nps=(promoters-detractors)*100/total_responses if total_responses!=0 else 0.0

    return {
        "data_status":total_responses>0,
        "total_responses":total_responses,
        "overall_nps":round(overall_nps,2),
        "promoters":promoters,
        "passives":passives,
        "detractors":detractors,
        "promoter_percentage":round(promoters*100/total_responses if total_responses!=0 else 0.0,2),
        "passive_percentage":round(passives*100/total_responses if total_responses!=0 else 0.0,2),
        "detractor_percentage":round(detractors*100/total_responses if total_responses!=0 else 0.0,2),
        "margin_of_error_95":round(float(moe_95),2) if moe_95 is not None else None,
        "monthly_trend":[
            {
                "month":month,
                "total_responses":int(row["total_count"]),
                "nps":round(float(row["nps_score"]),2)
            } for month, row in monthly_data.iterrows()
        ]
    }


def calculate_moe(scores):
    nps_values=np.where(
        scores>=9,
        1,
        np.where(scores<=6,-1,0)
    )

    n=len(nps_values)

    if n<=1:
        return None

    moe=1.96*np.std(nps_values,ddof=1)*100/np.sqrt(n)

    return round(float(moe),2)

@tool
def get_nps_by_segment(segment:str,month:str|None=None)->dict:
    """
    Calculate NPS statistics for each group in a segmentation dimension.

    Args:
        segment: Segmentation dimension: product, channel, or tenure.
        month: Optional month in YYYY-MM format. Defaults to all months.

    Returns:
        Dictionary containing segment-level NPS, classification counts,
        sample size, 95% margin of error, and low-sample indicator.
    """
    SEGMENTS=[
        "product",
        "channel",
        "tenure"
    ]
    df=load_survey_data().copy()
    if month is not None:
        df=df[df["month"]==month]
    if len(df)<1:
        raise ValueError("Given month doesn't exist in the dataset")
    segment=segment.strip().lower()
    if segment in SEGMENTS:
        segment_df = df.groupby(segment).agg(
            total_count=("score", "size"),
            promoters=("score", lambda x: (x >= 9).sum()),
            passives=("score", lambda x: x.between(7, 8).sum()),
            detractors=("score", lambda x: (x <= 6).sum())
        )
    else:
        raise ValueError(f"Segment can be one among {SEGMENTS}")

    segment_df["nps_score"]=(
        (segment_df["promoters"]-segment_df["detractors"])*100/segment_df["total_count"]
    )

    segment_df["moe"]=df.groupby(segment)["score"].apply(calculate_moe)
    
    return {
        "data_status":len(segment_df)>0,
        "segment_dimension":segment,
        "month":month,
        "segments":[
            {
                "segment_name":str(segment_key),
                "total_responses":int(row["total_count"]),
                "promoters":int(row["promoters"]),
                "passives":int(row["passives"]),
                "detractors":int(row["detractors"]),
                "nps":round(float(row["nps_score"]),2),
                "moe":float(row["moe"]) if not pd.isna(row["moe"]) else None,
                "low_sample":bool(row["total_count"]<30)
            } for segment_key, row in segment_df.iterrows()
        ]
    }


@tool
def get_revenue_at_risk():
    pass


@tool
def get_callback_list():
    pass


@tool
def find_score_comment_mismatches():
    pass