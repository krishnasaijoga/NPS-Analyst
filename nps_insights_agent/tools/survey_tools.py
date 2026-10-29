from typing import cast
from strands import tool

from nps_insights_agent.data_loader import load_survey_data

import numpy as np
import pandas as pd

CHURN_PROBABILITIES={
        "promoter":0.03,
        "passive":0.1,
        "detractor":0.25
    }


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
def get_revenue_at_risk()->dict:
    """
    Calculate estimated annual revenue at risk using predefined
    churn probabilities for promoters, passives, and detractors.

    Args:
        Doesn't take any inputs

    Returns:
        Dictionary containing overall financial exposure and
        breakdowns by NPS category, product, channel, and tenure.
        All monetary values are expressed in INR rupees.
    """
    df=load_survey_data().copy()
    if len(df)<=0:
        return {
            "data_status":False,
            "total_responses":0,
            "currency":"INR",
            "unit":"rupees",
        }

    df["nps_class"]=np.where(
        df["score"]>=9,
        "promoter",
        np.where(df["score"]<=6,"detractor","passive")
    )

    df["revenue_at_risk"]=df["annual_revenue_inr"]*df["nps_class"].map(CHURN_PROBABILITIES)
    total_revenue=df["annual_revenue_inr"].sum()
    total_revenue_at_risk=df["revenue_at_risk"].sum()
    total_risk_percentage=round(total_revenue_at_risk*100/total_revenue,2) if total_revenue!=0 else 0.0
    total_responses=len(df)

    df_by_product=df.groupby("product").agg(
        customers=("response_id","size"),
        annual_revenue=("annual_revenue_inr","sum"),
        revenue_at_risk=("revenue_at_risk","sum")
    )
    df_by_product["risk_percentage"]=np.where(
        df_by_product["annual_revenue"]>0,
        df_by_product["revenue_at_risk"]*100/df_by_product["annual_revenue"].replace(0,np.nan),
        0.0
    )

    df_by_channel=df.groupby("channel").agg(
        annual_revenue=("annual_revenue_inr","sum"),
        revenue_at_risk=("revenue_at_risk","sum")
    )
    df_by_channel["risk_percentage"]=np.where(
        df_by_channel["annual_revenue"]>0,
        df_by_channel["revenue_at_risk"]*100/df_by_channel["annual_revenue"].replace(0,np.nan),
        0.0
    )

    df_by_tenure=df.groupby("tenure").agg(
        annual_revenue=("annual_revenue_inr","sum"),
        revenue_at_risk=("revenue_at_risk","sum")
    )
    df_by_tenure["risk_percentage"]=np.where(
        df_by_tenure["annual_revenue"]>0,
        df_by_tenure["revenue_at_risk"]*100/df_by_tenure["annual_revenue"].replace(0,np.nan),
        0.0
    )

    df_by_nps_class=df.groupby("nps_class").agg(
        annual_revenue=("annual_revenue_inr","sum"),
        revenue_at_risk=("revenue_at_risk","sum")
    )
    df_by_nps_class["risk_percentage"]=np.where(
        df_by_nps_class["annual_revenue"]>0,
        df_by_nps_class["revenue_at_risk"]*100/df_by_nps_class["annual_revenue"].replace(0,np.nan),
        0.0
    )

    return {
        "data_status":True,
        "total_responses":total_responses,
        "currency":"INR",
        "unit":"rupees",
        "overall":{
            "annual_revenue":total_revenue,
            "revenue_at_risk":total_revenue_at_risk,
            "risk_percentage":total_risk_percentage
        },
        "churn_assumptions":CHURN_PROBABILITIES,
        "by_nps_category":[
            {
                "category":str(category),
                "annual_revenue":round(float(row["annual_revenue"]),2),
                "revenue_at_risk":round(float(row["revenue_at_risk"]),2),
                "risk_percentage":round(float(row["risk_percentage"]),2)
            } for category, row in df_by_nps_class.iterrows()
        ],
        "by_product":[
            {
                "category":str(category),
                "customers":int(row["customers"]),
                "annual_revenue":round(float(row["annual_revenue"]),2),
                "revenue_at_risk":round(float(row["revenue_at_risk"]),2),
                "risk_percentage":round(float(row["risk_percentage"]),2)
            } for category, row in df_by_product.iterrows()
        ],
        "by_channel":[
            {
                "category":str(category),
                "annual_revenue":round(float(row["annual_revenue"]),2),
                "revenue_at_risk":round(float(row["revenue_at_risk"]),2),
                "risk_percentage":round(float(row["risk_percentage"]),2)
            } for category, row in df_by_channel.iterrows()
        ],
        "by_tenure":[
            {
                "category":str(category),
                "annual_revenue":round(float(row["annual_revenue"]),2),
                "revenue_at_risk":round(float(row["revenue_at_risk"]),2),
                "risk_percentage":round(float(row["risk_percentage"]),2)
            } for category, row in df_by_tenure.iterrows()
        ],
        "data_caveats":[
            "Churn probabilities are assumed, not predicted.",
            "Revenue at risk is an expected annual loss estimate.",
            "Financial values are expressed in INR rupees."
        ]
    }


@tool
def get_callback_list(top_n:int=10)->dict:
    """
    Identify and prioritize detractor customers for follow-up.

    Customers with NPS scores between 0 and 6 are ranked by
    estimated annual revenue at risk in descending order.

    Args:
        top_n: Maximum number of callback candidates to return.
            Defaults to 10.

    Returns:
        Dictionary containing total eligible candidates,
        returned candidate count, and a prioritized callback list.
    """
    if top_n<=0:
        raise ValueError("top_n must be a positive number")
    
    df=load_survey_data().copy()

    if len(df)<=0:
        return {
            "data_status":False,
            "total_responses":0
        }

    df["nps_class"]=np.where(
        df["score"]>=9,
        "promoter",
        np.where(df["score"]<=6,"detractor","passive")
    )

    df["revenue_at_risk"]=df["annual_revenue_inr"]*df["nps_class"].map(CHURN_PROBABILITIES)

    df=df[df["nps_class"]=="detractor"].copy()

    total_candidates=len(df)

    df=df.sort_values(by=["revenue_at_risk"],ascending=[False]).head(top_n).copy()

    return {
        "data_status": True,
        "total_candidates": total_candidates,
        "returned_candidates":len(df),
        "callback_list": [
            {
                "response_id": str(row["response_id"]),
                "nps_score": int(row["score"]),
                "product": str(row["product"]),
                "annual_revenue_inr": int(row["annual_revenue_inr"]),
                "revenue_at_risk_inr": round(float(row["revenue_at_risk"]),2),
                "comment": str(row["comment"]) if pd.notna(row["comment"]) else None,
                "priority": index
            } for index, (_,row) in enumerate(df.iterrows(),start=1)
        ] 
    }


@tool
def find_score_comment_mismatches():
    pass