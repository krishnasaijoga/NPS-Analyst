from strands import tool

from nps_insights_agent.data_loader import load_survey_data

import numpy as np


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


@tool
def get_nps_by_segment():
    pass


@tool
def get_revenue_at_risk():
    pass


@tool
def get_callback_list():
    pass


@tool
def find_score_comment_mismatches():
    pass