from strands import tool

from nps_insights_agent.tools.theme_tools import THEMES
from nps_insights_agent.tools.survey_tools import CHURN_PROBABILITIES

from nps_insights_agent.data_loader import load_survey_data

import numpy as np
import math


@tool
def estimate_fix_impact(
    theme:str,
    resolution_rate:float=0.5
)->dict:
    """
    Estimate the hypothetical impact of resolving a customer issue.

    Affected customers are identified using the predefined theme
    keywords.

    Successfully resolved customers move up one NPS category:
    - Detractor -> Passive
    - Passive -> Promoter
    - Promoter -> Promoter

    Within each eligible category, customers with the highest
    annual revenue are selected first.

    The simulation assumes that resolving an issue improves
    the customer's NPS category. It is not a prediction.

    Args:
        theme: One of the predefined customer feedback themes.
        resolution_rate: Assumed fraction of affected customers
            whose issue is successfully resolved (0 to 1).

    Returns:
        Dictionary containing affected customers, category
        transitions, NPS impact and revenue-at-risk impact.
    """
    if not (0<=resolution_rate<=1):
        raise ValueError("resolution_rate must be between 0 and 1")
    if not theme in THEMES:
        raise ValueError(f"{theme} must be in {THEMES}")
    df=load_survey_data().copy()

    if df.empty:
        return {
            "data_status": False,
            "theme": theme,
            "resolution_rate": resolution_rate,
            "total_responses": 0,
            "affected_responses": 0,
            "nps_impact": None,
            "revenue_impact": None
        }

    total_responses = len(df)

    df["nps_class"] = np.where(
        df["score"] >= 9,
        "promoter",
        np.where(
            df["score"] <= 6,
            "detractor",
            "passive"
        )
    )

    baseline_promoters = (
        df["nps_class"] == "promoter"
    ).sum()

    baseline_passives = (
        df["nps_class"] == "passive"
    ).sum()

    baseline_detractors = (
        df["nps_class"] == "detractor"
    ).sum()

    baseline_nps = (
        (baseline_promoters - baseline_detractors)
        / total_responses
    ) * 100

    df["baseline_risk"] = (
        df["annual_revenue_inr"]
        * df["nps_class"].map(CHURN_PROBABILITIES)
    )

    baseline_revenue_at_risk = df["baseline_risk"].sum()

    keywords=THEMES[theme]

    affected_mask = (
        df["comment"]
        .fillna("")
        .astype(str)
        .str.lower()
        .apply(
            lambda comment: any(
                keyword.lower() in comment
                for keyword in keywords
            )
        )
    )

    affected_df = df.loc[affected_mask].copy()

    affected_responses = len(affected_df)

    affected_promoters = (
        affected_df["nps_class"] == "promoter"
    ).sum()

    affected_passives = (
        affected_df["nps_class"] == "passive"
    ).sum()

    affected_detractors = (
        affected_df["nps_class"] == "detractor"
    ).sum()

    detractors_recovered = math.floor(
        affected_detractors * resolution_rate
    )

    passives_promoted = math.floor(
        affected_passives * resolution_rate
    )

    recovered_detractors = (
        affected_df[
            affected_df["nps_class"] == "detractor"
        ]
        .sort_values(
            by=["annual_revenue_inr", "response_id"],
            ascending=[False, True]
        )
        .head(detractors_recovered)
    )

    promoted_passives = (
        affected_df[
            affected_df["nps_class"] == "passive"
        ]
        .sort_values(
            by=["annual_revenue_inr", "response_id"],
            ascending=[False, True]
        )
        .head(passives_promoted)
    )

    simulated_detractors = (
        baseline_detractors - detractors_recovered
    )

    simulated_passives = (
        baseline_passives
        + detractors_recovered
        - passives_promoted
    )

    simulated_promoters = (
        baseline_promoters + passives_promoted
    )

    simulated_nps = (
        (simulated_promoters - simulated_detractors)
        / total_responses
    ) * 100

    nps_improvement = simulated_nps - baseline_nps

    detractor_risk_reduction = (
        recovered_detractors["annual_revenue_inr"].sum()
        * (
            CHURN_PROBABILITIES["detractor"]
            - CHURN_PROBABILITIES["passive"]
        )
    )

    # Passive -> Promoter
    passive_risk_reduction = (
        promoted_passives["annual_revenue_inr"].sum()
        * (
            CHURN_PROBABILITIES["passive"]
            - CHURN_PROBABILITIES["promoter"]
        )
    )

    total_risk_reduction = (
        detractor_risk_reduction
        + passive_risk_reduction
    )

    simulated_revenue_at_risk = (
        baseline_revenue_at_risk
        - total_risk_reduction
    )

    return {
        "data_status": True,
        "theme": theme,
        "resolution_rate": resolution_rate,

        "total_responses": int(total_responses),
        "affected_responses": int(affected_responses),

        "affected_customers": {
            "promoters": int(affected_promoters),
            "passives": int(affected_passives),
            "detractors": int(affected_detractors)
        },

        "simulated_transitions": {
            "detractor_to_passive": int(
                detractors_recovered
            ),
            "passive_to_promoter": int(
                passives_promoted
            )
        },

        "nps_impact": {
            "baseline_nps": round(
                float(baseline_nps), 2
            ),
            "simulated_nps": round(
                float(simulated_nps), 2
            ),
            "nps_improvement_points": round(
                float(nps_improvement), 2
            )
        },

        "revenue_impact": {
            "currency": "INR",
            "unit": "rupees",
            "baseline_revenue_at_risk": round(
                float(baseline_revenue_at_risk), 2
            ),
            "simulated_revenue_at_risk": round(
                float(simulated_revenue_at_risk), 2
            ),
            "estimated_risk_reduction": round(
                float(total_risk_reduction), 2
            )
        },

        "assumptions": {
            "resolution_rate": resolution_rate,
            "recovery_rule": (
                "Detractor to Passive; "
                "Passive to Promoter"
            ),
            "selection_rule": (
                "Highest annual revenue first "
                "within each NPS category"
            ),
            "rounding_rule": (
                "Floor the number of recovered "
                "customers in each category"
            ),
            "churn_probabilities": CHURN_PROBABILITIES
        },

        "data_caveats": [
            "This is a hypothetical scenario, not a forecast.",
            "Successful resolution is assumed to improve "
            "the customer's NPS category by one level.",
            "Churn probabilities are fixed assumptions, "
            "not individually predicted probabilities.",
            "Theme detection is keyword-based.",
            "Customers are prioritized by annual revenue "
            "for the simulated transitions."
        ]
    }