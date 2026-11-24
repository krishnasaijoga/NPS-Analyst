import json
from nps_insights_agent.schemas.agent_outputs import (
    ScoreAnalysisOutput,
    VoiceOfCustomerOutput,
    ChurnRiskAgentOutput
)


def compact_handoff(
        score: ScoreAnalysisOutput,
        voice: VoiceOfCustomerOutput,
        churn: ChurnRiskAgentOutput
)->dict:
    """Build a compact, decision-focused agent handoff."""
    score_data=score.model_dump(mode="json")
    voice_data=voice.model_dump(mode="json")
    churn_data=churn.model_dump(mode="json")

    return {
        "score_analysis":{
            "data_status":score_data.get("data_status"),
            "overall":score_data.get("overall"),
            "monthly_trend":score_data.get("monthly_trend",[]),
            "segment":score_data.get("segments",[])[:5],
            "key_findings":score_data.get("key_findings",[])[:5],
            "limitations":score_data.get("limitations",[])[:5]
        },
        "voice_of_customer": {
            "data_status": voice_data.get("data_status"),
            "total_comments_analyzed": voice_data.get(
                "total_comments_analyzed"
            ),
            "top_themes": [
                {
                    "theme": theme["theme"],
                    "count": theme["count"],
                    "percentage": theme["percentage"],
                }
                for theme in sorted(
                    voice_data.get("themes", []),
                    key=lambda t: t["count"],
                    reverse=True,
                )[:5]
            ],
            "key_findings": voice_data.get(
                "key_findings", []
            )[:3],
            "limitations": voice_data.get(
                "limitations", []
            )[:3],
        },

        "churn_risk": {
            "data_status": churn_data.get("data_status"),
            "currency": churn_data.get("currency"),
            "unit": churn_data.get("unit"),
            "overall": churn_data.get("overall"),
            "churn_assumptions": churn_data.get(
                "churn_assumptions"
            ),
            "risk_by_product": churn_data.get(
                "risk_by_product", []
            )[:3],
            "risk_by_category": churn_data.get(
                "risk_by_category", []
            )[:3],
            "risk_findings": churn_data.get(
                "risk_findings", []
            )[:3],
            "mismatch_summary": (
                {
                    "total_comments_analyzed": mismatch.get(
                        "total_comments_analyzed"
                    ),
                    "total_mismatches": mismatch.get(
                        "total_mismatches"
                    ),
                    "uncertain_comments": mismatch.get(
                        "uncertain_comments"
                    ),
                }
                if (
                    mismatch := churn_data.get(
                        "mismatch_analysis"
                    )
                )
                else None
            ),
        }
    }


def handoff_to_json(context:dict)->str:
    return json.dumps(
        context,
        ensure_ascii=False,
        separators=(",",":")
    )