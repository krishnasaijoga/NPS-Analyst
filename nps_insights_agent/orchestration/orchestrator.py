from dataclasses import dataclass
from typing import Any

from nps_insights_agent.schemas.agent_outputs import (
    ScoreAnalysisOutput,
    VoiceOfCustomerOutput,
    ChurnRiskAgentOutput,
    ActionPlannerOutput
)
from nps_insights_agent.orchestration.handoff import compact_handoff,handoff_to_json


@dataclass
class OrchestrationResult:
    score_analysis:ScoreAnalysisOutput
    voice_of_customer:VoiceOfCustomerOutput
    churn_risk:ChurnRiskAgentOutput
    action_plan:ActionPlannerOutput


def invoke_structured(
    agent:Any,
    prompt:str,
    expected_model:type
):
    """Execute an agent and verify its output contract."""
    result=agent(prompt)
    output=result.structured_output

    if not isinstance(output, expected_model):
        raise TypeError(
            f"Expected {expected_model.__name__}, received {type(output).__name__}"
        )
    return output


def run_nps_workflow(
        score_agent:Any,
        voice_agent:Any,
        churn_agent:Any,
        action_agent:Any
)->OrchestrationResult:
    score=invoke_structured(
        score_agent,
        (
            "Analyze overall NPS, monthly trends, "
            "and product-level performance using your tools. "
            "Return evidence-backed findings."
        ),
        ScoreAnalysisOutput
    )

    voice=invoke_structured(
        voice_agent,
        (
            "Analyze customer feedback across the dataset. "
            "Identify recurring themes and supporting evidence "
            "using your tools."
        ),
        VoiceOfCustomerOutput
    )

    churn=invoke_structured(
        churn_agent,
        (
            "Analyze financial revenue exposure and "
            "score-comment mismatches using your tools. "
            "Return concise, structured findings."
        ),
        ChurnRiskAgentOutput
    )

    context = compact_handoff(
        score=score,
        voice=voice,
        churn=churn,
    )

    import json

    
    action = invoke_structured(
        action_agent,
        (
            "Prepare a concise customer improvement action plan.\n\n"
            "SPECIALIST FINDINGS:\n"
            f"{handoff_to_json(context)}\n\n"

            "EXECUTION REQUIREMENTS:\n"
            "1. Call get_callback_list with top_n=3.\n"
            "2. Call estimate_fix_impact exactly once with:\n"
            "   theme='OTP & Verification Issues'\n"
            "   resolution_rate=0.5.\n"
            "3. Return the required ActionPlannerOutput.\n\n"

            "OUTPUT LIMITS:\n"
            "- Maximum 3 proposed actions.\n"
            "- Maximum 3 callback records.\n"
            "- Exactly 1 simulated scenario.\n"
            "- Maximum 2 caveats and 2 limitations.\n"
            "- Keep each action, reason, and evidence concise.\n"
            "- Do not repeat the specialist analysis.\n"
            "- Preserve exact numerical values from tools.\n"
            "- Treat simulations as hypothetical, not forecasts.\n"
        ),
        ActionPlannerOutput,
    )

    return OrchestrationResult(
        score_analysis=score,
        voice_of_customer=voice,
        churn_risk=churn,
        action_plan=action
    )