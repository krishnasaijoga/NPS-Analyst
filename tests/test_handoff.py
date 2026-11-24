
import json

from nps_insights_agent.orchestration.handoff import (
    compact_handoff,
    handoff_to_json,
)
from nps_insights_agent.schemas.agent_outputs import (
    ScoreAnalysisOutput,
    VoiceOfCustomerOutput,
    ChurnRiskAgentOutput,
)


def make_outputs():
    score = ScoreAnalysisOutput.model_construct(
        data_status=True,
        overall=None,
        monthly_trend=[],
        segments=[],
        key_findings=[],
        limitations=[],
    )

    voice = VoiceOfCustomerOutput.model_construct(
        data_status=True,
        total_comments_analyzed=100,
        themes=[],
        key_findings=[],
        limitations=[],
    )

    churn = ChurnRiskAgentOutput.model_construct(
        data_status=True,
        currency="INR",
        unit="rupees",
        overall=None,
        risk_by_product=[],
        risk_by_category=[],
        risk_findings=[],
        mismatch_analysis=None,
    )

    return score, voice, churn


def test_handoff_has_three_specialists():
    context = compact_handoff(*make_outputs())

    assert set(context) == {
        "score_analysis",
        "voice_of_customer",
        "churn_risk",
    }


def test_handoff_is_json_serializable():
    context = compact_handoff(*make_outputs())

    encoded = handoff_to_json(context)
    decoded = json.loads(encoded)

    assert decoded == context


def test_handoff_preserves_data_status():
    context = compact_handoff(*make_outputs())

    assert context["score_analysis"]["data_status"] is True
    assert context["voice_of_customer"]["data_status"] is True
    assert context["churn_risk"]["data_status"] is True


def test_handoff_preserves_currency():
    context = compact_handoff(*make_outputs())

    assert context["churn_risk"]["currency"] == "INR"
    assert context["churn_risk"]["unit"] == "rupees"


def test_handoff_excludes_raw_mismatch_records():
    context = compact_handoff(*make_outputs())

    assert "mismatches" not in context["churn_risk"]
    assert (
        "mismatch_analysis"
        not in context["churn_risk"]
    )
