
import copy

import pytest
from pydantic import ValidationError

from nps_insights_agent.schemas.agent_outputs import (
    ChurnProbabilities,
    ProposedAction,
    CallbackCandidate,
    CallbackRecommendations,
    SimulatedTransition,
    NPSImpact,
    RevenueImpact,
    AssumptionsOutput,
    FixImpactScenario,
    ActionPlannerOutput,
)


# -------------------------------------------------------
# Fixtures
# -------------------------------------------------------

@pytest.fixture
def valid_action_output():
    return {
        "data_status": True,
        "proposed_actions": [
            {
                "action": "Prioritize OTP verification improvements",
                "reason": "Recurring OTP complaints affect customer experience",
                "supporting_evidence": (
                    "OTP issues were identified in customer feedback"
                ),
            }
        ],
        "callback_recommendations": {
            "total_candidates": 4,
            "returned_candidates": 2,
            "callback_list": [
                {
                    "response_id": "R001",
                    "nps_score": 2,
                    "product": "Wealth",
                    "annual_revenue_inr": 300000,
                    "revenue_at_risk_inr": 75000,
                    "comment": "Unexpected charges",
                    "priority": 1,
                },
                {
                    "response_id": "R002",
                    "nps_score": 5,
                    "product": "Savings Account",
                    "annual_revenue_inr": 100000,
                    "revenue_at_risk_inr": 25000,
                    "comment": "OTP failure",
                    "priority": 2,
                },
            ],
        },
        "simulated_scenarios": [
            {
                "theme": "OTP & Verification Issues",
                "resolution_rate": 0.5,
                "total_responses": 600,
                "affected_responses": 40,
                "simulated_transitions": {
                    "detractor_to_passive": 10,
                    "passive_to_promoter": 5,
                },
                "nps_impact": {
                    "baseline_nps": 18.5,
                    "simulated_nps": 21.0,
                    "nps_improvement_points": 2.5,
                },
                "revenue_impact": {
                    "currency": "INR",
                    "unit": "rupees",
                    "baseline_revenue_at_risk": 1000000,
                    "simulated_revenue_at_risk": 850000,
                    "estimated_risk_reduction": 150000,
                },
                "assumptions": {
                    "resolution_rate": 0.5,
                    "recovery_rule": (
                        "Detractor to Passive; Passive to Promoter"
                    ),
                    "selection_rule": "Highest annual revenue first",
                    "rounding_rule": "Floor recovered customer counts",
                    "churn_probabilities": {
                        "promoter": 0.03,
                        "passive": 0.10,
                        "detractor": 0.25,
                    },
                },
            }
        ],
        "caveats": [
            "Scenario outcomes are hypothetical."
        ],
        "limitations": [
            "Theme detection is keyword-based."
        ],
    }


@pytest.fixture
def minimal_action_output():
    return {
        "data_status": False,
    }


# -------------------------------------------------------
# Valid output and nested classes
# -------------------------------------------------------

def test_valid_action_output(valid_action_output):
    output = ActionPlannerOutput.model_validate(valid_action_output)

    assert output.data_status is True
    assert len(output.proposed_actions) == 1
    assert output.callback_recommendations is not None
    assert len(output.simulated_scenarios) == 1


def test_nested_models(valid_action_output):
    output = ActionPlannerOutput.model_validate(valid_action_output)

    assert isinstance(output.proposed_actions[0], ProposedAction)
    assert isinstance(
        output.callback_recommendations,
        CallbackRecommendations
    )
    assert isinstance(
        output.callback_recommendations.callback_list[0],
        CallbackCandidate
    )
    assert isinstance(
        output.simulated_scenarios[0],
        FixImpactScenario
    )
    assert isinstance(
        output.simulated_scenarios[0].simulated_transitions,
        SimulatedTransition
    )
    assert isinstance(
        output.simulated_scenarios[0].nps_impact,
        NPSImpact
    )
    assert isinstance(
        output.simulated_scenarios[0].revenue_impact,
        RevenueImpact
    )
    assert isinstance(
        output.simulated_scenarios[0].assumptions,
        AssumptionsOutput
    )
    assert isinstance(
        output.simulated_scenarios[0].assumptions.churn_probabilities,
        ChurnProbabilities
    )


# -------------------------------------------------------
# Callback validations
# -------------------------------------------------------

@pytest.mark.parametrize("score", [-1, 7, 10])
def test_invalid_callback_nps_score(score):
    with pytest.raises(ValidationError):
        CallbackCandidate(
            response_id="R001",
            nps_score=score,
            product="Wealth",
            annual_revenue_inr=100000,
            revenue_at_risk_inr=25000,
            priority=1,
        )


@pytest.mark.parametrize("priority", [0, -1])
def test_invalid_callback_priority(priority):
    with pytest.raises(ValidationError):
        CallbackCandidate(
            response_id="R001",
            nps_score=2,
            product="Wealth",
            annual_revenue_inr=100000,
            revenue_at_risk_inr=25000,
            priority=priority,
        )


def test_callback_count_cannot_exceed_total():
    with pytest.raises(ValidationError):
        CallbackRecommendations(
            total_candidates=2,
            returned_candidates=3,
            callback_list=[],
        )


def test_callback_count_matches_list():
    with pytest.raises(ValidationError):
        CallbackRecommendations(
            total_candidates=5,
            returned_candidates=2,
            callback_list=[],
        )


# -------------------------------------------------------
# Resolution rate and probabilities
# -------------------------------------------------------

@pytest.mark.parametrize("rate", [-0.1, 1.1])
def test_invalid_resolution_rate(rate):
    with pytest.raises(ValidationError):
        AssumptionsOutput(
            resolution_rate=rate,
            recovery_rule="Detractor to Passive",
            selection_rule="Highest revenue",
            rounding_rule="Floor",
            churn_probabilities=ChurnProbabilities(),
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("promoter", -0.1),
        ("passive", 1.1),
        ("detractor", 1.5),
    ],
)
def test_invalid_churn_probabilities(field, value):
    data = {
        "promoter": 0.03,
        "passive": 0.10,
        "detractor": 0.25,
    }
    data[field] = value

    with pytest.raises(ValidationError):
        ChurnProbabilities.model_validate(data)


# -------------------------------------------------------
# NPS and financial impact consistency
# -------------------------------------------------------

def test_invalid_nps_improvement():
    with pytest.raises(ValidationError):
        NPSImpact(
            baseline_nps=18.5,
            simulated_nps=21.0,
            nps_improvement_points=10.0,
        )


def test_invalid_revenue_reduction():
    with pytest.raises(ValidationError):
        RevenueImpact(
            baseline_revenue_at_risk=1000000,
            simulated_revenue_at_risk=850000,
            estimated_risk_reduction=500000,
        )


def test_scenario_resolution_rate_consistency(valid_action_output):
    data = copy.deepcopy(valid_action_output)

    data["simulated_scenarios"][0]["assumptions"][
        "resolution_rate"
    ] = 0.8

    with pytest.raises(ValidationError):
        ActionPlannerOutput.model_validate(data)


# -------------------------------------------------------
# Defaults and optional fields
# -------------------------------------------------------

def test_minimal_output(minimal_action_output):
    output = ActionPlannerOutput.model_validate(minimal_action_output)

    assert output.data_status is False
    assert output.proposed_actions == []
    assert output.callback_recommendations is None
    assert output.simulated_scenarios == []
    assert output.caveats == []
    assert output.limitations == []


def test_default_lists_are_independent():
    first = ActionPlannerOutput(data_status=False)
    second = ActionPlannerOutput(data_status=False)

    first.caveats.append("First scenario caveat")

    assert second.caveats == []


# -------------------------------------------------------
# Required fields and serialization
# -------------------------------------------------------

def test_missing_data_status(valid_action_output):
    data = copy.deepcopy(valid_action_output)
    del data["data_status"]

    with pytest.raises(ValidationError):
        ActionPlannerOutput.model_validate(data)


def test_json_serialization_roundtrip(valid_action_output):
    output = ActionPlannerOutput.model_validate(valid_action_output)

    json_string = output.model_dump_json()
    restored = ActionPlannerOutput.model_validate_json(json_string)

    assert restored == output
