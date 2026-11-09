
import copy

import pytest
from pydantic import ValidationError

from nps_insights_agent.schemas.agent_outputs import (
    RevenueRiskMetrics,
    CategoryRiskMetrics,
    ProductRiskMetrics,
    ChannelRiskMetrics,
    TenureRiskMetrics,
    RiskFindings,
    ScoreCommentMismatch,
    MismatchAnalysis,
    ChurnAssumptions,
    ChurnRiskAgentOutput,
)


# -----------------------------------------------------
# Fixtures
# -----------------------------------------------------

@pytest.fixture
def valid_risk_output():
    return {
        "data_status": True,
        "currency": "INR",
        "unit": "rupees",
        "churn_assumptions": {
            "promoter": 0.03,
            "passive": 0.10,
            "detractor": 0.25,
        },
        "overall": {
            "customers": 6,
            "annual_revenue": 210000,
            "revenue_at_risk": 31000,
            "risk_percentage": 14.76,
        },
        "risk_by_category": [
            {
                "nps_category": "promoter",
                "metrics": {
                    "customers": 2,
                    "annual_revenue": 50000,
                    "revenue_at_risk": 1500,
                    "risk_percentage": 3.0,
                },
            },
            {
                "nps_category": "passive",
                "metrics": {
                    "customers": 2,
                    "annual_revenue": 70000,
                    "revenue_at_risk": 7000,
                    "risk_percentage": 10.0,
                },
            },
            {
                "nps_category": "detractor",
                "metrics": {
                    "customers": 2,
                    "annual_revenue": 90000,
                    "revenue_at_risk": 22500,
                    "risk_percentage": 25.0,
                },
            },
        ],
        "risk_by_product": [
            {
                "product": "Savings Account",
                "metrics": {
                    "customers": 3,
                    "annual_revenue": 60000,
                    "revenue_at_risk": 9800,
                    "risk_percentage": 16.33,
                },
            },
            {
                "product": "Credit Card",
                "metrics": {
                    "customers": 2,
                    "annual_revenue": 90000,
                    "revenue_at_risk": 6200,
                    "risk_percentage": 6.89,
                },
            },
            {
                "product": "Wealth",
                "metrics": {
                    "customers": 1,
                    "annual_revenue": 60000,
                    "revenue_at_risk": 15000,
                    "risk_percentage": 25.0,
                },
            },
        ],
        "risk_by_channel": [
            {
                "channel": "Mobile App",
                "metrics": {
                    "customers": 3,
                    "annual_revenue": 100000,
                    "revenue_at_risk": 22800,
                    "risk_percentage": 22.8,
                },
            },
            {
                "channel": "Branch",
                "metrics": {
                    "customers": 2,
                    "annual_revenue": 60000,
                    "revenue_at_risk": 3200,
                    "risk_percentage": 5.33,
                },
            },
            {
                "channel": "Call Centre",
                "metrics": {
                    "customers": 1,
                    "annual_revenue": 50000,
                    "revenue_at_risk": 5000,
                    "risk_percentage": 10.0,
                },
            },
        ],
        "risk_by_tenure": [
            {
                "tenure": "<1 yr",
                "metrics": {
                    "customers": 2,
                    "annual_revenue": 70000,
                    "revenue_at_risk": 8700,
                    "risk_percentage": 12.43,
                },
            },
            {
                "tenure": "1-3 yrs",
                "metrics": {
                    "customers": 2,
                    "annual_revenue": 70000,
                    "revenue_at_risk": 7000,
                    "risk_percentage": 10.0,
                },
            },
            {
                "tenure": "3+ yrs",
                "metrics": {
                    "customers": 2,
                    "annual_revenue": 70000,
                    "revenue_at_risk": 15300,
                    "risk_percentage": 21.86,
                },
            },
        ],
        "risk_findings": [
            {
                "findings": "Wealth customers have high financial exposure.",
                "evidence": "Wealth revenue at risk is INR 15,000.",
                "implications": "Review high-value detractor experiences.",
            },
        ],
        "mismatch_analysis": {
            "total_comments_analyzed": 6,
            "total_mismatches": 1,
            "uncertain_comments": 1,
            "mismatches": [
                {
                    "response_id": "R003",
                    "score": 2,
                    "comment": "Excellent banking experience!",
                    "sentiment": "positive",
                    "sentiment_reason": "Expresses satisfaction.",
                    "mismatch_reason": (
                        "Positive comment with detractor NPS score"
                    ),
                },
            ],
        },
        "data_caveats": [
            "Churn probabilities are assumptions, not predictions."
        ],
    }


# -----------------------------------------------------
# Valid output and nested classes
# -----------------------------------------------------

def test_valid_churn_risk_output(valid_risk_output):
    output = ChurnRiskAgentOutput.model_validate(valid_risk_output)

    assert output.data_status is True
    assert output.currency == "INR"
    assert output.unit == "rupees"
    assert output.overall is not None
    assert output.overall.revenue_at_risk == 31000
    assert len(output.risk_by_category) == 3
    assert len(output.risk_by_product) == 3
    assert len(output.risk_findings) == 1


def test_nested_model_types(valid_risk_output):
    output = ChurnRiskAgentOutput.model_validate(valid_risk_output)

    assert isinstance(output.overall, RevenueRiskMetrics)
    assert isinstance(output.risk_by_category[0], CategoryRiskMetrics)
    assert isinstance(output.risk_by_product[0], ProductRiskMetrics)
    assert isinstance(output.risk_by_channel[0], ChannelRiskMetrics)
    assert isinstance(output.risk_by_tenure[0], TenureRiskMetrics)
    assert isinstance(output.risk_findings[0], RiskFindings)
    assert isinstance(output.churn_assumptions, ChurnAssumptions)
    assert isinstance(output.mismatch_analysis, MismatchAnalysis)
    assert isinstance(
        output.mismatch_analysis.mismatches[0],
        ScoreCommentMismatch
    )


# -----------------------------------------------------
# Field validations
# -----------------------------------------------------

@pytest.mark.parametrize(
    "field,value",
    [
        ("customers", -1),
        ("annual_revenue", -100),
        ("revenue_at_risk", -500),
        ("risk_percentage", -0.1),
        ("risk_percentage", 100.1),
    ],
)
def test_invalid_financial_metrics(field, value):
    data = {
        "customers": 10,
        "annual_revenue": 100000,
        "revenue_at_risk": 25000,
        "risk_percentage": 25.0,
    }
    data[field] = value

    with pytest.raises(ValidationError):
        RevenueRiskMetrics.model_validate(data)


@pytest.mark.parametrize(
    "field,value",
    [
        ("promoter", -0.01),
        ("promoter", 1.01),
        ("passive", -0.1),
        ("passive", 1.2),
        ("detractor", -0.2),
        ("detractor", 1.1),
    ],
)
def test_invalid_churn_assumptions(field, value):
    data = {
        "promoter": 0.03,
        "passive": 0.10,
        "detractor": 0.25,
    }
    data[field] = value

    with pytest.raises(ValidationError):
        ChurnAssumptions.model_validate(data)


@pytest.mark.parametrize(
    "field,value",
    [
        ("currency", "USD"),
        ("currency", "EUR"),
        ("unit", "lakhs"),
        ("unit", "millions"),
    ],
)
def test_invalid_currency_and_units(valid_risk_output, field, value):
    data = copy.deepcopy(valid_risk_output)
    data[field] = value

    with pytest.raises(ValidationError):
        ChurnRiskAgentOutput.model_validate(data)


@pytest.mark.parametrize(
    "field,value",
    [
        ("product", "Current Account"),
        ("channel", "Online Banking"),
        ("tenure", "5+ yrs"),
        ("nps_category", "neutral"),
    ],
)
def test_invalid_segmentation_categories(field, value):
    models_and_values = {
        "product": (ProductRiskMetrics, "Savings Account"),
        "channel": (ChannelRiskMetrics, "Mobile App"),
        "tenure": (TenureRiskMetrics, "<1 yr"),
        "nps_category": (CategoryRiskMetrics, "promoter"),
    }

    model, valid_value = models_and_values[field]

    valid_data = {
        field: valid_value,
        "metrics": {
            "customers": 10,
            "annual_revenue": 100000,
            "revenue_at_risk": 15000,
            "risk_percentage": 15.0,
        },
    }

    valid_data[field] = value

    with pytest.raises(ValidationError):
        model.model_validate(valid_data)


# -----------------------------------------------------
# Business rule validators
# -----------------------------------------------------

def test_revenue_at_risk_cannot_exceed_annual_revenue():
    with pytest.raises(ValidationError):
        RevenueRiskMetrics(
            customers=10,
            annual_revenue=100000,
            revenue_at_risk=150000,
            risk_percentage=50,
        )


def test_mismatch_count_cannot_exceed_comments():
    with pytest.raises(ValidationError):
        MismatchAnalysis(
            total_comments_analyzed=5,
            total_mismatches=8,
            uncertain_comments=0,
            mismatches=[],
        )


def test_mismatch_count_matches_list_length():
    with pytest.raises(ValidationError):
        MismatchAnalysis(
            total_comments_analyzed=10,
            total_mismatches=2,
            uncertain_comments=0,
            mismatches=[
                ScoreCommentMismatch(
                    response_id="R001",
                    score=1,
                    comment="Excellent service",
                    sentiment="positive",
                    sentiment_reason="Positive experience",
                    mismatch_reason="Positive comment with detractor score",
                )
            ],
        )


# -----------------------------------------------------
# Optional fields and defaults
# -----------------------------------------------------

def test_optional_fields(valid_risk_output):
    data = copy.deepcopy(valid_risk_output)

    data.pop("overall")
    data.pop("mismatch_analysis")
    data.pop("risk_findings")
    data.pop("data_caveats")

    output = ChurnRiskAgentOutput.model_validate(data)

    assert output.overall is None
    assert output.mismatch_analysis is None
    assert output.risk_findings == []
    assert output.data_caveats == []


def test_default_lists_are_independent():
    assumptions = {
        "promoter": 0.03,
        "passive": 0.10,
        "detractor": 0.25,
    }

    first = ChurnRiskAgentOutput(
        data_status=False,
        churn_assumptions=ChurnAssumptions(**assumptions),
    )

    second = ChurnRiskAgentOutput(
        data_status=False,
        churn_assumptions=ChurnAssumptions(**assumptions),
    )

    first.data_caveats.append("First model limitation")

    assert second.data_caveats == []


# -----------------------------------------------------
# Mismatch validations
# -----------------------------------------------------

@pytest.mark.parametrize(
    "score",
    [-1, 11, 15],
)
def test_invalid_mismatch_score(score):
    with pytest.raises(ValidationError):
        ScoreCommentMismatch(
            response_id="R001",
            score=score,
            comment="Excellent service",
            sentiment="positive",
            sentiment_reason="Positive experience",
            mismatch_reason="Positive comment with detractor score",
        )


@pytest.mark.parametrize(
    "sentiment",
    ["happy", "sad", "unknown"],
)
def test_invalid_mismatch_sentiment(sentiment):
    with pytest.raises(ValidationError):
        ScoreCommentMismatch.model_validate({
            "response_id": "R001",
            "score": 2,
            "comment": "Excellent service",
            "sentiment": sentiment,
            "sentiment_reason": "Test reason",
            "mismatch_reason": "Positive comment with detractor score",
        })


# -----------------------------------------------------
# Missing fields and serialization
# -----------------------------------------------------

def test_missing_required_assumptions(valid_risk_output):
    data = copy.deepcopy(valid_risk_output)
    del data["churn_assumptions"]

    with pytest.raises(ValidationError):
        ChurnRiskAgentOutput.model_validate(data)


def test_json_serialization_roundtrip(valid_risk_output):
    output = ChurnRiskAgentOutput.model_validate(valid_risk_output)

    json_string = output.model_dump_json()
    restored = ChurnRiskAgentOutput.model_validate_json(json_string)

    assert restored == output
