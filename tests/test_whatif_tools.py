
from unittest.mock import patch

import pandas as pd
import pytest

from nps_insights_agent.tools.whatif_tools import (
    estimate_fix_impact,
    THEMES,
)


# Access the underlying function without the Strands tool wrapper
estimate_impact = estimate_fix_impact.__wrapped__ # type: ignore


@pytest.fixture
def mock_survey():
    return pd.DataFrame([
        {
            "response_id": "R001",
            "score": 2,
            "annual_revenue_inr": 100000,
            "comment": "OTP verification failed"
        },
        {
            "response_id": "R002",
            "score": 5,
            "annual_revenue_inr": 200000,
            "comment": "OTP not working"
        },
        {
            "response_id": "R003",
            "score": 7,
            "annual_revenue_inr": 50000,
            "comment": "OTP issue resolved"
        },
        {
            "response_id": "R004",
            "score": 8,
            "annual_revenue_inr": 80000,
            "comment": "OTP delays"
        },
        {
            "response_id": "R005",
            "score": 10,
            "annual_revenue_inr": 150000,
            "comment": "OTP works perfectly"
        },
        {
            "response_id": "R006",
            "score": 9,
            "annual_revenue_inr": 120000,
            "comment": "Great mobile banking experience"
        },
        {
            "response_id": "R007",
            "score": 3,
            "annual_revenue_inr": 30000,
            "comment": "Poor customer support"
        },
        {
            "response_id": "R008",
            "score": 8,
            "annual_revenue_inr": 40000,
            "comment": "Good banking experience"
        },
    ])


@pytest.fixture
def mock_data_loader(mock_survey):
    with patch(
        "nps_insights_agent.tools.whatif_tools.load_survey_data",
        return_value=mock_survey,
    ):
        yield


@pytest.fixture
def otp_theme():
    # Ensure the exact theme name matches your dictionary.
    return next(
        name for name, keywords in THEMES.items()
        if any("otp" in word.lower() for word in keywords)
    )


def test_baseline_nps(mock_data_loader, otp_theme):
    result = estimate_impact(otp_theme, resolution_rate=0)

    # 2 promoters, 3 passives, 3 detractors
    # NPS = ((2 - 3) / 8) * 100 = -12.5
    assert result["nps_impact"]["baseline_nps"] == -12.5


def test_affected_customers(mock_data_loader, otp_theme):
    result = estimate_impact(otp_theme)

    assert result["total_responses"] == 8
    assert result["affected_responses"] == 5
    assert result["affected_customers"] == {
        "promoters": 1,
        "passives": 2,
        "detractors": 2,
    }


def test_half_resolution_transitions(mock_data_loader, otp_theme):
    result = estimate_impact(otp_theme, resolution_rate=0.5)

    assert result["simulated_transitions"] == {
        "detractor_to_passive": 1,
        "passive_to_promoter": 1,
    }


def test_half_resolution_nps(mock_data_loader, otp_theme):
    result = estimate_impact(otp_theme, resolution_rate=0.5)

    assert result["nps_impact"]["baseline_nps"] == -12.5
    assert result["nps_impact"]["simulated_nps"] == 12.5
    assert result["nps_impact"]["nps_improvement_points"] == 25.0


def test_baseline_revenue_at_risk(mock_data_loader, otp_theme):
    result = estimate_impact(otp_theme, resolution_rate=0)

    # Detractors: (100000 + 200000 + 30000) * 0.25 = 82500
    # Passives: (50000 + 80000 + 40000) * 0.10 = 17000
    # Promoters: (150000 + 120000) * 0.03 = 8100
    assert result["revenue_impact"]["baseline_revenue_at_risk"] == 107600.0


def test_half_resolution_revenue(mock_data_loader, otp_theme):
    result = estimate_impact(otp_theme, resolution_rate=0.5)

    # Highest-revenue detractor R002: 200000 * 0.15 = 30000
    # Highest-revenue passive R004: 80000 * 0.07 = 5600
    # Total risk reduction = 35600
    assert result["revenue_impact"]["estimated_risk_reduction"] == 35600.0
    assert result["revenue_impact"]["simulated_revenue_at_risk"] == 72000.0


def test_zero_resolution(mock_data_loader, otp_theme):
    result = estimate_impact(otp_theme, resolution_rate=0)

    assert result["simulated_transitions"] == {
        "detractor_to_passive": 0,
        "passive_to_promoter": 0,
    }
    assert result["nps_impact"]["nps_improvement_points"] == 0
    assert result["revenue_impact"]["estimated_risk_reduction"] == 0


def test_full_resolution(mock_data_loader, otp_theme):
    result = estimate_impact(otp_theme, resolution_rate=1)

    assert result["simulated_transitions"] == {
        "detractor_to_passive": 2,
        "passive_to_promoter": 2,
    }

    assert result["nps_impact"]["simulated_nps"] == 37.5
    assert result["nps_impact"]["nps_improvement_points"] == 50.0

    # Detractors: 300000 * 0.15 = 45000
    # Passives: 130000 * 0.07 = 9100
    assert result["revenue_impact"]["estimated_risk_reduction"] == 54100.0


@pytest.mark.parametrize("rate", [-0.1, 1.1, 2])
def test_invalid_resolution_rate(mock_data_loader, otp_theme, rate):
    with pytest.raises(ValueError):
        estimate_impact(otp_theme, resolution_rate=rate)


def test_invalid_theme(mock_data_loader):
    with pytest.raises(ValueError):
        estimate_impact("Invalid Theme", resolution_rate=0.5)


def test_empty_dataset(otp_theme):
    with patch(
        "nps_insights_agent.tools.whatif_tools.load_survey_data",
        return_value=pd.DataFrame(),
    ):
        result = estimate_impact(otp_theme)

    assert result["data_status"] is False
    assert result["total_responses"] == 0
    assert result["affected_responses"] == 0


def test_original_dataframe_unchanged(mock_survey, otp_theme):
    original = mock_survey.copy(deep=True)

    with patch(
        "nps_insights_agent.tools.whatif_tools.load_survey_data",
        return_value=mock_survey,
    ):
        estimate_impact(otp_theme)

    pd.testing.assert_frame_equal(mock_survey, original)
