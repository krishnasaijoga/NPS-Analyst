
import pytest
import pandas as pd
from unittest.mock import patch

from nps_insights_agent.tools.survey_tools import get_nps_overview


# Access the underlying Python function
# without invoking the Strands tool interface.
get_overview = get_nps_overview.__wrapped__


@pytest.fixture
def mock_survey_data():
    """A predictable dataset containing 10 responses."""

    return pd.DataFrame([
        {"response_id": "R001", "month": "2026-07", "score": 10},
        {"response_id": "R002", "month": "2026-07", "score": 9},
        {"response_id": "R003", "month": "2026-07", "score": 8},
        {"response_id": "R004", "month": "2026-07", "score": 5},
        {"response_id": "R005", "month": "2026-07", "score": 3},

        {"response_id": "R006", "month": "2026-08", "score": 10},
        {"response_id": "R007", "month": "2026-08", "score": 9},
        {"response_id": "R008", "month": "2026-08", "score": 9},
        {"response_id": "R009", "month": "2026-08", "score": 7},
        {"response_id": "R010", "month": "2026-08", "score": 4},
    ])


@pytest.fixture
def mock_loader(mock_survey_data):
    """Replace the actual CSV loader with sample data."""

    with patch(
        "nps_insights_agent.tools.survey_tools.load_survey_data",
        return_value=mock_survey_data
    ):
        yield


def test_total_responses(mock_loader):
    result = get_overview()

    assert result["data_status"] is True
    assert result["total_responses"] == 10


def test_nps_classification_counts(mock_loader):
    result = get_overview()

    assert result["promoters"] == 5
    assert result["passives"] == 2
    assert result["detractors"] == 3


def test_overall_nps(mock_loader):
    result = get_overview()

    # (5 promoters - 3 detractors) / 10 * 100 = 20
    assert result["overall_nps"] == 20.0


def test_nps_percentages(mock_loader):
    result = get_overview()

    assert result["promoter_percentage"] == 50.0
    assert result["passive_percentage"] == 20.0
    assert result["detractor_percentage"] == 30.0

    assert (
        result["promoter_percentage"]
        + result["passive_percentage"]
        + result["detractor_percentage"]
    ) == pytest.approx(100.0)


def test_monthly_nps(mock_loader):
    result = get_overview()

    monthly = {
        item["month"]: item
        for item in result["monthly_trend"]
    }

    # July: 2 promoters, 2 detractors, 5 responses
    assert monthly["2026-07"]["nps"] == 0.0

    # August: 3 promoters, 1 detractor, 5 responses
    assert monthly["2026-08"]["nps"] == 40.0


def test_monthly_response_counts(mock_loader):
    result = get_overview()

    monthly = {
        item["month"]: item
        for item in result["monthly_trend"]
    }

    assert monthly["2026-07"]["total_responses"] == 5
    assert monthly["2026-08"]["total_responses"] == 5


def test_margin_of_error(mock_loader):
    result = get_overview()

    # NPS classification values:
    # 5 promoters (+1), 2 passives (0), 3 detractors (-1)
    # Expected sample-based 95% MOE ≈ 57.56 points
    assert result["margin_of_error_95"] == pytest.approx(
        56.96, abs=0.02
    )


def test_monthly_trend_structure(mock_loader):
    result = get_overview()

    assert isinstance(result["monthly_trend"], list)
    assert len(result["monthly_trend"]) == 2

    for item in result["monthly_trend"]:
        assert isinstance(item["month"], str)
        assert isinstance(item["total_responses"], int)
        assert isinstance(item["nps"], float)


def test_empty_dataset():
    empty_df = pd.DataFrame(
        columns=["response_id", "month", "score"]
    )

    with patch(
        "nps_insights_agent.tools.survey_tools.load_survey_data",
        return_value=empty_df
    ):
        result = get_overview()

    assert result["data_status"] is False
    assert result["total_responses"] == 0
    assert result["overall_nps"] == 0.0
    assert result["margin_of_error_95"] is None
    assert result["monthly_trend"] == []


def test_single_response():
    single_df = pd.DataFrame([
        {
            "response_id": "R001",
            "month": "2026-07",
            "score": 10
        }
    ])

    with patch(
        "nps_insights_agent.tools.survey_tools.load_survey_data",
        return_value=single_df
    ):
        result = get_overview()

    assert result["total_responses"] == 1
    assert result["overall_nps"] == 100.0
    assert result["margin_of_error_95"] is None
