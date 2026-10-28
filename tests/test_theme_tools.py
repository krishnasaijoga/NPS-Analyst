
import pandas as pd
import pytest
from unittest.mock import patch

from nps_insights_agent.tools.theme_tools import get_comment_themes


# Access the original function without the Strands @tool wrapper
get_themes = get_comment_themes.__wrapped__ # type: ignore


@pytest.fixture
def mock_survey_data():
    return pd.DataFrame([
        {
            "month": "2026-09",
            "score": 3,
            "comment": "OTP not received during login"
        },
        {
            "month": "2026-09",
            "score": 5,
            "comment": "Unexpected annual fees"
        },
        {
            "month": "2026-09",
            "score": 9,
            "comment": "UPI is quick and smooth"
        },
        {
            "month": "2026-08",
            "score": 2,
            "comment": "Login keeps failing"
        },
        {
            "month": "2026-09",
            "score": 7,
            "comment": "Something completely unrelated"
        },
        {
            "month": "2026-09",
            "score": 4,
            "comment": None
        }
    ])


@pytest.fixture
def mock_loader(mock_survey_data):
    with patch(
        "nps_insights_agent.tools.theme_tools.load_survey_data",
        return_value=mock_survey_data
    ):
        yield


def test_total_comments(mock_loader):
    result = get_themes()

    assert result["data_status"] is True
    assert result["total_comments_analyzed"] == 5


def test_filter_by_month(mock_loader):
    result = get_themes(month="2026-09")

    assert result["total_comments_analyzed"] == 4


def test_filter_by_nps_category(mock_loader):
    result = get_themes(nps_category="Detractor")

    assert result["total_comments_analyzed"] == 3


def test_case_insensitive_category(mock_loader):
    result = get_themes(nps_category="dEtRaCtOr")

    assert result["total_comments_analyzed"] == 3


def test_theme_detection(mock_loader):
    result = get_themes(
        month="2026-09",
        nps_category="Detractor"
    )

    themes = {
        item["theme"]: item
        for item in result["themes"]
    }

    assert themes["OTP & Verification Issues"]["count"] == 1
    assert themes["Login & Account Access"]["count"] == 1
    assert themes["Fees & Unexpected Charges"]["count"] == 1


def test_multiple_themes_per_comment(mock_loader):
    result = get_themes(
        month="2026-09",
        nps_category="Detractor"
    )

    themes = {
        item["theme"]: item["count"]
        for item in result["themes"]
    }

    # Same comment belongs to both OTP and Login themes
    assert themes["OTP & Verification Issues"] == 1
    assert themes["Login & Account Access"] == 1

    # But it should count as only one matched comment
    assert result["matched_comments"] == 2
    assert result["unmatched_comments"] == 0


def test_unmatched_comments(mock_loader):
    result = get_themes(month="2026-09")

    assert result["matched_comments"] == 3
    assert result["unmatched_comments"] == 1


def test_percentage_calculation(mock_loader):
    result = get_themes(
        month="2026-09",
        nps_category="Detractor"
    )

    themes = {
        item["theme"]: item
        for item in result["themes"]
    }

    # One OTP comment out of two nonempty detractor comments
    assert themes["OTP & Verification Issues"]["percentage"] == 50.0


def test_empty_month(mock_loader):
    result = get_themes(month="2026-12")

    assert result["data_status"] is False
    assert result["total_comments_analyzed"] == 0
    assert result["themes"] == []


def test_invalid_nps_category(mock_loader):
    with pytest.raises(ValueError):
        get_themes(nps_category="Unknown")


def test_themes_sorted_by_count(mock_loader):
    result = get_themes()

    counts = [item["count"] for item in result["themes"]]

    assert counts == sorted(counts, reverse=True)


def test_maximum_three_example_comments(mock_loader):
    result = get_themes()

    for theme in result["themes"]:
        assert len(theme["example_comments"]) <= 3
