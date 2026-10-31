
import pytest
import pandas as pd
from unittest.mock import patch

from types import SimpleNamespace

from nps_insights_agent.tools.survey_tools import get_nps_overview, get_nps_by_segment, get_revenue_at_risk, get_callback_list, find_score_comment_mismatches


# Access the underlying Python function
# without invoking the Strands tool interface.
get_overview = get_nps_overview.__wrapped__ # type: ignore


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



# =====================================================
# Tests for get_nps_by_segment
# =====================================================

get_segments = get_nps_by_segment.__wrapped__ # type: ignore


@pytest.fixture
def mock_segment_data():
    """Sample dataset with product, channel and tenure."""

    return pd.DataFrame([
        # Savings Account: 2 promoters, 1 passive, 1 detractor
        {
            "response_id": "R001",
            "month": "2026-07",
            "product": "Savings Account",
            "channel": "Mobile App",
            "tenure": "3+ yrs",
            "score": 10
        },
        {
            "response_id": "R002",
            "month": "2026-07",
            "product": "Savings Account",
            "channel": "Branch",
            "tenure": "1-3 yrs",
            "score": 9
        },
        {
            "response_id": "R003",
            "month": "2026-08",
            "product": "Savings Account",
            "channel": "Mobile App",
            "tenure": "3+ yrs",
            "score": 8
        },
        {
            "response_id": "R004",
            "month": "2026-08",
            "product": "Savings Account",
            "channel": "Branch",
            "tenure": "1-3 yrs",
            "score": 4
        },

        # Credit Card: 1 promoter, 1 passive, 2 detractors
        {
            "response_id": "R005",
            "month": "2026-07",
            "product": "Credit Card",
            "channel": "Mobile App",
            "tenure": "<1 yr",
            "score": 9
        },
        {
            "response_id": "R006",
            "month": "2026-07",
            "product": "Credit Card",
            "channel": "Call Centre",
            "tenure": "<1 yr",
            "score": 7
        },
        {
            "response_id": "R007",
            "month": "2026-08",
            "product": "Credit Card",
            "channel": "Call Centre",
            "tenure": "1-3 yrs",
            "score": 3
        },
        {
            "response_id": "R008",
            "month": "2026-08",
            "product": "Credit Card",
            "channel": "Mobile App",
            "tenure": "3+ yrs",
            "score": 2
        },
    ])


@pytest.fixture
def mock_segment_loader(mock_segment_data):
    with patch(
        "nps_insights_agent.tools.survey_tools.load_survey_data",
        return_value=mock_segment_data
    ):
        yield


def test_segment_by_product(mock_segment_loader):
    result = get_segments(segment="product")

    segments = {
        item["segment_name"]: item
        for item in result["segments"]
    }

    assert result["data_status"] is True
    assert result["segment_dimension"] == "product"
    assert len(segments) == 2

    savings = segments["Savings Account"]

    assert savings["total_responses"] == 4
    assert savings["promoters"] == 2
    assert savings["passives"] == 1
    assert savings["detractors"] == 1
    assert savings["nps"] == 25.0

    credit = segments["Credit Card"]

    assert credit["total_responses"] == 4
    assert credit["promoters"] == 1
    assert credit["passives"] == 1
    assert credit["detractors"] == 2
    assert credit["nps"] == -25.0


def test_segment_by_channel(mock_segment_loader):
    result = get_segments(segment="channel")

    segments = {
        item["segment_name"]: item
        for item in result["segments"]
    }

    assert len(segments) == 3

    assert segments["Mobile App"]["nps"] == 25.0
    assert segments["Branch"]["nps"] == 0.0
    assert segments["Call Centre"]["nps"] == -50.0


def test_segment_by_tenure(mock_segment_loader):
    result = get_segments(segment="tenure")

    segments = {
        item["segment_name"]: item
        for item in result["segments"]
    }

    assert len(segments) == 3

    assert segments["3+ yrs"]["nps"] == 0.0
    assert segments["1-3 yrs"]["nps"] == -33.33
    assert segments["<1 yr"]["nps"] == 50.0


def test_filter_by_month(mock_segment_loader):
    result = get_segments(
        segment="product",
        month="2026-07"
    )

    segments = {
        item["segment_name"]: item
        for item in result["segments"]
    }

    assert result["month"] == "2026-07"

    assert segments["Savings Account"]["nps"] == 100.0
    assert segments["Credit Card"]["nps"] == 50.0

    assert segments["Savings Account"]["total_responses"] == 2
    assert segments["Credit Card"]["total_responses"] == 2


def test_month_optional(mock_segment_loader):
    result = get_segments(segment="product")

    assert result["month"] is None
    assert result["data_status"] is True


def test_case_insensitive_segment(mock_segment_loader):
    result = get_segments(segment=" PRODUCT ")

    assert result["segment_dimension"] == "product"


def test_invalid_segment(mock_segment_loader):
    with pytest.raises(ValueError):
        get_segments(segment="region")


def test_invalid_month(mock_segment_loader):
    with pytest.raises(ValueError):
        get_segments(
            segment="product",
            month="2026-12"
        )


def test_low_sample_flag(mock_segment_loader):
    result = get_segments(segment="product")

    for item in result["segments"]:
        assert item["low_sample"] is True


def test_segment_margin_of_error(mock_segment_loader):
    result = get_segments(segment="product")

    segments = {
        item["segment_name"]: item
        for item in result["segments"]
    }

    # Savings NPS values: [1, 1, 0, -1]
    # Credit NPS values:  [1, 0, -1, -1]
    # Both have the same sample standard deviation.

    assert segments["Savings Account"]["moe"] == pytest.approx(
        93.83, abs=0.02
    )

    assert segments["Credit Card"]["moe"] == pytest.approx(
        93.83, abs=0.02
    )


def test_segment_output_types(mock_segment_loader):
    result = get_segments(segment="product")

    for item in result["segments"]:
        assert isinstance(item["segment_name"], str)
        assert isinstance(item["total_responses"], int)
        assert isinstance(item["promoters"], int)
        assert isinstance(item["passives"], int)
        assert isinstance(item["detractors"], int)
        assert isinstance(item["nps"], float)
        assert isinstance(item["moe"], float)
        assert isinstance(item["low_sample"], bool)


def test_single_response_segment():
    df = pd.DataFrame([
        {
            "response_id": "R001",
            "month": "2026-07",
            "product": "Wealth",
            "channel": "Branch",
            "tenure": "3+ yrs",
            "score": 10
        }
    ])

    with patch(
        "nps_insights_agent.tools.survey_tools.load_survey_data",
        return_value=df
    ):
        result = get_segments(segment="product")

    wealth = result["segments"][0]

    assert wealth["total_responses"] == 1
    assert wealth["nps"] == 100.0
    assert wealth["moe"] is None
    assert wealth["low_sample"] is True




# =====================================================
# Tests for get_revenue_at_risk
# =====================================================

get_risk = get_revenue_at_risk.__wrapped__ # type: ignore


@pytest.fixture
def mock_revenue_data():
    return pd.DataFrame([
        {
            "response_id": "R001",
            "score": 10,
            "annual_revenue_inr": 10000,
            "product": "Savings Account",
            "channel": "Mobile App",
            "tenure": "3+ yrs"
        },
        {
            "response_id": "R002",
            "score": 8,
            "annual_revenue_inr": 20000,
            "product": "Savings Account",
            "channel": "Branch",
            "tenure": "1-3 yrs"
        },
        {
            "response_id": "R003",
            "score": 4,
            "annual_revenue_inr": 30000,
            "product": "Savings Account",
            "channel": "Mobile App",
            "tenure": "<1 yr"
        },
        {
            "response_id": "R004",
            "score": 9,
            "annual_revenue_inr": 40000,
            "product": "Credit Card",
            "channel": "Branch",
            "tenure": "<1 yr"
        },
        {
            "response_id": "R005",
            "score": 7,
            "annual_revenue_inr": 50000,
            "product": "Credit Card",
            "channel": "Call Centre",
            "tenure": "1-3 yrs"
        },
        {
            "response_id": "R006",
            "score": 2,
            "annual_revenue_inr": 60000,
            "product": "Wealth",
            "channel": "Mobile App",
            "tenure": "3+ yrs"
        }
    ])


@pytest.fixture
def mock_revenue_loader(mock_revenue_data):
    with patch(
        "nps_insights_agent.tools.survey_tools.load_survey_data",
        return_value=mock_revenue_data
    ):
        yield


def test_revenue_data_status(mock_revenue_loader):
    result = get_risk()

    assert result["data_status"] is True
    assert result["total_responses"] == 6
    assert result["currency"] == "INR"
    assert result["unit"] == "rupees"


def test_churn_assumptions(mock_revenue_loader):
    result = get_risk()

    assert result["churn_assumptions"] == {
        "promoter": 0.03,
        "passive": 0.10,
        "detractor": 0.25
    }


def test_overall_revenue_at_risk(mock_revenue_loader):
    result = get_risk()

    # Total annual revenue = 210000
    # Risk = 300 + 2000 + 7500 + 1200 + 5000 + 15000
    #      = 31000

    assert result["overall"]["annual_revenue"] == 210000
    assert result["overall"]["revenue_at_risk"] == 31000
    assert result["overall"]["risk_percentage"] == 14.76


def test_revenue_by_nps_category(mock_revenue_loader):
    result = get_risk()

    categories = {
        item["category"]: item
        for item in result["by_nps_category"]
    }

    assert categories["promoter"]["annual_revenue"] == 50000
    assert categories["promoter"]["revenue_at_risk"] == 1500

    assert categories["passive"]["annual_revenue"] == 70000
    assert categories["passive"]["revenue_at_risk"] == 7000

    assert categories["detractor"]["annual_revenue"] == 90000
    assert categories["detractor"]["revenue_at_risk"] == 22500


def test_revenue_by_product(mock_revenue_loader):
    result = get_risk()

    products = {
        item["category"]: item
        for item in result["by_product"]
    }

    assert len(products) == 3

    assert products["Savings Account"]["annual_revenue"] == 60000
    assert products["Savings Account"]["revenue_at_risk"] == 9800
    assert products["Savings Account"]["customers"] == 3

    assert products["Credit Card"]["annual_revenue"] == 90000
    assert products["Credit Card"]["revenue_at_risk"] == 6200
    assert products["Credit Card"]["customers"] == 2

    assert products["Wealth"]["annual_revenue"] == 60000
    assert products["Wealth"]["revenue_at_risk"] == 15000
    assert products["Wealth"]["customers"] == 1


def test_revenue_by_channel(mock_revenue_loader):
    result = get_risk()

    channels = {
        item["category"]: item
        for item in result["by_channel"]
    }

    assert len(channels) == 3

    assert channels["Mobile App"]["revenue_at_risk"] == 22800
    assert channels["Branch"]["revenue_at_risk"] == 3200
    assert channels["Call Centre"]["revenue_at_risk"] == 5000


def test_revenue_by_tenure(mock_revenue_loader):
    result = get_risk()

    tenures = {
        item["category"]: item
        for item in result["by_tenure"]
    }

    assert len(tenures) == 3

    assert tenures["<1 yr"]["revenue_at_risk"] == 8700
    assert tenures["1-3 yrs"]["revenue_at_risk"] == 7000
    assert tenures["3+ yrs"]["revenue_at_risk"] == 15300


def test_group_risk_percentages(mock_revenue_loader):
    result = get_risk()

    for group_name in [
        "by_product",
        "by_channel",
        "by_tenure",
        "by_nps_category"
    ]:
        for group in result[group_name]:
            expected = (
                group["revenue_at_risk"] * 100
                / group["annual_revenue"]
                if group["annual_revenue"] > 0 else 0.0
            )

            assert group["risk_percentage"] == pytest.approx(
                expected, abs=0.01
            )


def test_breakdown_totals_match_overall(mock_revenue_loader):
    result = get_risk()

    overall_revenue = result["overall"]["annual_revenue"]
    overall_risk = result["overall"]["revenue_at_risk"]

    for group_name in [
        "by_product",
        "by_channel",
        "by_tenure",
        "by_nps_category"
    ]:
        total_revenue = sum(
            item["annual_revenue"] for item in result[group_name]
        )

        total_risk = sum(
            item["revenue_at_risk"] for item in result[group_name]
        )

        assert total_revenue == pytest.approx(
            overall_revenue, abs=0.01
        )
        assert total_risk == pytest.approx(
            overall_risk, abs=0.01
        )


def test_zero_revenue():
    df = pd.DataFrame([
        {
            "response_id": "R001",
            "score": 3,
            "annual_revenue_inr": 0,
            "product": "Savings Account",
            "channel": "Branch",
            "tenure": "<1 yr"
        }
    ])

    with patch(
        "nps_insights_agent.tools.survey_tools.load_survey_data",
        return_value=df
    ):
        result = get_risk()

    assert result["overall"]["annual_revenue"] == 0
    assert result["overall"]["revenue_at_risk"] == 0
    assert result["overall"]["risk_percentage"] == 0.0

    assert result["by_product"][0]["risk_percentage"] == 0.0


def test_empty_revenue_dataset():
    empty_df = pd.DataFrame()

    with patch(
        "nps_insights_agent.tools.survey_tools.load_survey_data",
        return_value=empty_df
    ):
        result = get_risk()

    assert result["data_status"] is False
    assert result["total_responses"] == 0
    assert result["currency"] == "INR"
    assert result["unit"] == "rupees"


def test_original_dataframe_unchanged(mock_revenue_data):
    original = mock_revenue_data.copy(deep=True)

    with patch(
        "nps_insights_agent.tools.survey_tools.load_survey_data",
        return_value=mock_revenue_data
    ):
        get_risk()

    pd.testing.assert_frame_equal(mock_revenue_data, original)


def test_revenue_output_structure(mock_revenue_loader):
    result = get_risk()

    expected_keys = {
        "data_status",
        "total_responses",
        "currency",
        "unit",
        "overall",
        "churn_assumptions",
        "by_nps_category",
        "by_product",
        "by_channel",
        "by_tenure",
        "data_caveats"
    }

    assert expected_keys.issubset(result.keys())

    assert isinstance(result["overall"], dict)
    assert isinstance(result["by_product"], list)
    assert isinstance(result["by_channel"], list)
    assert isinstance(result["by_tenure"], list)
    assert isinstance(result["by_nps_category"], list)
    assert isinstance(result["data_caveats"], list)



# =====================================================
# Tests for get_callback_list
# =====================================================

get_callbacks = get_callback_list.__wrapped__ # type: ignore


@pytest.fixture
def mock_callback_data():
    return pd.DataFrame([
        {
            "response_id": "R001",
            "score": 2,
            "annual_revenue_inr": 300000,
            "product": "Wealth",
            "comment": "Unexpected advisory fees"
        },
        {
            "response_id": "R002",
            "score": 4,
            "annual_revenue_inr": 15000,
            "product": "Savings Account",
            "comment": "OTP not working"
        },
        {
            "response_id": "R003",
            "score": 6,
            "annual_revenue_inr": 8000,
            "product": "Credit Card",
            "comment": "Long customer support wait"
        },
        {
            "response_id": "R004",
            "score": 9,
            "annual_revenue_inr": 200000,
            "product": "Wealth",
            "comment": "Excellent service"
        },
        {
            "response_id": "R005",
            "score": 1,
            "annual_revenue_inr": 75000,
            "product": "Savings Account",
            "comment": "Unable to access account"
        },
        {
            "response_id": "R006",
            "score": 8,
            "annual_revenue_inr": 500000,
            "product": "Wealth",
            "comment": "Good experience"
        }
    ])


@pytest.fixture
def mock_callback_loader(mock_callback_data):
    with patch(
        "nps_insights_agent.tools.survey_tools.load_survey_data",
        return_value=mock_callback_data
    ):
        yield


def test_callback_data_status(mock_callback_loader):
    result = get_callbacks()

    assert result["data_status"] is True
    assert result["total_candidates"] == 4
    assert result["returned_candidates"] == 4


def test_only_detractors_in_callback_list(mock_callback_loader):
    result = get_callbacks()

    for customer in result["callback_list"]:
        assert 0 <= customer["nps_score"] <= 6

    returned_ids = {
        customer["response_id"]
        for customer in result["callback_list"]
    }

    # Promoter and passive must be excluded
    assert "R004" not in returned_ids
    assert "R006" not in returned_ids


def test_callback_revenue_at_risk(mock_callback_loader):
    result = get_callbacks()

    customers = {
        customer["response_id"]: customer
        for customer in result["callback_list"]
    }

    # Detractors have an assumed churn probability of 25%

    assert customers["R001"]["revenue_at_risk_inr"] == 75000
    assert customers["R002"]["revenue_at_risk_inr"] == 3750
    assert customers["R003"]["revenue_at_risk_inr"] == 2000
    assert customers["R005"]["revenue_at_risk_inr"] == 18750


def test_callback_ranking(mock_callback_loader):
    result = get_callbacks()

    returned_ids = [
        customer["response_id"]
        for customer in result["callback_list"]
    ]

    assert returned_ids == [
        "R001",
        "R005",
        "R002",
        "R003"
    ]


def test_priority_numbers(mock_callback_loader):
    result = get_callbacks()

    priorities = [
        customer["priority"]
        for customer in result["callback_list"]
    ]

    assert priorities == [1, 2, 3, 4]


def test_top_n_limit(mock_callback_loader):
    result = get_callbacks(top_n=2)

    assert result["total_candidates"] == 4
    assert result["returned_candidates"] == 2
    assert len(result["callback_list"]) == 2

    assert result["callback_list"][0]["response_id"] == "R001"
    assert result["callback_list"][1]["response_id"] == "R005"


def test_top_n_greater_than_candidates(mock_callback_loader):
    result = get_callbacks(top_n=100)

    assert result["total_candidates"] == 4
    assert result["returned_candidates"] == 4
    assert len(result["callback_list"]) == 4


@pytest.mark.parametrize("top_n", [0, -1, -10])
def test_invalid_top_n(mock_callback_loader, top_n):
    with pytest.raises(ValueError):
        get_callbacks(top_n=top_n)


def test_no_detractors():
    df = pd.DataFrame([
        {
            "response_id": "R001",
            "score": 10,
            "annual_revenue_inr": 100000,
            "product": "Wealth",
            "comment": "Excellent"
        },
        {
            "response_id": "R002",
            "score": 8,
            "annual_revenue_inr": 20000,
            "product": "Savings Account",
            "comment": "Good"
        }
    ])

    with patch(
        "nps_insights_agent.tools.survey_tools.load_survey_data",
        return_value=df
    ):
        result = get_callbacks()

    assert result["data_status"] is True
    assert result["total_candidates"] == 0
    assert result["returned_candidates"] == 0
    assert result["callback_list"] == []


def test_empty_callback_dataset():
    empty_df = pd.DataFrame()

    with patch(
        "nps_insights_agent.tools.survey_tools.load_survey_data",
        return_value=empty_df
    ):
        result = get_callbacks()

    assert result["data_status"] is False
    assert result["total_responses"] == 0


def test_callback_output_structure(mock_callback_loader):
    result = get_callbacks()

    expected_keys = {
        "response_id",
        "nps_score",
        "product",
        "annual_revenue_inr",
        "revenue_at_risk_inr",
        "comment",
        "priority"
    }

    for customer in result["callback_list"]:
        assert expected_keys.issubset(customer.keys())

        assert isinstance(customer["response_id"], str)
        assert isinstance(customer["nps_score"], int)
        assert isinstance(customer["product"], str)
        assert isinstance(customer["annual_revenue_inr"], int)
        assert isinstance(customer["revenue_at_risk_inr"], float)
        assert isinstance(customer["priority"], int)


def test_original_callback_dataframe_unchanged(mock_callback_data):
    original = mock_callback_data.copy(deep=True)

    with patch(
        "nps_insights_agent.tools.survey_tools.load_survey_data",
        return_value=mock_callback_data
    ):
        get_callbacks()

    pd.testing.assert_frame_equal(mock_callback_data, original)



# ====================================================
# Tests for find_score_comment_mismatches
# ====================================================

get_mismatches = find_score_comment_mismatches.__wrapped__ # type: ignore


@pytest.fixture
def mock_mismatch_data():
    return pd.DataFrame([
        {
            "response_id": "R001",
            "score": 2,
            "comment": "Excellent banking service!"
        },
        {
            "response_id": "R002",
            "score": 10,
            "comment": "Terrible customer support."
        },
        {
            "response_id": "R003",
            "score": 9,
            "comment": "Great experience."
        },
        {
            "response_id": "R004",
            "score": 3,
            "comment": "Very bad experience."
        },
        {
            "response_id": "R005",
            "score": 8,
            "comment": "I visited the branch."
        },
        {
            "response_id": "R006",
            "score": 6,
            "comment": "Good app, but slow support."
        },
        {
            "response_id": "R007",
            "score": 4,
            "comment": None
        },
        {
            "response_id": "R008",
            "score": 1,
            "comment": "   "
        }
    ])


@pytest.fixture
def mock_sentiment_classifier(mock_mismatch_data):
    sentiment_results = {
        "Excellent banking service!": ("positive", "Strong satisfaction"),
        "Terrible customer support.": ("negative", "Customer dissatisfaction"),
        "Great experience.": ("positive", "Positive experience"),
        "Very bad experience.": ("negative", "Negative experience"),
        "I visited the branch.": ("neutral", "Factual statement"),
        "Good app, but slow support.": ("mixed", "Both positive and negative")
    }

    def fake_classify(comment):
        sentiment, reason = sentiment_results[comment]
        return SimpleNamespace(sentiment=sentiment, reason=reason)

    with (
        patch(
            "nps_insights_agent.tools.survey_tools.load_survey_data",
            return_value=mock_mismatch_data
        ),
        patch(
            "nps_insights_agent.tools.survey_tools.classify_sentiment",
            side_effect=fake_classify
        )
    ):
        yield


def test_mismatch_detection(mock_sentiment_classifier):
    result = get_mismatches()

    assert result["data_status"] is True
    assert result["total_comments_analyzed"] == 6
    assert result["total_mismatches"] == 2

    ids = {item["response_id"] for item in result["mismatches"]}
    assert ids == {"R001", "R002"}


def test_mismatch_reasons(mock_sentiment_classifier):
    result = get_mismatches()

    mismatches = {
        item["response_id"]: item
        for item in result["mismatches"]
    }

    assert mismatches["R001"]["mismatch_reason"] == (
        "Positive comment with detractor NPS score"
    )
    assert mismatches["R002"]["mismatch_reason"] == (
        "Negative comment with promoter NPS score"
    )


def test_uncertain_comment_count(mock_sentiment_classifier):
    result = get_mismatches()

    assert result["uncertain_comments"] == 2


def test_mismatch_output_structure(mock_sentiment_classifier):
    result = get_mismatches()

    expected_keys = {
        "response_id",
        "score",
        "comment",
        "sentiment",
        "sentiment_reason",
        "mismatch_reason"
    }

    for item in result["mismatches"]:
        assert expected_keys.issubset(item)
        assert isinstance(item["score"], int)
        assert isinstance(item["sentiment_reason"], str)


def test_mismatch_classifier_called_only_for_valid_comments(
    mock_mismatch_data
):
    with (
        patch(
            "nps_insights_agent.tools.survey_tools.load_survey_data",
            return_value=mock_mismatch_data
        ),
        patch(
            "nps_insights_agent.tools.survey_tools.classify_sentiment",
            return_value=SimpleNamespace(
                sentiment="neutral",
                reason="No clear sentiment"
            )
        ) as mocked_classifier
    ):
        result = get_mismatches()

    assert result["total_comments_analyzed"] == 6
    assert mocked_classifier.call_count == 6


def test_mismatch_empty_dataset():
    with patch(
        "nps_insights_agent.tools.survey_tools.load_survey_data",
        return_value=pd.DataFrame()
    ):
        result = get_mismatches()

    assert result["data_status"] is False
    assert result["total_comments_analyzed"] == 0
    assert result["total_mismatches"] == 0
    assert result["mismatches"] == []


def test_mismatch_all_comments_missing():
    df = pd.DataFrame([
        {"response_id": "R001", "score": 2, "comment": None},
        {"response_id": "R002", "score": 10, "comment": "  "}
    ])

    with (
        patch(
            "nps_insights_agent.tools.survey_tools.load_survey_data",
            return_value=df
        ),
        patch(
            "nps_insights_agent.tools.survey_tools.classify_sentiment"
        ) as mocked_classifier
    ):
        result = get_mismatches()

    assert result["total_comments_analyzed"] == 0
    assert result["total_mismatches"] == 0
    assert result["mismatches"] == []
    mocked_classifier.assert_not_called()


def test_mismatch_original_dataframe_unchanged(mock_mismatch_data):
    original = mock_mismatch_data.copy(deep=True)

    with (
        patch(
            "nps_insights_agent.tools.survey_tools.load_survey_data",
            return_value=mock_mismatch_data
        ),
        patch(
            "nps_insights_agent.tools.survey_tools.classify_sentiment",
            return_value=SimpleNamespace(
                sentiment="neutral",
                reason="No clear sentiment"
            )
        )
    ):
        get_mismatches()

    pd.testing.assert_frame_equal(mock_mismatch_data, original)
