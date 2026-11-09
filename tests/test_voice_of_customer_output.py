
import pytest
from pydantic import ValidationError

from nps_insights_agent.schemas.agent_outputs import (
    VoiceOfCustomerOutput,
    ThemeSummary,
    ThemeFindings,
)


@pytest.fixture
def valid_output():
    return {
        "data_status": True,
        "total_comments_analyzed": 100,
        "matched_comments": 80,
        "unmatched_comments": 20,
        "themes": [
            {
                "theme": "OTP & Verification Issues",
                "count": 40,
                "percentage": 40.0,
                "example_comments": [
                    "OTP is not arriving",
                    "Verification keeps failing"
                ],
            },
            {
                "theme": "Customer Support Delays",
                "count": 30,
                "percentage": 30.0,
                "example_comments": [
                    "Support takes too long"
                ],
            },
        ],
        "key_findings": [
            {
                "finding": "OTP issues are frequently reported",
                "evidence": "40 out of 100 comments mention OTP issues",
                "implications": "Investigate OTP reliability",
                "supporting_themes": [
                    "OTP & Verification Issues"
                ],
            }
        ],
        "limitations": [
            "Theme extraction is keyword-based"
        ],
    }


def test_valid_output(valid_output):
    result = VoiceOfCustomerOutput.model_validate(valid_output)

    assert isinstance(result, VoiceOfCustomerOutput)
    assert result.data_status is True
    assert result.total_comments_analyzed == 100
    assert len(result.themes) == 2
    assert len(result.key_findings) == 1


def test_nested_models(valid_output):
    result = VoiceOfCustomerOutput.model_validate(valid_output)

    assert isinstance(result.themes[0], ThemeSummary)
    assert isinstance(result.key_findings[0], ThemeFindings)

    assert result.themes[0].count == 40
    assert result.key_findings[0].supporting_themes == [
        "OTP & Verification Issues"
    ]


def test_default_values():
    result = VoiceOfCustomerOutput(
        data_status=False,
        total_comments_analyzed=0,
        matched_comments=0,
        unmatched_comments=0,
    )

    assert result.month is None
    assert result.nps_category is None
    assert result.themes == []
    assert result.key_findings == []
    assert result.limitations == []


@pytest.mark.parametrize(
    "matched,unmatched,total",
    [
        (80, 10, 100),
        (90, 20, 100),
        (0, 0, 10),
    ],
)
def test_invalid_comment_counts(matched, unmatched, total):
    with pytest.raises(ValidationError):
        VoiceOfCustomerOutput(
            data_status=True,
            total_comments_analyzed=total,
            matched_comments=matched,
            unmatched_comments=unmatched,
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("total_comments_analyzed", -1),
        ("matched_comments", -1),
        ("unmatched_comments", -1),
    ],
)
def test_negative_counts_rejected(valid_output, field, value):
    data = valid_output.copy()
    data[field] = value

    with pytest.raises(ValidationError):
        VoiceOfCustomerOutput.model_validate(data)


@pytest.mark.parametrize(
    "percentage",
    [-0.1, 100.1, 150],
)
def test_invalid_percentage_rejected(valid_output, percentage):
    valid_output["themes"][0]["percentage"] = percentage

    with pytest.raises(ValidationError):
        VoiceOfCustomerOutput.model_validate(valid_output)


@pytest.mark.parametrize(
    "category",
    ["positive", "negative", "neutral", "invalid"],
)
def test_invalid_nps_category_rejected(valid_output, category):
    valid_output["nps_category"] = category

    with pytest.raises(ValidationError):
        VoiceOfCustomerOutput.model_validate(valid_output)


@pytest.mark.parametrize(
    "category",
    ["promoter", "passive", "detractor", None],
)
def test_valid_nps_categories(valid_output, category):
    valid_output["nps_category"] = category

    result = VoiceOfCustomerOutput.model_validate(valid_output)

    assert result.nps_category == category


def test_theme_count_cannot_be_negative(valid_output):
    valid_output["themes"][0]["count"] = -5

    with pytest.raises(ValidationError):
        VoiceOfCustomerOutput.model_validate(valid_output)


def test_missing_required_field(valid_output):
    del valid_output["matched_comments"]

    with pytest.raises(ValidationError):
        VoiceOfCustomerOutput.model_validate(valid_output)


def test_default_lists_are_independent():
    first = VoiceOfCustomerOutput(
        data_status=False,
        total_comments_analyzed=0,
        matched_comments=0,
        unmatched_comments=0,
    )

    second = VoiceOfCustomerOutput(
        data_status=False,
        total_comments_analyzed=0,
        matched_comments=0,
        unmatched_comments=0,
    )

    first.limitations.append("Test limitation")

    assert second.limitations == []


def test_serialization_roundtrip(valid_output):
    output = VoiceOfCustomerOutput.model_validate(valid_output)

    json_string = output.model_dump_json()
    restored = VoiceOfCustomerOutput.model_validate_json(json_string)

    assert restored == output
