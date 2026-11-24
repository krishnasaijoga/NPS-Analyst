
from nps_insights_agent.validation.fact_check import (
    FactCheckResult,
    check_numeric,
)


def test_matching_numeric_values():
    result = FactCheckResult()

    check_numeric(
        result,
        field="overall.nps",
        expected=18.5,
        actual=18.5,
    )

    assert result.passed is True
    assert result.issues == []


def test_incorrect_numeric_value():
    result = FactCheckResult()

    check_numeric(
        result,
        field="overall.nps",
        expected=18.5,
        actual=19.5,
    )

    assert result.passed is False
    assert len(result.issues) == 1
    assert result.issues[0].field == "overall.nps"


def test_rounding_tolerance():
    result = FactCheckResult()

    check_numeric(
        result,
        field="overall.nps",
        expected=18.504,
        actual=18.50,
        tolerance=0.01,
    )

    assert result.passed is True
