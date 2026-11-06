
import os

import pytest
from dotenv import load_dotenv

from nps_insights_agent.agents.churn_risk import (
    create_churn_risk_agent,
)

load_dotenv()


@pytest.mark.integration
def test_churn_risk_financial_analysis():
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not configured")

    agent = create_churn_risk_agent()

    response = agent(
        "Use get_revenue_at_risk to analyze the bank's "
        "estimated annual revenue at risk. "
        "Identify the highest-exposure products and explain "
        "the assumed churn probabilities."
    )

    answer = str(response)

    assert answer.strip()
    assert "revenue" in answer.lower()
    assert "risk" in answer.lower()


@pytest.mark.integration
def test_churn_risk_callback_analysis():
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not configured")

    agent = create_churn_risk_agent()

    response = agent(
        "Use get_callback_list to identify the five highest-priority "
        "detractor customers for follow-up. "
        "Explain the prioritization rule."
    )

    answer = str(response)

    assert answer.strip()
    assert "callback" in answer.lower() or "detractor" in answer.lower()
