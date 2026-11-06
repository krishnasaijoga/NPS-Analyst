
import os

import pytest
from dotenv import load_dotenv

from nps_insights_agent.agents.action_planner import (
    create_action_planner_agent,
)

load_dotenv()


@pytest.mark.integration
def test_callback_prioritization():
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not configured")

    agent = create_action_planner_agent()

    response = agent(
        "Use get_callback_list to identify the top five "
        "detractor customers for follow-up. "
        "Explain their prioritization."
    )

    answer = str(response)

    assert answer.strip()
    assert "detractor" in answer.lower() or "callback" in answer.lower()


@pytest.mark.integration
def test_fix_impact_simulation():
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not configured")

    agent = create_action_planner_agent()

    response = agent(
        "Use estimate_fix_impact to simulate resolving "
        "50% of OTP & Verification Issues. "
        "Report the NPS improvement and estimated "
        "revenue-at-risk reduction in INR."
    )

    answer = str(response)

    assert answer.strip()
    assert "nps" in answer.lower()
    assert "risk" in answer.lower()
