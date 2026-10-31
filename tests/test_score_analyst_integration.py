
import os

import pytest
from dotenv import load_dotenv

from nps_insights_agent.agents.score_analyst import (
    create_score_analyst,
)

load_dotenv()


@pytest.mark.integration
def test_score_analyst_overall_nps():
    if not os.getenv("GROQ_API_KEY"):
        pytest.skip("GROQ_API_KEY not configured")

    agent = create_score_analyst()

    response = agent(
        "Use the survey tools to report the overall "
        "NPS and total response count."
    )

    answer = str(response)

    assert answer.strip()
    assert "18.5" in answer
    assert "600" in answer


@pytest.mark.integration
def test_score_analyst_segment_analysis():
    if not os.getenv("GROQ_API_KEY"):
        pytest.skip("GROQ_API_KEY not configured")

    agent = create_score_analyst()

    response = agent(
        "Analyze NPS by product using the survey tools. "
        "Include sample sizes and identify low-sample segments."
    )

    answer = str(response)

    assert answer.strip()
    assert "NPS" in answer.upper()
    assert "product" in answer.lower()
