
import os

import pytest
from dotenv import load_dotenv

from nps_insights_agent.agents.voice_of_customer import (
    create_voice_of_customer,
)

load_dotenv()


@pytest.mark.integration
def test_voice_of_customer_theme_analysis():
    if not os.getenv("GROQ_API_KEY"):
        pytest.skip("GROQ_API_KEY not configured")

    agent = create_voice_of_customer()

    response = agent(
        "Use get_comment_themes to identify the most "
        "frequent feedback themes across all survey comments. "
        "Include counts, examples, and limitations."
    )

    answer = str(response)

    assert answer.strip()
    assert "theme" in answer.lower()


@pytest.mark.integration
def test_voice_of_customer_detractor_feedback():
    if not os.getenv("GROQ_API_KEY"):
        pytest.skip("GROQ_API_KEY not configured")

    agent = create_voice_of_customer()

    response = agent(
        "Analyze the recurring feedback themes among "
        "detractors using get_comment_themes. "
        "Explain their main concerns."
    )

    answer = str(response)

    assert answer.strip()
    assert "detractor" in answer.lower()
