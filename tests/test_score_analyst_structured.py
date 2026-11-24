
import os

import pytest
from dotenv import load_dotenv

from nps_insights_agent.agents.score_analyst import (
    create_score_analyst,
)
from nps_insights_agent.schemas.agent_outputs import (
    ScoreAnalysisOutput,
)
from nps_insights_agent.tools.survey_tools import get_nps_overview

load_dotenv()


@pytest.mark.integration
def test_score_analyst_structured_output():

    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OpenAI API key not configured")

    agent = create_score_analyst()

    result = agent(
        "Analyze overall NPS, monthly trends and "
        "product-level performance using your tools."
    )

    output = result.structured_output

    # Verify structured output
    assert isinstance(output, ScoreAnalysisOutput)
    assert output.data_status is True
    assert output.overall is not None

    # Verify numerical grounding
    expected = get_nps_overview.__wrapped__() # type: ignore

    assert output.overall.total_responses == expected["total_responses"]

    # Adjust this key to match your existing tool dictionary.
    expected_nps = expected["overall_nps"]
    assert output.overall.nps == pytest.approx(
        expected_nps,
        abs=0.01,
    )

    # Verify that monthly and product data are present
    assert len(output.monthly_trend) > 0
    assert len(output.segments) > 0