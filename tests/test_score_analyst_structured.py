
import os

import pytest
from dotenv import load_dotenv

from nps_insights_agent.agents.score_analyst import (
    create_score_analyst,
)
from nps_insights_agent.schemas.agent_outputs import (
    ScoreAnalysisOutput,
)

load_dotenv()


@pytest.mark.integration
def test_score_analyst_structured_output():

    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not configured")

    agent = create_score_analyst()

    result = agent(
        "Analyze overall NPS, monthly trends, and "
        "product-level performance using the tools.",
        structured_output_model=ScoreAnalysisOutput,
    )

    output = result.structured_output

    assert isinstance(output, ScoreAnalysisOutput)
    assert output.data_status is True
    assert output.overall is not None

    assert output.overall.total_responses == 600
    assert output.overall.nps == 18.5

    assert len(output.monthly_trend) > 0
    assert len(output.segments) > 0
