
import os

import pytest
from dotenv import load_dotenv

from nps_insights_agent.schemas.agent_outputs import (
    VoiceOfCustomerOutput,
)
from nps_insights_agent.agents.voice_of_customer import (
    create_voice_of_customer
)

load_dotenv()


@pytest.mark.integration
def test_voice_of_customer_structured_output():

    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OpenAI API key not configured")

    agent = create_voice_of_customer()

    result = agent(
        """
        Analyze detractor feedback for September 2026.

        Use get_comment_themes to identify recurring themes.

        Include:
        - Number of comments analyzed
        - Matched and unmatched comments
        - Theme counts and percentages
        - Representative customer comments
        - Evidence-backed findings
        - Limitations

        Do not invent themes or numerical values.
        """
    )

    output = result.structured_output

    assert isinstance(output, VoiceOfCustomerOutput)
    assert output.data_status is True

    assert output.month == "2026-09"
    assert output.nps_category == "detractor"

    assert (
        output.matched_comments + output.unmatched_comments
        == output.total_comments_analyzed
    )

    assert len(output.themes) > 0

    for theme in output.themes:
        assert theme.count >= 0
        assert 0 <= theme.percentage <= 100

    print(output.model_dump_json(indent=2))
