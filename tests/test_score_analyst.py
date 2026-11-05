
from unittest.mock import MagicMock, patch

from nps_insights_agent.agents.score_analyst import (
    create_score_analyst,
    SCORE_ANALYST_SYSTEM_PROMPT,
)


def test_agent_creation():
    with (
        patch(
            "nps_insights_agent.agents.score_analyst.OpenAIModel"
        ) as mock_model,
        patch(
            "nps_insights_agent.agents.score_analyst.Agent"
        ) as mock_agent,
        patch.dict(
            "os.environ",
            {"OPENAI_API_KEY": "test-key"}
        ),
    ):
        result = create_score_analyst()

        mock_model.assert_called_once()
        mock_agent.assert_called_once()
        assert result is mock_agent.return_value


def test_openai_model_configuration():
    with (
        patch(
            "nps_insights_agent.agents.score_analyst.OpenAIModel"
        ) as mock_model,
        patch(
            "nps_insights_agent.agents.score_analyst.Agent"
        ),
        patch.dict(
            "os.environ",
            {
                "OPENAI_API_KEY": "test-key",
                "OPENAI_BASE_URL": "https://api.openai.com/v1",
                "OPENAI_MODEL": "gpt-6-luna",
            },
        ),
    ):
        create_score_analyst()

        kwargs = mock_model.call_args.kwargs

        assert kwargs["model_id"] == "gpt-6-luna"
        assert kwargs["client_args"]["api_key"] == "test-key"
        assert (
            kwargs["client_args"]["base_url"]
            == "https://api.openai.com/v1"
        )


def test_agent_has_correct_tools():
    with (
        patch(
            "nps_insights_agent.agents.score_analyst.OpenAIModel"
        ),
        patch(
            "nps_insights_agent.agents.score_analyst.Agent"
        ) as mock_agent,
        patch.dict(
            "os.environ",
            {"OPENAI_API_KEY": "test-key"},
        ),
    ):
        create_score_analyst()

        kwargs = mock_agent.call_args.kwargs
        tool_names = {
            tool.tool_name
            for tool in kwargs["tools"]
        }

        assert tool_names == {
            "get_nps_overview",
            "get_nps_by_segment",
        }


def test_score_analyst_prompt():
    prompt = SCORE_ANALYST_SYSTEM_PROMPT.lower()

    assert "nps" in prompt
    assert "30" in prompt
    assert "invent" in prompt
    assert "tools" in prompt
