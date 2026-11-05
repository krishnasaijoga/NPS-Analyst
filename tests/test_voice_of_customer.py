
from unittest.mock import patch

from nps_insights_agent.agents.voice_of_customer import (
    create_voice_of_customer,
    CUSTOMER_VOICE_ANALYST_SYSTEM_PROMPT,
)


def test_agent_creation():
    with (
        patch(
            "nps_insights_agent.agents.voice_of_customer.OpenAIModel"
        ) as mock_model,
        patch(
            "nps_insights_agent.agents.voice_of_customer.Agent"
        ) as mock_agent,
    ):
        result = create_voice_of_customer()

        mock_model.assert_called_once()
        mock_agent.assert_called_once()
        assert result is mock_agent.return_value


def test_agent_has_comment_theme_tool():
    with (
        patch(
            "nps_insights_agent.agents.voice_of_customer.OpenAIModel"
        ),
        patch(
            "nps_insights_agent.agents.voice_of_customer.Agent"
        ) as mock_agent,
    ):
        create_voice_of_customer()

        tools = mock_agent.call_args.kwargs["tools"]

        assert len(tools) == 1
        assert tools[0].tool_name == "get_comment_themes"


def test_agent_system_prompt():
    prompt = CUSTOMER_VOICE_ANALYST_SYSTEM_PROMPT.lower()

    assert "theme" in prompt
    assert "invent" in prompt
    assert "keyword" in prompt
    assert "comments" in prompt


def test_agent_uses_groq():
    with (
        patch(
            "nps_insights_agent.agents.voice_of_customer.OpenAIModel"
        ) as mock_model,
        patch(
            "nps_insights_agent.agents.voice_of_customer.Agent"
        ),
    ):
        create_voice_of_customer()

        kwargs = mock_model.call_args.kwargs

        assert kwargs["model_id"] == "gpt-6-luna"
        assert kwargs["client_args"]["base_url"] == (
            "https://api.openai.com/v1"
        )
