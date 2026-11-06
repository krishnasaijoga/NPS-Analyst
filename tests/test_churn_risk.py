import os

from unittest.mock import patch

from nps_insights_agent.agents.churn_risk import (
    create_churn_risk_agent,
    CHURN_RISK_SYSTEM_PROMPT,
)


MODULE = "nps_insights_agent.agents.churn_risk"


def test_agent_creation():
    with (
        patch(f"{MODULE}.OpenAIModel") as mock_model,
        patch(f"{MODULE}.Agent") as mock_agent,
        patch.dict(
            "os.environ",
            {
                "OPENAI_API_KEY": "test-key",
                "OPENAI_MODEL": "gpt-4.1-mini",
            },
        ),
    ):
        result = create_churn_risk_agent()

    mock_model.assert_called_once()
    mock_agent.assert_called_once()
    assert result is mock_agent.return_value


def test_model_configuration():
    with (
        patch(f"{MODULE}.OpenAIModel") as mock_model,
        patch(f"{MODULE}.Agent"),
        patch.dict(
            "os.environ",
            {
                "OPENAI_API_KEY": "test-key",
                "OPENAI_MODEL": "gpt-6-luna",
            },
        ),
    ):
        create_churn_risk_agent()

    kwargs = mock_model.call_args.kwargs

    assert kwargs["model_id"] == os.environ["OPENAI_MODEL_ID"]
    assert kwargs["client_args"]["api_key"] == "test-key"


def test_registered_tools():
    with (
        patch(f"{MODULE}.OpenAIModel"),
        patch(f"{MODULE}.Agent") as mock_agent,
        patch.dict(
            "os.environ",
            {
                "OPENAI_API_KEY": "test-key",
                "OPENAI_MODEL": "gpt-6-luna",
            },
        ),
    ):
        create_churn_risk_agent()

    tools = mock_agent.call_args.kwargs["tools"]

    tool_names = {tool.tool_name for tool in tools}

    assert tool_names == {
        "get_revenue_at_risk",
        "get_nps_by_segment",
        "find_score_comment_mismatches",
    }


def test_system_prompt():
    prompt = CHURN_RISK_SYSTEM_PROMPT.lower()

    assert "revenue at risk" in prompt
    assert "assumption" in prompt
    assert "callback" in prompt
    assert "mismatch" in prompt
    assert "invent" in prompt
    assert "30" in prompt