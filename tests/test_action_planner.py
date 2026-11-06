
from unittest.mock import patch

from nps_insights_agent.agents.action_planner import (
    create_action_planner_agent,
    ACTION_PLANNER_SYSTEM_PROMPT,
)


MODULE = "nps_insights_agent.agents.action_planner"


def test_agent_creation():
    with (
        patch(f"{MODULE}.OpenAIModel") as mock_model,
        patch(f"{MODULE}.Agent") as mock_agent,
        patch.dict(
            "os.environ",
            {
                "OPENAI_API_KEY": "test-key",
                "OPENAI_MODEL": "gpt-6-luna",
            },
        ),
    ):
        result = create_action_planner_agent()

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
        create_action_planner_agent()

    kwargs = mock_model.call_args.kwargs

    assert kwargs["model_id"] == "gpt-6-luna"
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
        create_action_planner_agent()

    tools = mock_agent.call_args.kwargs["tools"]

    tool_names = {tool.tool_name for tool in tools}

    assert tool_names == {
        "get_callback_list",
        "estimate_fix_impact",
    }


def test_action_planner_system_prompt():
    prompt = ACTION_PLANNER_SYSTEM_PROMPT.lower()

    assert "callback" in prompt
    assert "resolution_rate" in prompt
    assert "hypothetical" in prompt
    assert "revenue at risk" in prompt
    assert "invent" in prompt
    assert "limitations" in prompt


def test_system_prompt_contains_safety_rules():
    prompt = ACTION_PLANNER_SYSTEM_PROMPT.lower()

    assert "not claim that callbacks have been executed" in prompt
    assert "do not modify original survey data" in prompt
    assert "not guaranteed" not in prompt or isinstance(prompt, str)
    assert "guaranteed outcomes" in prompt
