
import pytest
from types import SimpleNamespace

from nps_insights_agent.orchestration.orchestrator import (
    OrchestrationResult,
    invoke_structured,
    run_nps_workflow,
)

from nps_insights_agent.schemas.agent_outputs import (
    ScoreAnalysisOutput,
    VoiceOfCustomerOutput,
    ChurnRiskAgentOutput,
    ActionPlannerOutput,
)


class FakeAgent:
    def __init__(self, name, output, call_order, error=None):
        self.name = name
        self.output = output
        self.call_order = call_order
        self.error = error
        self.calls = []

    def __call__(self, prompt):
        self.calls.append(prompt)
        self.call_order.append(self.name)

        if self.error is not None:
            raise self.error

        return SimpleNamespace(
            structured_output=self.output
        )


@pytest.fixture
def fake_workflow():
    call_order = []

    # Construct typed objects without performing schema validation.
    # These tests target orchestration, not individual schemas.
    score_output = ScoreAnalysisOutput.model_construct(
        data_status=True
    )
    voice_output = VoiceOfCustomerOutput.model_construct(
        data_status=True
    )
    churn_output = ChurnRiskAgentOutput.model_construct(
        data_status=True
    )
    action_output = ActionPlannerOutput.model_construct(
        data_status=True
    )

    agents = {
        "score": FakeAgent(
            "score", score_output, call_order
        ),
        "voice": FakeAgent(
            "voice", voice_output, call_order
        ),
        "churn": FakeAgent(
            "churn", churn_output, call_order
        ),
        "action": FakeAgent(
            "action", action_output, call_order
        ),
    }

    return agents, call_order


def execute_workflow(agents):
    return run_nps_workflow(
        score_agent=agents["score"],
        voice_agent=agents["voice"],
        churn_agent=agents["churn"],
        action_agent=agents["action"],
    )


def test_all_agents_execute_once(fake_workflow):
    agents, _ = fake_workflow

    execute_workflow(agents)

    for agent in agents.values():
        assert len(agent.calls) == 1


def test_agent_execution_order(fake_workflow):
    agents, call_order = fake_workflow

    execute_workflow(agents)

    assert call_order == [
        "score",
        "voice",
        "churn",
        "action",
    ]


def test_action_planner_receives_context(fake_workflow):
    agents, _ = fake_workflow

    execute_workflow(agents)

    action_prompt = agents["action"].calls[0]

    assert "score_analysis" in action_prompt
    assert "voice_of_customer" in action_prompt
    assert "churn_risk" in action_prompt


def test_orchestration_result_type(fake_workflow):
    agents, _ = fake_workflow

    result = execute_workflow(agents)

    assert isinstance(result, OrchestrationResult)
    assert isinstance(
        result.score_analysis, ScoreAnalysisOutput
    )
    assert isinstance(
        result.voice_of_customer, VoiceOfCustomerOutput
    )
    assert isinstance(
        result.churn_risk, ChurnRiskAgentOutput
    )
    assert isinstance(
        result.action_plan, ActionPlannerOutput
    )


def test_invoke_structured_rejects_invalid_output():
    call_order = []

    agent = FakeAgent(
        name="invalid",
        output={"data_status": True},
        call_order=call_order,
    )

    with pytest.raises(TypeError):
        invoke_structured(
            agent,
            "Analyze NPS",
            ScoreAnalysisOutput,
        )


@pytest.mark.parametrize(
    "failed_agent",
    ["score", "voice", "churn"],
)
def test_specialist_failure_stops_workflow(
    fake_workflow,
    failed_agent,
):
    agents, call_order = fake_workflow

    agents[failed_agent].error = RuntimeError(
        f"{failed_agent} failed"
    )

    with pytest.raises(RuntimeError):
        execute_workflow(agents)

    assert "action" not in call_order


def test_action_planner_failure_propagates(fake_workflow):
    agents, call_order = fake_workflow

    agents["action"].error = RuntimeError(
        "ActionPlanner failed"
    )

    with pytest.raises(RuntimeError):
        execute_workflow(agents)

    assert call_order[-1] == "action"


def test_invalid_specialist_output_prevents_planning(
    fake_workflow,
):
    agents, call_order = fake_workflow

    agents["voice"].output = {
        "data_status": True
    }

    with pytest.raises(TypeError):
        execute_workflow(agents)

    assert "churn" not in call_order
    assert "action" not in call_order
