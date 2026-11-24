
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from nps_insights_agent.orchestration.orchestrator import (
    OrchestrationResult,
    run_nps_workflow,
)

from nps_insights_agent.orchestration.checkpoints import (
    CheckpointedAgent,
)

from nps_insights_agent.schemas.agent_outputs import (
    ScoreAnalysisOutput,
    VoiceOfCustomerOutput,
    ChurnRiskAgentOutput,
    ActionPlannerOutput,
)

from nps_insights_agent.agents.score_analyst import (
    create_score_analyst,
)
from nps_insights_agent.agents.voice_of_customer import (
    create_voice_of_customer,
)
from nps_insights_agent.agents.churn_risk import (
    create_churn_risk_agent,
)
from nps_insights_agent.agents.action_planner import (
    create_action_planner_agent,
)


load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = PROJECT_ROOT / ".cache" / "nps_workflow"

# Change this when the dataset, tools, prompts,
# model configuration, or output contracts change.
CHECKPOINT_VERSION = "v1"


def build_checkpointed_agents(refresh=False):
    return {
        "score_agent": CheckpointedAgent(
            name="score",
            agent=create_score_analyst(),
            output_model=ScoreAnalysisOutput,
            checkpoint_dir=CHECKPOINT_DIR,
            version=CHECKPOINT_VERSION,
            refresh=refresh,
        ),

        "voice_agent": CheckpointedAgent(
            name="voice",
            agent=create_voice_of_customer(),
            output_model=VoiceOfCustomerOutput,
            checkpoint_dir=CHECKPOINT_DIR,
            version=CHECKPOINT_VERSION,
            refresh=refresh,
        ),

        "churn_agent": CheckpointedAgent(
            name="churn",
            agent=create_churn_risk_agent(),
            output_model=ChurnRiskAgentOutput,
            checkpoint_dir=CHECKPOINT_DIR,
            version=CHECKPOINT_VERSION,
            refresh=refresh,
        ),

        "action_agent": CheckpointedAgent(
            name="action",
            agent=create_action_planner_agent(),
            output_model=ActionPlannerOutput,
            checkpoint_dir=CHECKPOINT_DIR,
            version=CHECKPOINT_VERSION,
            refresh=refresh,
        ),
    }


@pytest.mark.integration
def test_complete_nps_workflow():

    if os.getenv("RUN_FULL_NPS_LIVE") != "1":
        pytest.skip("Full workflow not enabled")

    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OpenAI API key not configured")

    refresh = os.getenv("REFRESH_NPS_CHECKPOINTS") == "1"

    agents = build_checkpointed_agents(
        refresh=refresh
    )

    result = run_nps_workflow(**agents)

    # Validate overall result
    assert isinstance(result, OrchestrationResult)

    assert result.score_analysis.data_status is True
    assert result.voice_of_customer.data_status is True
    assert result.churn_risk.data_status is True
    assert result.action_plan.data_status is True

    # Validate essential metrics
    assert result.score_analysis.overall is not None
    assert result.churn_risk.overall is not None

    assert result.action_plan.proposed_actions

    # Cross-agent response-count consistency
    assert (
        result.score_analysis.overall.total_responses
        == result.churn_risk.overall.customers
    )

    print("\n===== COMPLETE WORKFLOW PASSED =====")

    print(
        "Overall NPS:",
        result.score_analysis.overall.nps,
    )

    print(
        "Revenue at risk:",
        result.churn_risk.overall.revenue_at_risk,
    )

    print(
        "Proposed actions:",
        len(result.action_plan.proposed_actions),
    )
