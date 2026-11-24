
import os

import pytest
from dotenv import load_dotenv

from nps_insights_agent.schemas.agent_outputs import (
    ActionPlannerOutput,
)

# Adjust the factory import if your function name differs.
from nps_insights_agent.agents.action_planner import (
    create_action_planner_agent,
)

load_dotenv()


@pytest.mark.integration
def test_action_planner_structured_output():

    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OpenAI API key not configured")

    agent = create_action_planner_agent()

    result = agent(
        """
        Prepare an evidence-based customer improvement
        action plan using the available tools.

        1. Call get_callback_list with top_n=3.

        2. Call estimate_fix_impact for:
           theme="OTP & Verification Issues"
           resolution_rate=0.5

        3. Recommend actions supported by the tool results.

        Return the required ActionPlannerOutput structure.

        Important:
        - Include only the 3 returned callback candidates.
        - Include one simulated scenario.
        - Use the exact numerical results from the tools.
        - Revenue figures must be in INR rupees.
        - Treat simulated outcomes as hypothetical,
          not guaranteed improvements.
        - Do not invent customer information.
        - Keep findings concise.
        """
    )

    output = result.structured_output

    # 1. Schema compliance
    assert isinstance(output, ActionPlannerOutput)
    assert output.data_status is True

    # 2. Callback recommendations
    callbacks = output.callback_recommendations

    assert callbacks is not None
    assert callbacks.returned_candidates == 3
    assert len(callbacks.callback_list) == 3

    # 3. Validate callback priority
    priorities = [
        candidate.priority
        for candidate in callbacks.callback_list
    ]

    assert priorities == [1, 2, 3]

    for candidate in callbacks.callback_list:
        assert 0 <= candidate.nps_score <= 6
        assert candidate.annual_revenue_inr >= 0
        assert candidate.revenue_at_risk_inr >= 0

    # 4. Validate simulation
    assert len(output.simulated_scenarios) == 1

    scenario = output.simulated_scenarios[0]

    assert scenario.theme == "OTP & Verification Issues"
    assert scenario.resolution_rate == pytest.approx(0.5)

    # 5. Validate NPS improvement
    impact = scenario.nps_impact

    assert impact.nps_improvement_points == pytest.approx(
        impact.simulated_nps - impact.baseline_nps,
        abs=0.02,
    )

    # 6. Validate financial improvement
    revenue = scenario.revenue_impact

    assert revenue.currency == "INR"
    assert revenue.unit == "rupees"

    assert revenue.estimated_risk_reduction == pytest.approx(
        revenue.baseline_revenue_at_risk
        - revenue.simulated_revenue_at_risk,
        abs=0.02,
    )

    # 7. Validate assumptions
    assumptions = scenario.assumptions

    assert assumptions.resolution_rate == pytest.approx(0.5)
    assert assumptions.churn_probabilities.promoter == pytest.approx(0.03)
    assert assumptions.churn_probabilities.passive == pytest.approx(0.10)
    assert assumptions.churn_probabilities.detractor == pytest.approx(0.25)

    # 8. Validate proposed actions
    assert len(output.proposed_actions) > 0

    for action in output.proposed_actions:
        assert action.action.strip()
        assert action.reason.strip()
        assert action.supporting_evidence.strip()

    print("\n===== ACTION PLANNER STRUCTURED OUTPUT =====")
    print(output.model_dump_json(indent=2))
