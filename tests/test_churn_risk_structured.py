
import os

import pytest
from dotenv import load_dotenv

from nps_insights_agent.schemas.agent_outputs import (
    ChurnRiskAgentOutput,
)

# Adjust the factory name if your implementation differs.
from nps_insights_agent.agents.churn_risk import (
    create_churn_risk_agent,
)

load_dotenv()


@pytest.mark.integration
def test_churn_risk_structured_output():

    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OpenAI API key not configured")

    agent = create_churn_risk_agent()

    result = agent(
    """
    Analyze financial churn exposure using your tools.

    Call:
    1. get_revenue_at_risk
    2. get_nps_by_segment
    3. find_score_comment_mismatches

    Return ChurnRiskAgentOutput.

    Output requirements:
    - Include overall revenue-at-risk metrics.
    - Include risk breakdowns by category, product,
      channel and tenure.
    - Include the churn assumptions.
    - Summarize the key financial findings.
    - Include mismatch counts.
    - Include at most 5 representative mismatch records.
    - Keep findings and limitations concise.
    - Preserve financial values in INR rupees.
    - Do not invent any numerical values.
    """
)

    output = result.structured_output

    # 1. Schema validation
    assert isinstance(output, ChurnRiskAgentOutput)
    assert output.data_status is True

    # 2. Currency and units
    assert output.currency == "INR"
    assert output.unit == "rupees"

    # 3. Overall financial exposure
    assert output.overall is not None
    assert output.overall.customers == 600

    assert output.overall.annual_revenue >= 0
    assert output.overall.revenue_at_risk >= 0

    assert (
        output.overall.revenue_at_risk
        <= output.overall.annual_revenue
    )

    # 4. Churn probabilities
    assumptions = output.churn_assumptions

    assert assumptions.promoter == pytest.approx(0.03)
    assert assumptions.passive == pytest.approx(0.10)
    assert assumptions.detractor == pytest.approx(0.25)

    # 5. Segment breakdowns
    assert len(output.risk_by_category) == 3
    assert len(output.risk_by_product) > 0
    assert len(output.risk_by_channel) > 0
    assert len(output.risk_by_tenure) > 0

    # 6. Score-comment mismatch analysis
    assert output.mismatch_analysis is not None

    mismatches = output.mismatch_analysis

    assert mismatches.total_mismatches >= 0
    assert mismatches.uncertain_comments >= 0

    assert (
        len(mismatches.mismatches)
        == mismatches.total_mismatches
    )

    # 7. Inspect full structured response
    print("\n===== CHURN RISK STRUCTURED OUTPUT =====")
    print(output.model_dump_json(indent=2))
