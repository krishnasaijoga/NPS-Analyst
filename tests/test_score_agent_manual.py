
from nps_insights_agent.agents.score_analyst import (
    create_score_analyst,
)

from nps_insights_agent.schemas.agent_outputs import (
    ScoreAnalysisOutput,
)


agent = create_score_analyst()

result = agent(
    """
    Analyze the banking survey data using your tools.

    1. Retrieve the overall NPS and monthly trend.
    2. Analyze NPS by product.
    3. Identify important findings supported by the data.
    4. Highlight sample-size limitations.

    Return all results according to ScoreAnalysisOutput.
    Do not invent metrics or fill missing values with estimates.
    """
)

output = result.structured_output

assert isinstance(output, ScoreAnalysisOutput)

print("\n===== STRUCTURED SCORE ANALYSIS =====")
print(output.model_dump_json(indent=2))
