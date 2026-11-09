
from nps_insights_agent.agents.score_analyst import (
    create_score_analyst,
)


agent = create_score_analyst()

response = agent(
    "What is the overall NPS and how did it change "
    "across July, August and September?"
)

print("\n===== SCORE ANALYST RESPONSE =====")
print(response)


from nps_insights_agent.agents.score_analyst import create_score_analyst
from nps_insights_agent.schemas.agent_outputs import ScoreAnalysisOutput

agent = create_score_analyst()

result = agent(
    """
    Analyze the overall NPS performance.

    Use the available tools to retrieve:
    - Overall NPS and response distribution
    - Monthly NPS trends
    - Product-level NPS performance

    Provide evidence-based findings and limitations.
    Do not invent numerical values.
    """,
    structured_output_model=ScoreAnalysisOutput,
)

output = result.structured_output

assert output is not None
print(output.model_dump_json(indent=2))