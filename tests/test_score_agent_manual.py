
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
