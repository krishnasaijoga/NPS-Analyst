
from nps_insights_agent.agents.churn_risk import (
    create_churn_risk_agent,
)


agent = create_churn_risk_agent()

response = agent(
    "Analyze the overall estimated annual revenue at risk. "
    "Identify which products and customer segments have "
    "the highest financial exposure. "
    "Clearly state the churn probability assumptions."
)

print("\n===== CHURN RISK ANALYSIS =====")
print(response)
