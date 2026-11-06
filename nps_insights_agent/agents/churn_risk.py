import logging
import os

from strands import Agent
from dotenv import load_dotenv

from strands.models.openai import OpenAIModel

from nps_insights_agent.tools.survey_tools import get_nps_by_segment, get_revenue_at_risk, find_score_comment_mismatches

load_dotenv()

logger=logging.getLogger(__name__)

MODEL_ID = os.getenv("OPENAI_MODEL_ID", "gpt-6-luna")

CHURN_RISK_SYSTEM_PROMPT="""
You are a ChurnRiskAgent, a customer retention and financial risk analyst for a fictional bank.

Your responsibilities:
- Analyze estimated annual revenue at risk.
- Identify financially exposed customer segments.
- Prioritize detractor customers for follow-up.
- Identify potential score-comment inconsistencies.
- Communicate customer retention risks to management.

Available Tools:
1. get_revenue_at_risk
2. get_nps_by_segment
3. find_score_comment_mismatches

Rules:
1. Always use relevant tools for numerical findings.
2. Never invent customer counts or financial values.
3. Never recalculate or override tool-generated metrics.
4. All financial amounts are expressed in INR.
5. Churn probabilities are assumptions:
    - Promoter: 3%
    - Passive: 10%
    - Detractors: 25%
6. Revenue at risk is a hypothetical expected-loss estimate, not an actual observed financial loss.
7. Treat callback recommendations as prioritzation, not confirmed customer churn.
8. Score-comment mismatches are review flags, not proven survey errors.
9. Flag segments with fewer than 30 responses.
10. Do not recommend low-sample segments as definitively highest or lowest risk based on NPS.
11. Do not alter original survey scores or values.
12. Use only evidence supplied by the tools.
13. Clearly state assumptions and limitations.
14. keep findings concise and business-friendly.

"""

def create_churn_risk_agent()->Agent:
    model=OpenAIModel(
        client_args={
            "api_key": os.environ["OPENAI_API_KEY"],
            "base_url": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),  # or a local OpenAI-compatible server
            "max_retries": 5,  # retry dropped connections (the client default is 2)
        },
        model_id=MODEL_ID,
        params={
                "reasoning_effort":"none",
                "max_completion_tokens":1500
        }
    )

    print(f"Using {MODEL_ID} on OpenAI")
    logger.info("Initializing ChurnRiskAgent with model: %s", MODEL_ID)

    return Agent(
        name="ChurnRiskAgent",
        model=model,
        system_prompt=CHURN_RISK_SYSTEM_PROMPT,
        tools=[get_nps_by_segment,get_revenue_at_risk,find_score_comment_mismatches]
    )