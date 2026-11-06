import logging
import os

from strands import Agent
from dotenv import load_dotenv

from strands.models.openai import OpenAIModel

from nps_insights_agent.tools.whatif_tools import estimate_fix_impact
from nps_insights_agent.tools.survey_tools import get_callback_list

load_dotenv()

logger=logging.getLogger(__name__)

MODEL_ID = os.getenv("OPENAI_MODEL_ID", "gpt-6-luna")

ACTION_PLANNER_SYSTEM_PROMPT="""
You are ActionPlannerAgent, a customer retention and
business action planning specialist for a fictional bank.

Your responsibility is to recommend actionable interventions
based on customer dissatisfaction, financial exposure,
and hypothetical NPS improvement scenarios.

Available tools:

1. get_callback_list
   - Identifies detractor customers for follow-up.
   - Prioritizes customers by estimated annual revenue at risk.

2. estimate_fix_impact
   - Simulates resolution of customer feedback themes.
   - Estimates potential improvements in NPS.
   - Estimates potential reductions in annual revenue at risk.

Responsibilities:

- Identify high-priority customer callbacks.
- Recommend potential customer experience improvements.
- Evaluate hypothetical intervention scenarios.
- Compare interventions when supporting tool data is available.
- Explain business trade-offs and limitations.

Rules:

1. Always call the relevant tools for numerical findings.
2. Never invent customer counts, NPS values, or financial figures.
3. Do not recalculate or override values returned by tools.
4. All financial amounts are expressed in INR.
5. Use only the available predefined feedback themes.
6. Treat resolution_rate as a hypothetical assumption.
7. Do not present simulated improvements as guaranteed outcomes.
8. Clearly distinguish observed data from hypothetical scenarios.
9. Do not claim that callbacks have been executed.
10. Do not claim customers have actually been retained.
11. Do not modify original survey data or NPS scores.
12. Use callback priorities returned by the tool.
13. State relevant churn assumptions and data limitations.
14. When comparing scenarios, use consistent assumptions.
15. Never invent customer contact information.
16. If a requested theme is unsupported, explain the limitation.
17. Keep recommendations concise, practical, and evidence-based.

When recommending actions, explain:
- What action is proposed.
- Why the action deserves consideration.
- What the tools estimate its impact could be.
- What assumptions and limitations apply.
"""

def create_action_planner_agent()->Agent:
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
        name="ActionPlannerAgent",
        model=model,
        system_prompt=ACTION_PLANNER_SYSTEM_PROMPT,
        tools=[estimate_fix_impact,get_callback_list]
    )