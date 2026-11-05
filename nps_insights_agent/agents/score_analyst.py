import logging
import os

from strands import tool, Agent
from dotenv import load_dotenv

from strands.models.openai import OpenAIModel

from nps_insights_agent.tools.survey_tools import get_nps_overview, get_nps_by_segment

load_dotenv()

logger=logging.getLogger(__name__)

MODEL_ID = os.getenv("OPENAI_MODEL_ID", "gpt-6-luna")

SCORE_ANALYST_SYSTEM_PROMPT = """
You are ScoreAnalystAgent, an expert NPS data analyst
for a fictional banking organization.

Responsibilities:
- Analyze overall and monthly NPS performance.
- Compare NPS across products, channels and customer tenure.
- Explain findings using the provided tool outputs.
- Highlight low-sample segments and uncertainty.

Rules:
1. Always use the relevant tools for numerical findings.
2. Never invent NPS scores, sample sizes or percentages.
3. Do not recalculate or override tool-generated values.
4. NPS is expressed in points, not as a percentage.
5. Flag segments with fewer than 30 responses.
6. Do not rank low-sample segments as definitively
   best or worst.
7. If requested information is unavailable, say so.
8. Do not make customer churn or financial predictions.
9. Keep your analysis concise and business-friendly.
"""

def create_score_analyst()->Agent:
    model=OpenAIModel(
        client_args={
                "api_key": os.environ["OPENAI_API_KEY"],
                "base_url": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),  # or a local OpenAI-compatible server
                "max_retries": 5,  # retry dropped connections (the client default is 2)
            },
            model_id=MODEL_ID,
            # Groq reasoning models stream their thinking by default; hide it so only the answer prints.
            params={
                "reasoning_effort":"none",
                "max_completion_tokens":1500
        }
    )

    print(f"Using {MODEL_ID} on OpenaAI")
    logger.info("Initializing ScoreAnalystAgent with model: %s", MODEL_ID)

    return Agent(
        name="ScoreAnalystAgent",
        model=model,
        # callback_handler=None,
        system_prompt=SCORE_ANALYST_SYSTEM_PROMPT,
        tools=[get_nps_overview,get_nps_by_segment],
    )

