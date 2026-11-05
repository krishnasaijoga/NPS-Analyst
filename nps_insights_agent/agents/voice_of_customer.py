import logging
import os

from strands import tool, Agent
from dotenv import load_dotenv

from strands.models.openai import OpenAIModel

from nps_insights_agent.tools.theme_tools import get_comment_themes

load_dotenv()

logger=logging.getLogger(__name__)

MODEL_ID = os.getenv("OPENAI_MODEL_ID", "gpt-6-luna")

CUSTOMER_VOICE_ANALYST_SYSTEM_PROMPT="""
You are VoiceOfCustomerAgent, a customer feedback analyst
for a fictional banking organization.

Your responsibility is to understand and explain recurring
themes in NPS survey comments.

You have access to:
- get_comment_themes: Returns keyword-based feedback themes,
  comment counts, percentages, and representative examples.

Rules:
1. Always use get_comment_themes for factual analysis.
2. Never invent theme counts, percentages, or customer quotes.
3. Use only examples supplied by the tool.
4. Distinguish recurring complaints from positive feedback.
5. Do not assume that mentioning a theme implies negative sentiment.
6. Explain that keyword-based themes can overlap.
7. Do not add percentages across themes as if they are exclusive.
8. Highlight missing or unmatched feedback where relevant.
9. Do not calculate NPS, churn risk, or financial impact.
10. Keep the analysis concise, clear, and business-friendly.
"""

def create_voice_of_customer()->Agent:
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

    print(f"Using {MODEL_ID} on OpenAI")
    logger.info("Initializing CustomerVoiceAnalystAgent with model: %s", MODEL_ID)

    return Agent(
        name="CustomerVoiceAnalystAgent",
        model=model,
        # callback_handler=None,
        system_prompt=CUSTOMER_VOICE_ANALYST_SYSTEM_PROMPT,
        tools=[get_comment_themes],
    )