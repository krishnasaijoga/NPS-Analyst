from nps_insights_agent.tools.theme_tools import get_comment_themes
from pprint import pprint

result = get_comment_themes(
    month="2026-09",
    nps_category="Detractor"
)

pprint(result)

from nps_insights_agent.tools.survey_tools import classify_sentiment

comments = [
    "Excellent service! I love this bank.",
    "Terrible experience. OTP never arrives.",
    "I visited the branch yesterday.",
    "The app is great, but customer support is terrible.",
    "I am not happy with the service."
]

for comment in comments:
    result = classify_sentiment(comment)

    print(f"\nComment: {comment}")
    print(f"Sentiment: {result.sentiment}")
    print(f"Reason: {result.reason}")