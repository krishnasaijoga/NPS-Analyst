from nps_insights_agent.tools.theme_tools import get_comment_themes
from pprint import pprint

result = get_comment_themes(
    month="2026-09",
    nps_category="Detractor"
)

pprint(result)