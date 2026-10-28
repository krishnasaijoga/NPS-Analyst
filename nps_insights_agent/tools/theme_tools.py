from strands import tool

from nps_insights_agent.data_loader import load_survey_data



THEMES = {
    "OTP & Verification Issues": [
        "otp", "verification code", "code that never comes",
        "code does not come", "code after"
    ],

    "Login & Account Access": [
        "login", "log in", "logging me out", "locked out",
        "couldn't get into my account", "account access"
    ],

    "App Performance & Stability": [
        "app is slow", "app works but", "app crashes",
        "app freezes", "freezes", "hangs", "glitches",
        "retry payments", "slow during peak hours"
    ],

    "Fees & Unexpected Charges": [
        "fee", "fees", "charges", "charged",
        "penalty", "deducted", "hidden costs",
        "fee structure"
    ],

    "Interest Rates & Returns": [
        "interest rate", "interest rates", "fd rates",
        "earns more", "lower than other banks"
    ],

    "Customer Support Delays": [
        "customer care", "call centre", "waiting time",
        "on hold", "ivr", "complaint still open",
        "call dropped", "never reached a human"
    ],

    "Callback & Follow-up Delays": [
        "callback", "promised", "came a day late"
    ],

    "Rewards & Cashback": [
        "reward points", "cashback", "points expire",
        "points on my card"
    ],

    "Branch & Relationship Manager Experience": [
        "branch team", "branch staff",
        "relationship manager", "my rm",
        "kyc update", "branch"
    ],

    "Digital Banking Convenience": [
        "upi", "mobile recharge", "bill payments",
        "quick", "easy to use", "fully online",
        "seamless", "smooth"
    ],

    "Security & Trust": [
        "feel safe", "safe banking",
        "transfers are fast"
    ],

    "General Satisfaction": [
        "good bank", "happy", "excellent service",
        "average experience", "decent",
        "nothing special", "does the job",
        "not happy", "no complaints", "best bank",
        "overall service"
    ]
}


@tool
def get_comment_themes(month=None,nps_category=None)->dict:
    """
    Retrieve comment themes for a particular month and nps_category

    Args:
        month: YYYY-MM
        nps_category: <Promoter | Passive | Detractor>
    
    Returns:
        Dictionary containing comment themes, count for each, percentage, and 3 examples
    """
    df=load_survey_data().copy()

    if month:
        df=df[df["month"]==month]

    if nps_category:
        nps_category=nps_category.lower()
        if nps_category=="promoter":
            df=df[df["score"]>=9]

        elif nps_category=="passive":
            df=df[df["score"].between(7,8)]

        elif nps_category=="detractor":
            df=df[df["score"]<7]
        else:
            raise ValueError(f"Invalid NPS category: {nps_category}")

    df=df.dropna(subset=["comment"])
    df=df[df["comment"].str.strip()!=""]

    results=[]

    for theme, keywords in THEMES.items():
        matched_comments=[]
        for comment in df["comment"]:
            comment_lower=comment.lower()

            if any(keyword in comment_lower for keyword in keywords):
                matched_comments.append(comment)
        if len(matched_comments)>0:
            results.append({
                "theme":theme,
                "count":len(matched_comments),
                "percentage":round(
                    len(matched_comments)/len(df)*100,2
                ) if len(df)>0 else 0,
                "example_comments":matched_comments[:3]
            })

    matched_comments = 0
    unmatched_comments = 0

    for comment in df["comment"]:
        comment_lower = comment.lower()

        if any(
            keyword in comment_lower
            for keywords in THEMES.values()
            for keyword in keywords
        ):
            matched_comments += 1
        else:
            unmatched_comments += 1

    return {
        "data_status": len(df) > 0,
        "total_comments_analyzed": len(df),
        "matched_comments": matched_comments,
        "unmatched_comments": unmatched_comments,
        "themes": sorted(
            results,
            key=lambda x: x["count"],
            reverse=True
        )
    }