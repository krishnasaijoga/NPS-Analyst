from pydantic import BaseModel,Field, model_validator
from typing import Literal

class NPSMetrics(BaseModel):
    total_responses:int=Field(ge=0)
    nps:float=Field(ge=-100,le=100)
    promoters:int=Field(ge=0)
    passives:int=Field(ge=0)
    detractors:int=Field(ge=0)
    promoter_percentage:float=Field(ge=0,le=100)
    passive_percentage:float=Field(ge=0,le=100)
    detractor_percentage:float=Field(ge=0,le=100)
    margin_of_error:float=Field(ge=0)


class MonthlyNPS(BaseModel):
    month:str
    metrics:NPSMetrics


class SegmentNPS(BaseModel):
    segment_name:str
    low_sample:bool
    metrics:NPSMetrics


class NPSFindings(BaseModel):
    finding:str
    evidence:str
    implications:str


class ScoreAnalysisOutput(BaseModel):
    data_status:bool
    overall:NPSMetrics|None=None
    monthly_trend:list[MonthlyNPS]=Field(default_factory=list)
    segment_dimension:Literal["product","channel","tenure"]|None=None
    month:str|None=None
    segments:list[SegmentNPS]=Field(default_factory=list)
    key_findings:list[NPSFindings]=Field(default_factory=list)
    limitations:list[str]=Field(default_factory=list)


class ThemeFindings(BaseModel):
    finding:str
    evidence:str
    implications:str
    supporting_themes:list[str]=Field(default_factory=list)


class ThemeSummary(BaseModel):
    theme:str
    count:int=Field(ge=0)
    percentage:float=Field(ge=0.0,le=100.0)
    example_comments:list[str]=Field(default_factory=list)


class VoiceOfCustomerOutput(BaseModel):
    data_status:bool
    month:str|None=None
    nps_category:Literal["promoter","passive","detractor"]|None=None
    total_comments_analyzed: int=Field(ge=0)
    matched_comments: int=Field(ge=0)
    unmatched_comments: int=Field(ge=0)
    themes:list[ThemeSummary]=Field(default_factory=list)
    key_findings:list[ThemeFindings]=Field(default_factory=list)
    limitations:list[str]=Field(default_factory=list)

    @model_validator(mode="after")
    def validate_comment_counts(self):
        if (self.matched_comments+self.unmatched_comments != self.total_comments_analyzed):
            raise ValueError("Matched and unmatched comments must equal total comments analyzed.")
        return self


class RevenueRiskMetrics(BaseModel):
    customers:int=Field(ge=0)
    annual_revenue:float=Field(ge=0)
    revenue_at_risk:float=Field(ge=0)
    risk_percentage:float=Field(ge=0,le=100)
    @model_validator(mode="after")
    def validate_revenue_risk(self):
        if self.revenue_at_risk > self.annual_revenue:
            raise ValueError(
                "Revenue at risk cannot exceed annual revenue"
            )
        return self


class CategoryRiskMetrics(BaseModel):
    nps_category:Literal["promoter","passive","detractor"]
    metrics:RevenueRiskMetrics


class ProductRiskMetrics(BaseModel):
    product:Literal["Credit Card","Wealth","Savings Account"]
    metrics:RevenueRiskMetrics

class ChannelRiskMetrics(BaseModel):
    channel:Literal["Branch","Mobile App","Call Centre"]
    metrics:RevenueRiskMetrics


class TenureRiskMetrics(BaseModel):
    tenure:Literal["<1 yr","1-3 yrs","3+ yrs"]
    metrics:RevenueRiskMetrics


class RiskFindings(BaseModel):
    findings:str
    evidence:str
    implications:str


class ScoreCommentMismatch(BaseModel):
    response_id:str
    score:int=Field(ge=0,le=10)
    comment:str
    sentiment:Literal["positive","negative","neutral","mixed"]
    sentiment_reason:str
    mismatch_reason:str


class MismatchAnalysis(BaseModel):
    total_comments_analyzed:int=Field(ge=0)
    total_mismatches:int=Field(ge=0)
    uncertain_comments:int=Field(ge=0)
    mismatches:list[ScoreCommentMismatch]=Field(default_factory=list)
    @model_validator(mode="after")
    def validate_mismatch_counts(self):
        if self.total_mismatches > self.total_comments_analyzed:
            raise ValueError(
                "Mismatches cannot exceed analyzed comments"
            )

        if self.total_mismatches != len(self.mismatches):
            raise ValueError(
                "Mismatch count must equal mismatch list length"
            )

        return self


class ChurnAssumptions(BaseModel):
    promoter:float=Field(ge=0,le=1)
    passive:float=Field(ge=0,le=1)
    detractor:float=Field(ge=0,le=1)


class ChurnRiskAgentOutput(BaseModel):
    data_status:bool
    currency:Literal["INR"]="INR"
    unit:Literal["rupees"]="rupees"
    churn_assumptions:ChurnAssumptions
    overall:RevenueRiskMetrics|None =None
    risk_by_category:list[CategoryRiskMetrics]=Field(default_factory=list)
    risk_by_product:list[ProductRiskMetrics]=Field(default_factory=list)
    risk_by_channel:list[ChannelRiskMetrics]=Field(default_factory=list)
    risk_by_tenure:list[TenureRiskMetrics]=Field(default_factory=list)
    data_caveats:list[str]=Field(default_factory=list)
    risk_findings:list[RiskFindings]=Field(default_factory=list)
    mismatch_analysis:MismatchAnalysis|None=None


class ChurnProbabilities(BaseModel):
    promoter:float=Field(default=0.03,ge=0,le=1)
    passive:float=Field(default=0.1,ge=0,le=1)
    detractor:float=Field(default=0.25,ge=0,le=1)


class ProposedAction(BaseModel):
    action:str
    reason:str
    supporting_evidence:str


class CallbackCandidate(BaseModel):
    response_id:str
    nps_score:int=Field(ge=0,le=6)
    product:str
    annual_revenue_inr:float=Field(ge=0)
    revenue_at_risk_inr:float=Field(ge=0)
    comment:str|None=None
    priority:int=Field(ge=1)


class CallbackRecommendations(BaseModel):
    total_candidates:int=Field(ge=0)
    returned_candidates:int=Field(ge=0)
    callback_list:list[CallbackCandidate]=Field(default_factory=list)
    @model_validator(mode="after")
    def validate_callback_count(self):
        if self.returned_candidates>self.total_candidates:
            raise ValueError("Returned candidates cannot exceed total candidates")
        if self.returned_candidates!=len(self.callback_list):
            raise ValueError("Returned candidates must equal callback list lenght")
        return self


class SimulatedTransition(BaseModel):
    detractor_to_passive:int=Field(ge=0)
    passive_to_promoter:int=Field(ge=0)


class NPSImpact(BaseModel):
    baseline_nps:float=Field(ge=-100,le=100)
    simulated_nps:float=Field(ge=-100,le=100)
    nps_improvement_points:float=Field(ge=0)
    @model_validator(mode="after")
    def validate_nps_impact(self):
        expected_improvement=self.simulated_nps-self.baseline_nps
        if abs(expected_improvement-self.nps_improvement_points)>0.02:
            raise ValueError("NPS improvement is inconsistent with baseline and simulated NPS")
        return self


class RevenueImpact(BaseModel):
    currency:Literal["INR"]="INR"
    unit:Literal["rupees"]="rupees"
    baseline_revenue_at_risk:float=Field(ge=0)
    simulated_revenue_at_risk:float=Field(ge=0)
    estimated_risk_reduction:float=Field(ge=0)
    @model_validator(mode="after")
    def validate_revenue_impact(self):
        expected_reduction = (
            self.baseline_revenue_at_risk
            - self.simulated_revenue_at_risk
        )
        if abs(expected_reduction - self.estimated_risk_reduction) > 0.02:
            raise ValueError(
                "Revenue risk reduction is inconsistent with financial values"
            )
        return self


class AssumptionsOutput(BaseModel):
    resolution_rate: float = Field(ge=0, le=1)
    recovery_rule:str
    selection_rule:str
    rounding_rule:str
    churn_probabilities:ChurnProbabilities


class FixImpactScenario(BaseModel):
    theme:str
    resolution_rate:float=Field(ge=0,le=1)
    total_responses:int=Field(ge=0)
    affected_responses:int=Field(ge=0)
    simulated_transitions:SimulatedTransition
    nps_impact:NPSImpact
    revenue_impact:RevenueImpact
    assumptions:AssumptionsOutput
    @model_validator(mode="after")
    def validate_resolution_rate_consistency(self):
        if self.resolution_rate != self.assumptions.resolution_rate:
            raise ValueError(
                "Scenario resolution_rate must match assumptions resolution_rate"
            )

        return self


class ActionPlannerOutput(BaseModel):
    data_status:bool
    proposed_actions:list[ProposedAction]=Field(default_factory=list)
    callback_recommendations:CallbackRecommendations|None=None
    simulated_scenarios:list[FixImpactScenario]=Field(default_factory=list)
    caveats:list[str]=Field(default_factory=list)
    limitations:list[str]=Field(default_factory=list)