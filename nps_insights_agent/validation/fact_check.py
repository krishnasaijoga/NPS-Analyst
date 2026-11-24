from dataclasses import dataclass, field
from math import isclose


@dataclass
class FactCheckIssue:
    field:str
    expected:object
    actual:object
    message:str


@dataclass
class FactCheckResult:
    issues:list[FactCheckIssue]=field(default_factory=list)

    @property
    def passed(self)->bool:
        return len(self.issues)==0

    def add_issue(
            self,
            field:str,
            expected:object,
            actual:object,
            message:str
    ):
        self.issues.append(
            FactCheckIssue(
                field=field,
                expected=expected,
                actual=actual,
                message=message
            )
        )


def check_numeric(
        result:FactCheckResult,
        field:str,
        expected:float,
        actual:float,
        tolerance:float=0.01
):
    if not isclose(
        expected,
        actual,
        rel_tol=0,
        abs_tol=tolerance
    ):
        result.add_issue(
            field=field,
            expected=expected,
            actual=actual,
            message="Numerical mismatch"
        )