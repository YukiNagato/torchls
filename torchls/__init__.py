"""Batched block-sparse nonlinear least squares, implemented in Python."""

from .cost import Cost
from .geometry import SE2Var, SE3Var, se3_exp, so3_exp
from .named import Problem
from .problem import AnalyzedLeastSquaresProblem, LeastSquaresProblem, Linearization
from .solvers import (
    ConjugateGradientConfig,
    PCGResult,
    SolveSummary,
    TerminationConfig,
    TrustRegionConfig,
    block_jacobi,
    pcg,
)
from .sparse import BlockJacobian
from .variables import Var, VarValues

__all__ = [
    "AnalyzedLeastSquaresProblem",
    "BlockJacobian",
    "ConjugateGradientConfig",
    "Cost",
    "LeastSquaresProblem",
    "Linearization",
    "PCGResult",
    "Problem",
    "SE2Var",
    "SE3Var",
    "SolveSummary",
    "TerminationConfig",
    "TrustRegionConfig",
    "Var",
    "VarValues",
    "block_jacobi",
    "pcg",
    "se3_exp",
    "so3_exp",
]
__version__ = "0.1.0"
