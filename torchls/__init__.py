"""Batched block-sparse nonlinear least squares, implemented in Python."""

from .batching import AutoBatchSize
from .cost import Cost
from .geometry import (
    SE2Var,
    SE3Var,
    Sim3Var,
    SO3Var,
    se3_exp,
    se3_retract_jvp,
    sim3_exp,
    sim3_retract_jvp,
    so3_exp,
    so3_retract_jvp,
)
from .linear_system import FixedLinearSystem, LADResult, LinearFactor, least_absolute_deviations
from .losses import HuberLoss, Loss
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
    "AutoBatchSize",
    "BlockJacobian",
    "ConjugateGradientConfig",
    "Cost",
    "FixedLinearSystem",
    "HuberLoss",
    "LADResult",
    "LeastSquaresProblem",
    "LinearFactor",
    "Linearization",
    "Loss",
    "PCGResult",
    "Problem",
    "SE2Var",
    "SE3Var",
    "SO3Var",
    "Sim3Var",
    "SolveSummary",
    "TerminationConfig",
    "TrustRegionConfig",
    "Var",
    "VarValues",
    "block_jacobi",
    "least_absolute_deviations",
    "pcg",
    "se3_exp",
    "se3_retract_jvp",
    "sim3_exp",
    "sim3_retract_jvp",
    "so3_exp",
    "so3_retract_jvp",
]
__version__ = "0.1.0"
