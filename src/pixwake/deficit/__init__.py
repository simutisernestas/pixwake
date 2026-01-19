from .gaussian import (
    BastankhahGaussianDeficit,
    NiayifarGaussianDeficit,
    TurboGaussianDeficit,
)
from .noj import NOJDeficit
from .selfsimilarity import (
    SelfSimilarityBlockageDeficit,
    SelfSimilarityBlockageDeficit2020,
)

__all__ = [
    # Wake deficit models
    "BastankhahGaussianDeficit",
    "NiayifarGaussianDeficit",
    "TurboGaussianDeficit",
    "NOJDeficit",
    # Blockage deficit models
    "SelfSimilarityBlockageDeficit",
    "SelfSimilarityBlockageDeficit2020",
]
