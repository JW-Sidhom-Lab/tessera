"""TESSERA: Tumour Embeddings via Self-Supervised Encoding and Reconstruction of Alterations.

A foundation model for the cancer genome, jointly pretrained on somatic SNVs and CNAs.
"""

from tessera.model import TESSERA, FeaturizeResult, ReconstructResult
from tessera.base import BaseModel
from tessera.hub import load_pretrained, download_pretrained, featurize, reconstruct
from tessera.data.liftover import lift_snv, lift_cna

__all__ = [
    "TESSERA",
    "BaseModel",
    "FeaturizeResult",
    "ReconstructResult",
    "load_pretrained",
    "download_pretrained",
    "featurize",
    "reconstruct",
    "lift_snv",
    "lift_cna",
]
