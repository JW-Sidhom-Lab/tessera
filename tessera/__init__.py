"""TESSERA: Tumour Embeddings via Self-Supervised Encoding and Reconstruction of Alterations.

A foundation model for the cancer genome, jointly pretrained on somatic SNVs and CNAs.
"""

from tessera.model import TESSERA
from tessera.base import BaseModel

__all__ = ["TESSERA", "BaseModel"]
