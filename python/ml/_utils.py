"""Small shared helpers for the ml/ package."""
import pandas as pd
import torch

from ..config import _cfg_path


def resolve_device(device: str) -> torch.device:
    """Map ``'auto' | 'cpu' | 'cuda'`` to a concrete :class:`torch.device`.

    ``'auto'`` picks ``'cuda'`` if available, else ``'cpu'``.
    """
    if device == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device)


def load_chrom_hmm_lookup() -> dict[str, str]:
    """Load the ChromHMM annotation as a ``"seqnames-start" → state`` dict.

    :return: Mapping from ``"chr{n}-{start}"`` CpG keys to their ChromHMM state label.
    """
    chrom_hmm = pd.read_csv(_cfg_path('chrom_hmm'), usecols=['seqnames', 'start', 'ChromHMM_E059_15'], dtype=str)
    return dict(zip(chrom_hmm['seqnames'] + '-' + chrom_hmm['start'], chrom_hmm['ChromHMM_E059_15']))
