"""
Canonical graph featurisation for the SAIR IC50 oracle.

INVARIANT: mol_to_graph, ATOM_FEATURE_DIM, and BOND_FEATURE_DIM are the single
source of truth shared with the GFlowNet repo. Never change these without
updating both repos simultaneously.
"""

from __future__ import annotations

from typing import Optional

import torch
from rdkit import Chem
from torch_geometric.data import Data


# ── Constants ──────────────────────────────────────────────────────────────────

ATOM_SYMBOLS = ["C", "N", "O", "S", "F", "P", "Cl", "Br", "I", "other"]
# 10 symbol one-hot + 1 formal charge + 1 H count + 1 aromaticity + 1 mass/100
ATOM_FEATURE_DIM = 14
BOND_FEATURE_DIM = 3


# ── Atom features ──────────────────────────────────────────────────────────────

def one_hot(value, choices: list) -> list[int]:
    return [int(value == c) for c in choices]


def atom_features(atom: Chem.Atom) -> list[float]:
    """
    14-dimensional atom feature vector.
    Index breakdown:
        0-9:  one-hot over ATOM_SYMBOLS (C, N, O, S, F, P, Cl, Br, I, other)
        10:   formal charge (int, typically -2 to +2)
        11:   total H count (int)
        12:   is aromatic (0 or 1)
        13:   atomic mass / 100.0 (float, normalised)
    """
    symbol = atom.GetSymbol()
    return (
        one_hot(symbol if symbol in ATOM_SYMBOLS[:-1] else "other", ATOM_SYMBOLS)
        + [atom.GetFormalCharge()]
        + [atom.GetTotalNumHs()]
        + [float(atom.GetIsAromatic())]
        + [atom.GetMass() / 100.0]
    )


# ── Bond features ──────────────────────────────────────────────────────────────

def bond_features(bond: Chem.Bond) -> list[float]:
    """
    3-dimensional bond feature vector.
    Index breakdown:
        0: bond order / 3.0  (single=0.33, double=0.67, triple=1.0, aromatic=0.5)
        1: is conjugated (0 or 1)
        2: is in ring (0 or 1)
    """
    return [
        bond.GetBondTypeAsDouble() / 3.0,
        float(bond.GetIsConjugated()),
        float(bond.IsInRing()),
    ]


# ── Core graph construction ────────────────────────────────────────────────────

def mol_to_graph(mol: Chem.Mol) -> Optional[Data]:
    """
    Convert an RDKit Mol object to a PyG Data object.
    Returns None if mol is None or has no real atoms.

    Dummy atoms (atomic number 0, from BRICS decomposition) are SKIPPED so
    the GFlowNet can call this on partial molecules with * attachment points.

    Args:
        mol: RDKit Mol object. Must already be sanitized.

    Returns:
        PyG Data with:
            x:          FloatTensor [N_real_atoms, ATOM_FEATURE_DIM]
            edge_index: LongTensor  [2, 2 * N_bonds]   (undirected)
            edge_attr:  FloatTensor [2 * N_bonds, BOND_FEATURE_DIM]
    """
    if mol is None:
        return None

    real_atom_indices = [
        a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() != 0
    ]
    if len(real_atom_indices) == 0:
        return None

    idx_remap = {old: new for new, old in enumerate(real_atom_indices)}

    x = torch.tensor(
        [atom_features(mol.GetAtomWithIdx(i)) for i in real_atom_indices],
        dtype=torch.float,
    )

    edge_index, edge_attr = [], []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        if i not in idx_remap or j not in idx_remap:
            continue
        feat = bond_features(bond)
        edge_index += [[idx_remap[i], idx_remap[j]], [idx_remap[j], idx_remap[i]]]
        edge_attr += [feat, feat]

    if len(edge_index) == 0:
        # Single real atom, no bonds: self-loop placeholder
        edge_index = [[0, 0]]
        edge_attr = [[0.0, 0.0, 0.0]]

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def smiles_to_graph(smiles: str) -> Optional[Data]:
    """Convenience wrapper for training on SMILES strings from the parquet."""
    mol = Chem.MolFromSmiles(smiles)
    return mol_to_graph(mol)


def fallback_graph() -> Data:
    """Return a dummy single-atom graph for error recovery."""
    return Data(
        x=torch.zeros(1, ATOM_FEATURE_DIM),
        edge_index=torch.zeros(2, 1, dtype=torch.long),
        edge_attr=torch.zeros(1, BOND_FEATURE_DIM),
    )
