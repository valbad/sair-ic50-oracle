"""
Smoke tests for featurise.py invariants.
Run with: pytest tests/test_featurise.py
"""

import sys
from pathlib import Path

import pytest
import torch
from rdkit import Chem

sys.path.insert(0, str(Path(__file__).parent.parent))

from oracle.featurise import (
    ATOM_FEATURE_DIM,
    BOND_FEATURE_DIM,
    fallback_graph,
    mol_to_graph,
    smiles_to_graph,
)

DIVERSE_SMILES = [
    "c1ccccc1",                          # benzene
    "CC(=O)Oc1ccccc1C(=O)O",            # aspirin
    "CC(N)C(=O)O",                       # alanine (stereocentre)
    "C1CCCCC1",                          # cyclohexane
    "[NH4+]",                            # charged species
    "O=C([O-])c1ccccc1",                 # benzoate anion
    "C1=CC2=CC=CC=C2C=C1",              # naphthalene
    "Brc1ccc(Br)cc1",                    # dibromobenzene
    "O=S(=O)(O)c1ccccc1",               # benzenesulfonic acid
    "ClC(Cl)(Cl)Cl",                     # carbon tetrachloride
]


def test_feature_dimensions_benzene():
    g = smiles_to_graph("c1ccccc1")
    assert g is not None
    assert g.x.shape == (6, ATOM_FEATURE_DIM)
    assert g.edge_attr.shape[1] == BOND_FEATURE_DIM
    assert g.edge_index.shape[0] == 2


def test_smiles_and_mol_produce_identical_output():
    for smi in ["c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O", "CC(N)C(=O)O"]:
        mol = Chem.MolFromSmiles(smi)
        g_from_mol = mol_to_graph(mol)
        g_from_smi = smiles_to_graph(smi)
        assert g_from_mol is not None and g_from_smi is not None
        assert torch.allclose(g_from_mol.x, g_from_smi.x)
        assert torch.equal(g_from_mol.edge_index, g_from_smi.edge_index)
        assert torch.allclose(g_from_mol.edge_attr, g_from_smi.edge_attr)


def test_dummy_atoms_skipped():
    # benzene with one BRICS attachment point (*) - should produce 6 atoms not 7
    mol = Chem.MolFromSmiles("*c1ccccc1")
    assert mol is not None
    g = mol_to_graph(mol)
    assert g is not None
    assert g.x.shape[0] == 6, f"Expected 6 atoms, got {g.x.shape[0]}"


def test_mol_to_graph_none_returns_none():
    assert mol_to_graph(None) is None


def test_invalid_smiles_returns_none():
    assert smiles_to_graph("not_a_smiles_@@##") is None


def test_single_atom_no_crash():
    # Single carbon: no bonds, should get self-loop placeholder
    mol = Chem.MolFromSmiles("[CH4]")
    g = mol_to_graph(mol)
    assert g is not None
    assert g.x.shape == (1, ATOM_FEATURE_DIM)
    assert g.edge_index.shape == (2, 1)
    assert g.edge_attr.shape == (1, BOND_FEATURE_DIM)


def test_fallback_graph_shape():
    g = fallback_graph()
    assert g.x.shape == (1, ATOM_FEATURE_DIM)
    assert g.edge_index.shape == (2, 1)
    assert g.edge_attr.shape == (1, BOND_FEATURE_DIM)


def test_round_trip_diverse_smiles():
    for smi in DIVERSE_SMILES:
        g = smiles_to_graph(smi)
        assert g is not None, f"smiles_to_graph returned None for: {smi}"
        assert g.x.shape[1] == ATOM_FEATURE_DIM, f"Wrong x dim for: {smi}"
        assert g.edge_attr.shape[1] == BOND_FEATURE_DIM, f"Wrong edge_attr dim for: {smi}"
        assert g.edge_index.shape[0] == 2, f"Wrong edge_index shape for: {smi}"
        assert g.x.shape[0] > 0, f"Zero atoms for: {smi}"


def test_atom_feature_dim_constant_is_14():
    assert ATOM_FEATURE_DIM == 14, "ATOM_FEATURE_DIM invariant violated"


def test_bond_feature_dim_constant_is_3():
    assert BOND_FEATURE_DIM == 3, "BOND_FEATURE_DIM invariant violated"


def test_all_dummy_atoms_returns_none():
    # Molecule with only dummy atoms (all atomic number 0)
    mol = Chem.RWMol()
    dummy = Chem.Atom(0)
    mol.AddAtom(dummy)
    mol.AddAtom(dummy)
    assert mol_to_graph(mol.GetMol()) is None


def test_edge_index_is_undirected():
    # Each bond should appear twice (i->j and j->i)
    g = smiles_to_graph("CC")  # ethane: 1 bond -> 2 directed edges
    assert g.edge_index.shape[1] == 2
    assert set(g.edge_index[0].tolist()) == {0, 1}
    assert set(g.edge_index[1].tolist()) == {0, 1}
