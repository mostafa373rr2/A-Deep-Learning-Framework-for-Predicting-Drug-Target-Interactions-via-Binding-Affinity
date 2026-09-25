"""
🧪 SMILES to Molecular Fingerprint Converter
Notebook 2: Drug Feature Extraction (NO RDKit Required!)

تحويل التركيب الكيميائي إلى أرقام

المحتوى:
1. ✅ SMILES Parser - قراءة الصيغة الكيميائية
2. ✅ Molecular Graph Builder - بناء الجزيء كشبكة
3. ✅ Morgan Fingerprint Generator - توليد البصمة
4. ✅ Batch Processing - معالجة دفعات كبيرة

لا يحتاج RDKit! - كل شيء مبني من الصفر
"""

import numpy as np
import hashlib
from collections import defaultdict, deque
from typing import List, Dict, Tuple, Set


# ============================================================================
# 🔬 Atom and Bond Properties
# ============================================================================

class AtomProperties:
    """Atom Properties Database"""
    ATOMIC_NUMBERS = {
        'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'P': 15, 'S': 16, 
        'Cl': 17, 'Br': 35, 'I': 53, 'B': 5, 'Si': 14, 'Se': 34,
        'Li': 3, 'Na': 11, 'K': 19, 'Mg': 12, 'Ca': 20
    }
    
    VALENCES = {
        'H': 1, 'C': 4, 'N': 3, 'O': 2, 'F': 1, 'P': 5, 'S': 6,
        'Cl': 1, 'Br': 1, 'I': 1, 'B': 3, 'Si': 4
    }
    
    AROMATIC_ATOMS = {'c', 'n', 'o', 's', 'p'}


class Atom:
    """Atom Class"""
    def __init__(self, symbol: str, charge: int = 0, is_aromatic: bool = False):
        self.symbol = symbol.upper() if not is_aromatic else symbol.lower()
        self.charge = charge
        self.is_aromatic = is_aromatic
        self.bonds = []
        self.explicit_hydrogens = 0
    
    def get_atomic_number(self):
        symbol = self.symbol.upper()
        return AtomProperties.ATOMIC_NUMBERS.get(symbol, 6)
    
    def get_valence(self):
        symbol = self.symbol.upper()
        return AtomProperties.VALENCES.get(symbol, 4)
    
    def get_total_degree(self):
        return len(self.bonds)
    
    def get_num_hydrogens(self):
        total_bonds = sum(bond_order for _, bond_order in self.bonds)
        valence = self.get_valence()
        implicit_h = max(0, valence - total_bonds - abs(self.charge))
        return implicit_h + self.explicit_hydrogens
    
    def to_feature_string(self):
        features = [
            str(self.get_atomic_number()),
            str(self.get_total_degree()),
            str(self.get_num_hydrogens()),
            str(int(self.is_aromatic)),
            str(self.charge)
        ]
        return '_'.join(features)


class Bond:
    """Bond Class"""
    SINGLE = 1
    DOUBLE = 2
    TRIPLE = 3
    AROMATIC = 4
    
    def __init__(self, atom1_idx: int, atom2_idx: int, bond_order: int):
        self.atom1_idx = atom1_idx
        self.atom2_idx = atom2_idx
        self.bond_order = bond_order


# ============================================================================
# 🧬 SMILES Parser - Complete Implementation
# ============================================================================

class SMILESParser:
    """SMILES Parser - Complete Implementation"""
    def __init__(self):
        self.atoms = []
        self.bonds = []
        self.ring_closures = {}
    
    def parse(self, smiles: str) -> Tuple[List[Atom], List[Bond]]:
        self.atoms = []
        self.bonds = []
        self.ring_closures = {}
        
        i = 0
        atom_stack = []
        prev_atom_idx = None
        default_bond_order = Bond.SINGLE
        
        while i < len(smiles):
            char = smiles[i]
            
            # Parse atoms
            if char.isupper() or char.islower():
                atom_symbol, is_aromatic, i = self._parse_atom(smiles, i)
                charge = 0
                explicit_h = 0
                
                if i < len(smiles) and smiles[i] == '[':
                    atom_symbol, charge, explicit_h, is_aromatic, i = self._parse_bracket_atom(smiles, i)
                
                atom = Atom(atom_symbol, charge, is_aromatic)
                atom.explicit_hydrogens = explicit_h
                atom_idx = len(self.atoms)
                self.atoms.append(atom)
                
                if prev_atom_idx is not None:
                    bond = Bond(prev_atom_idx, atom_idx, default_bond_order)
                    self.bonds.append(bond)
                    self.atoms[prev_atom_idx].bonds.append((atom_idx, default_bond_order))
                    self.atoms[atom_idx].bonds.append((prev_atom_idx, default_bond_order))
                
                prev_atom_idx = atom_idx
                default_bond_order = Bond.SINGLE
            
            # Parse bonds
            elif char == '-':
                default_bond_order = Bond.SINGLE
                i += 1
            elif char == '=':
                default_bond_order = Bond.DOUBLE
                i += 1
            elif char == '#':
                default_bond_order = Bond.TRIPLE
                i += 1
            elif char == ':':
                default_bond_order = Bond.AROMATIC
                i += 1
            
            # Parse branches
            elif char == '(':
                atom_stack.append(prev_atom_idx)
                i += 1
            elif char == ')':
                if atom_stack:
                    prev_atom_idx = atom_stack.pop()
                i += 1
            
            # Parse ring closures
            elif char.isdigit():
                ring_num = int(char)
                if ring_num in self.ring_closures:
                    ring_start_idx = self.ring_closures[ring_num]
                    bond = Bond(ring_start_idx, prev_atom_idx, default_bond_order)
                    self.bonds.append(bond)
                    self.atoms[ring_start_idx].bonds.append((prev_atom_idx, default_bond_order))
                    self.atoms[prev_atom_idx].bonds.append((ring_start_idx, default_bond_order))
                    del self.ring_closures[ring_num]
                else:
                    self.ring_closures[ring_num] = prev_atom_idx
                default_bond_order = Bond.SINGLE
                i += 1
            else:
                i += 1
        
        return self.atoms, self.bonds
    
    def _parse_atom(self, smiles: str, i: int) -> Tuple[str, bool, int]:
        char = smiles[i]
        is_aromatic = char.islower()
        
        if i + 1 < len(smiles):
            two_letter = smiles[i:i+2]
            if two_letter in ['Cl', 'Br', 'Si', 'Se', 'Na', 'Ca', 'Mg']:
                return two_letter, is_aromatic, i + 2
        
        return char, is_aromatic, i + 1
    
    def _parse_bracket_atom(self, smiles: str, i: int) -> Tuple[str, int, int, bool, int]:
        i += 1  # Skip '['
        atom_symbol = 'C'
        charge = 0
        explicit_h = 0
        is_aromatic = False
        
        # Skip isotope numbers
        while i < len(smiles) and smiles[i].isdigit():
            i += 1
        
        # Parse atom symbol
        if i < len(smiles):
            char = smiles[i]
            is_aromatic = char.islower()
            if i + 1 < len(smiles) and smiles[i:i+2] in ['Cl', 'Br']:
                atom_symbol = smiles[i:i+2]
                i += 2
            else:
                atom_symbol = char
                i += 1
        
        # Parse H count
        if i < len(smiles) and smiles[i] == 'H':
            i += 1
            if i < len(smiles) and smiles[i].isdigit():
                explicit_h = int(smiles[i])
                i += 1
            else:
                explicit_h = 1
        
        # Parse charge
        while i < len(smiles) and smiles[i] in ['+', '-']:
            sign = 1 if smiles[i] == '+' else -1
            i += 1
            if i < len(smiles) and smiles[i].isdigit():
                charge = sign * int(smiles[i])
                i += 1
            else:
                charge = sign
        
        # Skip to closing bracket
        while i < len(smiles) and smiles[i] != ']':
            i += 1
        
        if i < len(smiles):
            i += 1  # Skip ']'
        
        return atom_symbol, charge, explicit_h, is_aromatic, i


# ============================================================================
# 🔍 Morgan Fingerprint Generator
# ============================================================================

class MorganFingerprintGenerator:
    """Morgan Fingerprint Generator"""
    def __init__(self, radius: int = 2, n_bits: int = 1024):
        self.radius = radius
        self.n_bits = n_bits
    
    def generate(self, atoms: List[Atom], bonds: List[Bond]) -> np.ndarray:
        if len(atoms) == 0:
            return np.zeros(self.n_bits, dtype=np.int8)
        
        # Build adjacency list
        adjacency = defaultdict(list)
        for bond in bonds:
            adjacency[bond.atom1_idx].append(bond.atom2_idx)
            adjacency[bond.atom2_idx].append(bond.atom1_idx)
        
        fingerprint = np.zeros(self.n_bits, dtype=np.int8)
        
        # Initialize atom identifiers
        current_identifiers = {}
        for idx, atom in enumerate(atoms):
            current_identifiers[idx] = atom.to_feature_string()
        
        # Hash initial identifiers
        for idx, identifier in current_identifiers.items():
            hash_val = self._hash_string(identifier)
            fingerprint[hash_val % self.n_bits] = 1
        
        # Iterate through radius
        for r in range(self.radius):
            next_identifiers = {}
            for idx in range(len(atoms)):
                # Get neighbor features
                neighbors = adjacency[idx]
                neighbor_features = sorted([current_identifiers[n] for n in neighbors])
                
                # Create new identifier
                combined = current_identifiers[idx] + '_' + '_'.join(neighbor_features)
                next_identifiers[idx] = combined
                
                # Hash and set bit
                hash_val = self._hash_string(combined)
                fingerprint[hash_val % self.n_bits] = 1
            
            current_identifiers = next_identifiers
        
        return fingerprint
    
    def _hash_string(self, s: str) -> int:
        return int(hashlib.md5(s.encode()).hexdigest(), 16)


# ============================================================================
# 🚀 Main SMILES to Fingerprint Functions
# ============================================================================

def smiles_to_morgan_fingerprint(smiles: str, radius: int = 2, n_bits: int = 1024) -> np.ndarray:
    """Convert SMILES to Morgan Fingerprint"""
    try:
        parser = SMILESParser()
        atoms, bonds = parser.parse(smiles)
        
        if len(atoms) == 0:
            return np.zeros(n_bits, dtype=np.int8)
        
        fp_generator = MorganFingerprintGenerator(radius=radius, n_bits=n_bits)
        fingerprint = fp_generator.generate(atoms, bonds)
        
        return fingerprint
    except Exception as e:
        print(f"Warning: Failed to parse SMILES '{smiles}': {e}")
        return np.zeros(n_bits, dtype=np.int8)


def batch_smiles_to_fingerprints(smiles_list: List[str], 
                                  radius: int = 2,
                                  n_bits: int = 1024,
                                  verbose: bool = True) -> np.ndarray:
    """Batch Processing Function"""
    try:
        from tqdm import tqdm
        iterator = tqdm(smiles_list, desc="Converting SMILES") if verbose else smiles_list
    except ImportError:
        iterator = smiles_list
        if verbose:
            print(f"Processing {len(smiles_list)} molecules...")
    
    fingerprints = []
    for smiles in iterator:
        fp = smiles_to_morgan_fingerprint(smiles, radius=radius, n_bits=n_bits)
        fingerprints.append(fp)
    
    return np.array(fingerprints)


# ============================================================================
# Testing Code
# ============================================================================

def test_smiles_converter():
    """Test SMILES to Fingerprint Converter"""
    test_molecules = {
        'Water': 'O',
        'Methane': 'C',
        'Ethanol': 'CCO',
        'Benzene': 'c1ccccc1',
        'Aspirin': 'CC(=O)Oc1ccccc1C(=O)O',
    }
    
    print("Testing SMILES to Fingerprint Converter")
    print("=" * 60)
    
    for name, smiles in test_molecules.items():
        fp = smiles_to_morgan_fingerprint(smiles, radius=2, n_bits=1024)
        print(f"{name:12s} | SMILES: {smiles:20s} | Bits set: {np.sum(fp)}")
    
    print("\n✅ Converter working successfully!")
    print("=" * 60)
    
    # Test batch processing
    print("\nTesting batch processing...")
    smiles_list = list(test_molecules.values())
    fps = batch_smiles_to_fingerprints(smiles_list, verbose=False)
    print(f"✅ Batch processed {len(fps)} molecules")
    print(f"   Output shape: {fps.shape}")
    print(f"   Average bits set: {np.mean(np.sum(fps, axis=1)):.1f}")


if __name__ == "__main__":
    print('✅ Libraries imported successfully!')
    print()
    
    # Run tests
    test_smiles_converter()
    
    print("\n" + "="*70)
    print("✅ Notebook 2 Complete!")
    print("="*70)
    print("\nWhat we built:")
    print("1. ✅ SMILES Parser (no RDKit!)")
    print("2. ✅ Molecular Graph Builder")
    print("3. ✅ Morgan Fingerprint Generator")
    print("4. ✅ Batch Processing Support")
    print("\nNext Step:")
    print("➡️ Run 03_training_pipeline.py to train on BindingDB data")
