import os
import re

import numpy as np
import pandas as pd
from mordred import Calculator, descriptors
from padelpy import padeldescriptor
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.preprocessing import OneHotEncoder
from tqdm import notebook


def generate_descriptors(smiles):
    """
    Generate molecular descriptors using the Mordred package for a list of SMILE strings.
    :param smiles: An iterable collection of SMILE strings.
    :return: A pandas data frame of generated mordred descriptors by SMILE string.
    """
    descriptors_table = np.ndarray((len(smiles), 1826), dtype=object)
    print("Generating mordred descriptors:")
    for index in notebook.tqdm(range(descriptors_table.shape[0])):
        structure = smiles[index]
        mol = Chem.MolFromSmiles(structure)
        if mol is None:
            descriptors_table[index, :] = [None] * 1826
        else:
            AllChem.EmbedMolecule(mol, useExpTorsionAnglePrefs=True, useBasicKnowledge=True)
            descriptors_table[index, :] = Calculator(descriptors, ignore_3D=False)(mol).fill_missing()

    return pd.DataFrame(descriptors_table, columns=Calculator(descriptors, ignore_3D=False).descriptors)


def generate_fingerprints(smiles: pd.Series) -> pd.DataFrame:
    """
    Generate PubChem fingerprints for a list of SMILE strings using PaDEL.
    :param smiles: A pandas series of SMILE strings.
    :return: A pandas data frame of PubChem fingerprint bits by SMILE string.
    """
    print("Generating fingerprints:")
    smiles.to_csv("temp_smiles.smi", index=False, header=False)
    padeldescriptor(mol_dir="temp_smiles.smi", d_file="fingerprints.csv", fingerprints=True, retainorder=True)
    fingerprints_table = pd.read_csv("fingerprints.csv").drop("Name", axis="columns")
    os.remove("temp_smiles.smi")
    os.remove("fingerprints.csv")
    print("\tDone.\n")

    return fingerprints_table


def extract_smiles(smiles, max_length=250) -> np.ndarray:
    """
    Extract a stack of one-hot encoded 2D matrices from a list of SMILE strings.
    :param smiles: An iterable collection of SMILE strings.
    :param max_length: The length of the SMILE string dimension, those shorter than this are 0-padded to this length.
    :return: A 3-dimensional ndarray of (samples, smile positions, one-hot encoded features).
    """
    smile_list = []
    print("Extracting SMILE matrices:")
    for smile in notebook.tqdm(smiles):
        smile_list.append(__extract_smile_features(smile, max_length))

    return np.stack(smile_list)


def extract_smile_structures(smiles, resolution=100, scale=(-15, 15)) -> np.ndarray:
    """
    Extract a stack of 2-dimensional matricies from a list of SMILE strings.
    :param smiles: An iterable collection of SMILE strings.
    :return: A 4-dimensional ndarray of (samples, x-coordinates, y-coordinates, filters).
    """
    mol_list = []
    print("Extracting 2D structures:")
    for smile in notebook.tqdm(smiles):
        matrix = None
        mol = Chem.MolFromSmiles(smile)
        if mol is not None:
            mol.Compute2DCoords()
            matrix = __extract_mol_structure(mol, 0, resolution, scale)
        if matrix is None:
            matrix = np.full((resolution, resolution, 2), -1, dtype='b')
        mol_list.append(matrix)

    return np.stack(mol_list)


def __extract_atom_features(molecule, atom_index):
    symbol_features = []
    atom = molecule.GetAtomWithIdx(atom_index)
    symbol_features.append(atom.GetSymbol())
    symbol_features.append(atom.GetTotalNumHs())
    symbol_features.append(atom.GetTotalDegree())
    symbol_features.append(atom.GetFormalCharge())
    symbol_features.append(atom.GetTotalValence())
    symbol_features.append(atom.IsInRing() * 1)
    symbol_features.append(atom.GetIsAromatic() * 1)
    symbol_features.append(str(atom.GetChiralTag()))
    symbol_features.append(str(atom.GetHybridization()))
    symbol_features.append(0)
    two_char_abbr_flag = True if len(atom.GetSymbol()) > 1 else False
    return symbol_features, two_char_abbr_flag


def __extract_smile_features(smile, max_length):
    molecule = Chem.MolFromSmiles(smile)
    ion_flag = False
    two_char_abbr_flag = False
    two_digit_ring_flag = False
    ring_first_digit = 0
    ring_indices = []
    atom_index = 0
    smile_array = []
    if molecule:
        for character in smile:
            if re.match(r'[a-gi-z]', character, re.IGNORECASE):
                if two_char_abbr_flag:
                    two_char_abbr_flag = False
                else:
                    symbol_features, two_char_abbr_flag = __extract_atom_features(molecule, atom_index)
                    smile_array.append(symbol_features)
                    atom_index += 1
            elif re.match(r'[\\/.=#)(]', character):
                symbol_features = [character, 0, 0, 0, 0, 0, 0, 'CHI_UNSPECIFIED', 'UNSPECIFIED', 0]
                smile_array.append(symbol_features)
            elif re.match(r'[+-]', character):
                ion_flag = True
            elif re.match(r']', character):
                ion_flag = False
            elif re.match(r'%', character):
                two_digit_ring_flag = True
            elif re.match(r'[0-9]', character):
                if two_digit_ring_flag:
                    if ring_first_digit == 0:
                        ring_first_digit = character
                        continue
                    else:
                        character = ring_first_digit + character
                        ring_first_digit = 0
                        two_digit_ring_flag = False
                if not ion_flag:
                    symbol_features = ['ring']
                    if character not in ring_indices:
                        # Ring start.
                        symbol_features.extend([0, 0, 0, 0, 0, 0, 'CHI_UNSPECIFIED', 'UNSPECIFIED', 1])
                        ring_indices.append(character)
                    else:
                        # Ring end.
                        symbol_features.extend([0, 0, 0, 0, 0, 0, 'CHI_UNSPECIFIED', 'UNSPECIFIED', 2])
                    smile_array.append(symbol_features)
    # 0-Padding
    smile_array.extend([[0] * 10] * (max_length - len(smile_array)))
    smile_array = pd.DataFrame(smile_array)

    encoder = OneHotEncoder([['C', 'N', 'O', 'Br', 'Cl', 'F', 'P', 'S', 'ring', '(', ')', '/', '\\', '=', '#'],
                             ['CHI_TETRAHEDRAL_CCW', 'CHI_TETRAHEDRAL_CW'],
                             ['SP', 'SP2', 'SP3'], ['1', '2']], sparse=False, handle_unknown='ignore')

    smile_array = np.concatenate([smile_array[[1, 2, 3, 4, 5, 6]].to_numpy(),
                                  encoder.fit_transform(smile_array[[0, 7, 8, 9]].astype(str))], axis=1)
    return smile_array


def __extract_mol_structure(mol, conf_id, resolution, scale):
    digitizer = {'SINGLE': 1, 'AROMATIC': 2, 'DOUBLE': 3, 'TRIPLE': 4, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'P': 15, 'S': 16,
                 'Cl': 17, 'Br': 35, 'Other': 40}
    pixel_scale = (scale[1] - scale[0]) / resolution
    matrix = np.zeros((resolution, resolution, 2), dtype='b')

    conformer = mol.GetConformer(conf_id)
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        x = conformer.GetAtomPosition(atom.GetIdx()).x
        y = conformer.GetAtomPosition(atom.GetIdx()).y
        if x < scale[0] or x > scale[1] or y < scale[0] or y > scale[1]:
            return None

        j = int(np.floor((x - scale[0]) / pixel_scale))
        i = int(np.floor((scale[1] - y) / pixel_scale))
        if symbol not in digitizer.keys():
            symbol = 'Other'

        matrix[i, j, 0] = digitizer[symbol]

    for bond in mol.GetBonds():
        bond_type = bond.GetBondType()
        x_start = conformer.GetAtomPosition(bond.GetBeginAtomIdx()).x
        y_start = conformer.GetAtomPosition(bond.GetBeginAtomIdx()).y
        j_start = int(np.floor((x_start - scale[0]) / pixel_scale))
        i_start = int(np.floor((scale[1] - y_start) / pixel_scale))
        x_end = conformer.GetAtomPosition(bond.GetEndAtomIdx()).x
        y_end = conformer.GetAtomPosition(bond.GetEndAtomIdx()).y
        j_end = int(np.floor((x_end - scale[0]) / pixel_scale))
        i_end = int(np.floor((scale[1] - y_end) / pixel_scale))
        pixel_coords = __pixelate(i_start, j_start, i_end, j_end)
        for pixel in pixel_coords[1:]:
            matrix[pixel[0], pixel[1], 1] = digitizer[str(bond_type)]

    return matrix


def __pixelate(x0, y0, x1, y1):
    pixel_coords = []
    if abs(y1 - y0) < abs(x1 - x0):
        if x0 > x1:
            x0, x1 = x1, x0
            y0, y1 = y1, y0
        dx = x1 - x0
        dy = y1 - y0
        yi = 1
        if dy < 0:
            yi = -1
            dy = -dy
        d = 2 * dy - dx
        y = y0
        for x in range(x0, x1):
            pixel_coords.append((x, y))
            if d > 0:
                y += yi
                d -= 2 * dx
            d += 2 * dy
    else:
        if y0 > y1:
            x0, x1 = x1, x0
            y0, y1 = y1, y0
        dx = x1 - x0
        dy = y1 - y0
        xi = 1
        if dx < 0:
            xi = -1
            dx = -dx
        d = 2 * dx - dy
        x = x0
        for y in range(y0, y1):
            pixel_coords.append((x, y))
            if d > 0:
                x += xi
                d -= 2 * dy
            d += 2 * dx

    return pixel_coords
