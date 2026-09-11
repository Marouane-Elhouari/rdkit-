from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import (
    AllChem,
    Descriptors,
    Draw,
    MACCSkeys,
    PandasTools,
    rdFingerprintGenerator,
    rdMolDescriptors,
)
from rdkit.ML.Descriptors import MoleculeDescriptors

MOL_FILE = Path("morphine.mol")
EXCEL_FILE = Path("Heteroaromatics.xlsx")


def infos_molecule(mol_path: Path) -> Chem.Mol:
    mol = Chem.MolFromMolFile(str(mol_path), removeHs=False)
    if mol is None:
        raise ValueError(f"Impossible de lire la molécule depuis {mol_path}")

    symboles = [atom.GetSymbol() for atom in mol.GetAtoms()]

    print("Formula:", rdMolDescriptors.CalcMolFormula(mol))
    print("Molecular weight:", Descriptors.MolWt(mol))
    print("Number of atoms:", mol.GetNumAtoms())
    print("Number of bonds:", mol.GetNumBonds())
    print("Symboles des atomes :", symboles)
    print("Nombre d'azotes (N) :", symboles.count("N"))
    print("Nombre d'oxygènes (O) :", symboles.count("O"))
    print("Nombre de carbones (C) :", symboles.count("C"))

    logp = Descriptors.MolLogP(mol)
    if logp > 0:
        print("La molécule est hydrophobe / non polaire")
    elif logp == 0:
        print("La molécule est neutre")
    else:
        print("La molécule est hydrophile / polaire")

    print("Nombre de liaisons rotatives :", Descriptors.NumRotatableBonds(mol))
    print("SMILES canonique :", Chem.MolToSmiles(mol))
    return mol


def visualiser_2d(mol: Chem.Mol):
    mol2d = Chem.Mol(mol)
    AllChem.Compute2DCoords(mol2d)
    return Draw.MolToImage(mol2d, size=(600, 600))


def visualiser_3d(mol: Chem.Mol):
    import py3Dmol

    viewer = py3Dmol.view(width=600, height=500)
    viewer.addModel(Chem.MolToMolBlock(mol), "mol")
    viewer.setStyle({"stick": {}, "sphere": {"scale": 0.25}})
    viewer.setStyle({"elem": "N"}, {"stick": {"color": "purple"}, "sphere": {"color": "purple", "scale": 0.25}})
    viewer.setBackgroundColor("black")
    viewer.zoomTo()
    return viewer


MOLECULES_REF = {
    "Water": "O",
    "Methane": "C",
    "Ethanol": "CCO",
    "Acetic acid": "CC(=O)O",
    "Benzene": "c1ccccc1",
    "Toluene": "Cc1ccccc1",
    "Phenol": "Oc1ccccc1",
    "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "Caffeine": "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
    "Paracetamol": "CC(=O)Nc1ccc(O)cc1",
    "Ibuprofen": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "Glucose": "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O",
    "Morphine": "CN1CC[C@]23c4c5ccc(O)c4O[C@H]2[C@@H](O)C=C[C@H]3[C@H]1C5",
    "Cholesterol": "CC(C)CCCC(C)C1CCC2C1(CCC3C2CC=C4C3(CCC(C4)O)C)C",
    "Nicotine": "CN1CCC[C@H]1c1cccnc1",
    "Vitamin C": "OC[C@H](O)[C@H]1OC(=O)C(O)=C1O",
    "Dopamine": "NCCc1ccc(O)c(O)c1",
    "Serotonin": "NCCc1c[nH]c2ccc(O)cc12",
    "Adrenaline": "CNC[C@H](O)c1ccc(O)c(O)c1",
    "Cortisone": "CC12CCC(=O)C=C1CCC1C2C(=O)CC2(C)C1CCC2(O)C(=O)CO",
    "TNT": "Cc1c(cc(cc1[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]",
}


def construire_molecules_ref(smiles_dict: dict, sauvegarder_images: bool = False):
    mol_list = []
    for name, smiles in smiles_dict.items():
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"SMILES invalide pour {name}, ignoré")
            continue
        mol_list.append((name, mol))
        print(f"{name}: {Chem.MolToSmiles(mol)}")

        if sauvegarder_images:
            AllChem.Compute2DCoords(mol)
            Draw.MolToImage(mol, size=(300, 300)).save(f"{name}.png")
    return mol_list


def charger_dataframe_smiles(excel_path: Path, add_hs: bool = True) -> pd.DataFrame:
    df = pd.read_excel(excel_path)
    if "Smiles" not in df.columns:
        raise ValueError("Le fichier doit contenir une colonne 'Smiles'")

    mols, valid_mask, bad_smiles = [], [], []
    for s in df["Smiles"]:
        if pd.isna(s):
            valid_mask.append(False)
            continue
        mol = Chem.MolFromSmiles(str(s))
        if mol is None:
            bad_smiles.append(s)
            valid_mask.append(False)
            continue
        mols.append(Chem.AddHs(mol) if add_hs else mol)
        valid_mask.append(True)

    df = df[valid_mask].reset_index(drop=True)
    df["mol"] = mols

    print("Molécules valides :", len(mols))
    print("SMILES invalides ignorés :", len(bad_smiles))
    return df


def apercu_grille(df: pd.DataFrame, n: int = 8):
    return Draw.MolsToGridImage(df["mol"].tolist()[:n], molsPerRow=4, subImgSize=(200, 200))


def calculer_fingerprints(df: pd.DataFrame, methode: str, n_bits: int = 2048) -> pd.DataFrame:
    generateurs = {
        "morgan": lambda: rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=n_bits),
        "rdkit": lambda: rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=n_bits),
        "atompair": lambda: rdFingerprintGenerator.GetAtomPairGenerator(fpSize=n_bits),
        "torsion": lambda: rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=n_bits),
    }

    lignes = []
    prefixe = methode.upper()
    for mol in df["mol"]:
        arr = np.zeros((n_bits,), dtype=np.int8)
        if mol is not None:
            if methode == "maccs":
                bv = MACCSkeys.GenMACCSKeys(mol)
                arr = np.zeros((bv.GetNumBits(),), dtype=np.int8)
                DataStructs.ConvertToNumpyArray(bv, arr)
            elif methode == "pattern":
                from rdkit.Chem import PatternFingerprint
                bv = PatternFingerprint(mol, fpSize=n_bits)
                DataStructs.ConvertToNumpyArray(bv, arr)
            elif methode == "avalon":
                from rdkit.Avalon import pyAvalonTools
                bv = pyAvalonTools.GetAvalonFP(mol, nBits=n_bits)
                DataStructs.ConvertToNumpyArray(bv, arr)
            elif methode in generateurs:
                bv = generateurs[methode]().GetFingerprint(mol)
                DataStructs.ConvertToNumpyArray(bv, arr)
            else:
                raise ValueError(f"Méthode inconnue : {methode}")
        lignes.append(arr)

    colonnes = [f"{prefixe}_{i}" for i in range(len(lignes[0]))]
    df_fp = pd.DataFrame(lignes, columns=colonnes)
    return pd.concat([df.reset_index(drop=True), df_fp], axis=1)


def resume_bits(df_fp: pd.DataFrame, row_idx: int, prefixe_colonnes_a_ignorer: int = 2):
    ligne = df_fp.iloc[row_idx, prefixe_colonnes_a_ignorer:]
    print(f"Nombre de bits à 1 : {(ligne == 1).sum()}")
    print(f"Nombre de bits à 0 : {(ligne == 0).sum()}")
    print(ligne[ligne == 1])


def calculer_descripteurs_un(mol: Chem.Mol, noms: list[str]) -> dict:
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(noms)
    return dict(zip(noms, calc.CalcDescriptors(mol)))


def calculer_descripteurs_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    noms_descripteurs = [x[0] for x in Descriptors._descList]
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(noms_descripteurs)

    valeurs = [
        calc.CalcDescriptors(mol) if mol is not None else [np.nan] * len(noms_descripteurs)
        for mol in df["mol"]
    ]
    desc_df = pd.DataFrame(valeurs, columns=noms_descripteurs)
    return pd.concat([df.reset_index(drop=True), desc_df], axis=1)


def calculer_descripteurs_padel(df: pd.DataFrame, xml_dir: Path, smi_path: Path):
    import glob
    from padelpy import padeldescriptor

    df.to_csv(smi_path, index=None, header=None)
    resultats = {}
    for xml_file in sorted(glob.glob(str(xml_dir / "*.xml"))):
        out_csv = Path(xml_file).with_suffix(".csv")
        padeldescriptor(
            mol_dir=str(smi_path),
            d_file=str(out_csv),
            descriptortypes=xml_file,
            retainorder=True,
            fingerprints=True,
            d_2d=False,
            d_3d=False,
        )
        fp = pd.read_csv(out_csv)
        fp = pd.concat([df.reset_index(drop=True), fp.drop(columns="Name")], axis=1)
        fp.to_csv(out_csv, index=None)
        resultats[Path(xml_file).stem] = fp
        print(f"{Path(xml_file).stem} terminé")
    return resultats


if __name__ == "__main__":
    mol = infos_molecule(MOL_FILE)

    construire_molecules_ref(MOLECULES_REF, sauvegarder_images=False)

    df = charger_dataframe_smiles(EXCEL_FILE, add_hs=True)
    PandasTools.RenderImagesInAllDataFrames(images=True)

    df_maccs = calculer_fingerprints(df, methode="maccs")
    df_morgan = calculer_fingerprints(df, methode="morgan", n_bits=2048)

    if len(df) > 0:
        resume_bits(df_morgan, row_idx=0)

    df_final = calculer_descripteurs_dataframe(df)
    print(df_final.head())