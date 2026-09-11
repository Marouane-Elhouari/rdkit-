import pandas as pd
import warnings

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem, PandasTools
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import PandasTools
df = pd.read_excel(r'C:\Users\pc\Desktop\project\tutorial_rdkit\Heteroaromatics.xlsx')

print(df.head(10))
mol_list = []

for smile in df['Smiles']:
  mol = Chem.MolFromSmiles(smile)
  mol = Chem.AddHs(mol)
  mol_list.append(mol)

df = pd.concat([df, pd.DataFrame(mol_list, columns = (['mol']))], axis=1)
     

desc_list = MoleculeDescriptors.MolecularDescriptorCalculator([x[0] for x in Descriptors._descList])


names = desc_list.GetDescriptorNames()

print(names)

MoleculeDescriptors.MolecularDescriptorCalculator(['MolLogP']).GetDescriptorSummaries()
     
func_logp = MoleculeDescriptors.MolecularDescriptorCalculator(['MolLogP'])
print(f'the logp of the first molecule is {func_logp.CalcDescriptors(mol_list[0])}')


print(Draw.MolToImage(mol_list[0]))


