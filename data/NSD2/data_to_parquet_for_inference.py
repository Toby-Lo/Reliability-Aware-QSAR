import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict
from tqdm import tqdm

def generate_scaffold_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    scaffold_mol = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold_mol)

def convert_split_datasets(csv_path, output_dir, split_seed=12345, train_size=0.8):
    df = pd.read_csv(csv_path)
    print(f"Original data shape: {df.shape}")

    print(f"Calculating Scaffolds (Split Seed: {split_seed})...")
    scaffolds = defaultdict(list)
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        scaffold_smiles = generate_scaffold_smiles(row['smiles'])
        scaffolds[scaffold_smiles].append(idx)
    
    scaffold_list = sorted(scaffolds.keys())
    
    rng = np.random.RandomState(split_seed)
    rng.shuffle(scaffold_list)
    
    train_indices = []
    n_train_target = int(len(df) * train_size)
    
    for scaffold in scaffold_list:
        if len(train_indices) + len(scaffolds[scaffold]) <= n_train_target:
            train_indices.extend(scaffolds[scaffold])
        else:
            break
    
    test_indices = list(set(df.index) - set(train_indices))
    
    df_train = df.iloc[train_indices].copy()
    df_test = df.iloc[test_indices].copy()

    def finalize_df(temp_df):
        # save as `zinc_id` column, make sure int64
        temp_df['zinc_id'] = temp_df.index.astype('int64')
        num_cols = temp_df.select_dtypes(include=[np.number]).columns
        temp_df[num_cols] = temp_df[num_cols].fillna(0.0)
        temp_df['smiles'] = temp_df['smiles'].astype(str)
        return temp_df

    df_train = finalize_df(df_train)
    df_test = finalize_df(df_test)

    # 4. Save as Parquet
    train_path = f"{output_dir}/nsd2_dev_set_seed{split_seed}.parquet"
    test_path = f"{output_dir}/nsd2_test_set_seed{split_seed}.parquet"
    
    pq.write_table(pa.Table.from_pandas(df_train, preserve_index=False), train_path, compression="zstd")
    pq.write_table(pa.Table.from_pandas(df_test, preserve_index=False), test_path, compression="zstd")
    
    print(f"\n✅ Split Completed(Split Seed: {split_seed})")
    print(f"--- (Development): {len(df_train)} Samples -> {train_path}")
    print(f"--- (External): {len(df_test)} Samples -> {test_path}")
    return train_path, test_path

if __name__ == "__main__":
    convert_split_datasets(
        csv_path="./data/NSD2/nsd2_final_dataset_feature_fingerprint.csv",
        output_dir="./data/NSD2",
        split_seed=12345
    )