import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from biotransformers import BioTransformers
import numpy as np
import argparse

def read_data(avg_csv, emb_csv):
    y = pd.read_csv(avg_csv)['average']
    seq = pd.read_csv(emb_csv, header=None)
    return y, seq

def train_model(X, y):
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    clf = RidgeCV().fit(X_scaled, y)
    return clf, scaler

def mutate_protein(WT, int_to_aa, mut_position, aa_mut):
    aa_to_int = {
    'M':1,
    'R':2,
    'H':3,
    'K':4,
    'D':5,
    'E':6,
    'S':7,
    'T':8,
    'N':9,
    'Q':10,
    'C':11,
    'U':12,
    'G':13,
    'P':14,
    'A':15,
    'V':16,
    'I':17,
    'F':18,
    'Y':19,
    'W':20,
    'L':21}
    int_to_aa = {value:key for key, value in aa_to_int.items()}
    new_protein_list, record = [], []
    for pos in mut_position:
        for mut in aa_mut:
            new_protein = list(WT)
            new_protein[pos] = int_to_aa[mut]
            mut_record = ''.join([str(pos+1), int_to_aa[mut]])  # Adjusting index for human-readable format
            new_protein = ''.join(new_protein)
            new_protein_list.append(new_protein)
            record.append(mut_record)
    return new_protein_list, record

def compute_embeddings(new_protein_list):
    bio_trans = BioTransformers(backend="esm1b_t33_650M_UR50S", multi_gpu=True)
    embeddings = bio_trans.compute_embeddings(new_protein_list, pool_mode=('cls', 'mean'), batch_size=2)
    return embeddings['mean']

def main(avg_csv, emb_csv):
    y, seq = read_data(avg_csv, emb_csv)
    clf, scaler = train_model(seq, y)

    # Assuming mut_position and aa_mut as given in the original code
    new_protein_list, record = mutate_protein(WT, int_to_aa, np.arange(1,480), np.arange(1, 22))
    mean_emb = compute_embeddings(new_protein_list)
    mean_emb_scaled = scaler.transform(mean_emb)
    predictions = clf.predict(mean_emb_scaled)

    # Save predictions and records
    np.savetxt('predict.csv', np.hstack((mean_emb_scaled, predictions[:, None])), delimiter=",")
    
    with open("record.txt", "w") as textfile:
        for element in record:
            textfile.write(element + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict values for protein mutations.")
    parser.add_argument("--avg_csv", required=True, help="CSV file containing 'average' values.")
    parser.add_argument("--emb_csv", required=True, help="CSV file containing sequence embeddings.")
    args = parser.parse_args()

    main(args.avg_csv, args.emb_csv)
