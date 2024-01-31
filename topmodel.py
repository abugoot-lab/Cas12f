import pandas as pd
from sklearn import linear_model
import random
from biotransformers import BioTransformers
import numpy as np
from sklearn.preprocessing import StandardScaler

function=pd.read_csv('round5.csv')
y=function['average']
seq=pd.read_csv('psa_round5_representation.csv',header=None)

scaler = StandardScaler()
scaler.fit(seq)
a=scaler.transform(seq)
# pkl_filename = "preprocess.pkl"
# with open(pkl_filename, 'wb') as file:
#     pickle.dump(scaler, file)

# # Load from file
# with open(pkl_filename, 'rb') as file:
#     pickle_model = pickle.load(file)


clf=linear_model.RidgeCV()
clf.fit(a,y)
print(clf.score(a,y))

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

WT='MPSETYITKTLSLKLIPSDEEKQALENYFITFQRAVNFAIDRIVDIRSSFRYLNKNEQFPAVCDCCGKKEKIMYVNISNKTFKFKPSRNQKDRYTKDIYTIKPNAHICKTCYSGVAGNMFIRKQMYPNDKEGWKVSRSYNIKVNAPGLTGTEYAMAIRKAISILRSFEKRRRNAERRIIEYEKSKKEYLELIDDVEKGKTNKIVVLEKEGHQRVKRYKHKNWPEKWQGISLNKAKSKVKDIEKRIKKLKEWKHPTLNRPYVELHKNNVRIVGYETVELKLGNKMYTIHFASISNLRKPFRKQKKKSIEYLKHLLTLALKRNLETYPSIIKRGKNFFLQYPVRVTVKVPKLTKNFKAFGIDRGVNRLAVGCIISKDGKLTNKNIFFFHGKEAWAKENRYKKIRDRLYAMAKKLRGDKTKKIRLYHEIRKKFRHKVKYFRRNYLHNISKQIVEIAKENTPTVIVLEDLRYLRERTYRGKGRSKKAKKTNYKLNTFTYRMLIDMIKYKAEEAGVPVMIIDPRNTSRKCSKCGYVDENNRKQASFKCLKCGYSLNADLNAAVNIAKAFYECPTFRWEEKLHAYVCSEPDK'
new_protein_list=[]
record=[]
mut_position=np.arange(480)+11
aa_mut=np.arange(21)+1
for pos in mut_position:
    for mut in aa_mut:
        new_protein = list(WT)
        new_protein[int(pos)] = int_to_aa[mut]
        mut_record=''.join([str(pos),int_to_aa[mut]])
        new_protein = ''.join([str(item) for item in new_protein])
        new_protein_list.append(new_protein)
        record.append(mut_record)
if (len(record) % 2) == 1: 
    record.append('WT')
    new_protein_list.append(WT)


bio_trans = BioTransformers(backend="esm1b_t33_650M_UR50S",multi_gpu=True)
embeddings = bio_trans.compute_embeddings(new_protein_list, pool_mode=('cls','mean'),batch_size=2)

mean_emb = embeddings['mean']
mean_emb=scaler.transform(mean_emb)

c=clf.predict(mean_emb)
d=np.concatenate((mean_emb,c[:,None]),axis=1)

np.savetxt('predict.csv',d,delimiter=",")

textfile = open("record.txt", "w")
for element in record:
    textfile.write(element + "\n")
textfile.close()
