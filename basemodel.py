from biotransformers import BioTransformers
import numpy as np
import pandas as pd
import torch

#PSA_WT Sequence
WT='MPSETYITKTLSLKLIPSDEEKQALENYFITFQRAVNFAIDRIVDIRSSFRYLNKNEQFPAVCDCCGKKEKIMYVNISNKTFKFKPSRNQKDRYTKDIYTIKPNAHICKTCYSGVAGNMFIRKQMYPNDKEGWKVSRSYNIKVNAPGLTGTEYAMAIRKAISILRSFEKRRRNAERRIIEYEKSKKEYLELIDDVEKGKTNKIVVLEKEGHQRVKRYKHKNWPEKWQGISLNKAKSKVKDIEKRIKKLKEWKHPTLNRPYVELHKNNVRIVGYETVELKLGNKMYTIHFASISNLRKPFRKQKKKSIEYLKHLLTLALKRNLETYPSIIKRGKNFFLQYPVRVTVKVPKLTKNFKAFGIDRGVNRLAVGCIISKDGKLTNKNIFFFHGKEAWAKENRYKKIRDRLYAMAKKLRGDKTKKIRLYHEIRKKFRHKVKYFRRNYLHNISKQIVEIAKENTPTVIVLEDLRYLRERTYRGKGRSKKAKKTNYKLNTFTYRMLIDMIKYKAEEAGVPVMIIDPRNTSRKCSKCGYVDENNRKQASFKCLKCGYSLNADLNAAVNIAKAFYECPTFRWEEKLHAYVCSEPDK'
table=pd.read_csv('round5.csv')
mut_index=table['name']
new_protein_list=[]
#last one is WT, just copy protein sequnce over
for index in range(len(mut_index)-1):
    #origin=mut_index[index][0]
    target=mut_index[index][-1]
    #print(target)
    position=mut_index[index][0:-1]
    #print(position)
    new_protein=list(WT)
    new_protein[int(position)-1]=target
    new_protein = ''.join([str(item) for item in new_protein])
    new_protein_list.append(new_protein)
#table=pd.concat([table,new_protein_list],axis=1)
new_protein_list.append(WT)

textfile = open("psa_round5_sequence.txt", "w")
for element in new_protein_list:
    textfile.write(element + "\n")
textfile.close()


file=open('psa_round5_sequence.txt','r')
label=file.readlines()
sequences=[item.strip() for item in label]


bio_trans = BioTransformers(backend="esm1b_t33_650M_UR50S")
embeddings = bio_trans.compute_embeddings(sequences, pool_mode=('cls','mean'))

mean_emb = embeddings['mean']
print(mean_emb)

np.savetxt('psa_round5_representation.csv',mean_emb,delimiter=",")

