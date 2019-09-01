#-*- encoding:utf8 -*-

import os
import time
import pickle
import torch as t
import numpy as np
from torch.utils import data
import nibabel as nib

class dataSet_V1(data.Dataset):
    def __init__(self,sequences_file=None,label_file=None,gos_gile=None,embeddings_file=None, PPI_embedding_file=None):
        super(dataSet_V1,self).__init__()
        with open(sequences_file,"rb") as fp_seq:
            self.all_sequences = pickle.load(fp_seq)

        with open(label_file,"rb") as fp_label:
            self.all_labels  = pickle.load(fp_label)

        with open(gos_gile, "rb") as fp_gos:
            self.all_gos = pickle.load(fp_gos)

        self.embeddings = np.load(embeddings_file)
        self.lookup_matrix = self.embeddings

        if PPI_embedding_file:
            with open(PPI_embedding_file,"rb") as fp_emb:
                self.all_net_embeddings = pickle.load(fp_emb,encoding="bytes")
        else:
            self.all_net_embeddings = None

        

    def __getitem__(self,index):
    
        feavalue_value = []
        for idx in self.all_sequences[index][:1000]:
            feavalue_value.append(self.lookup_matrix[idx])
        

        feavalue_value =  np.stack(feavalue_value)
        
        label =  self.all_labels[index]
        label = np.array(label,dtype=np.float32)

        gos = self.all_gos[index]

        if self.all_net_embeddings:
            embedding = np.array(self.all_net_embeddings[index],dtype=np.float32)
            return feavalue_value,embedding,label,gos
        else:
            return feavalue_value, label,gos

                

    def __len__(self):
    
        return len(self.all_sequences)
   
   
class dataSet_V2(data.Dataset):
    def __init__(self,sequences_file=None,label_file=None,gos_gile=None,embeddings_file=None, PPI_embedding_file=None,interpro_file=None):
        super(dataSet_V2,self).__init__()
        with open(sequences_file,"rb") as fp_seq:
            self.all_sequences = pickle.load(fp_seq)

        with open(label_file,"rb") as fp_label:
            self.all_labels  = pickle.load(fp_label)

        with open(gos_gile, "rb") as fp_gos:
            self.all_gos = pickle.load(fp_gos)

        with open(interpro_file,"rb") as fp_pro:
            self.inter_pro = pickle.load(fp_pro)

        self.embeddings = np.load(embeddings_file)
        self.lookup_matrix = self.embeddings

        if PPI_embedding_file:
            with open(PPI_embedding_file,"rb") as fp_emb:
                self.all_net_embeddings = pickle.load(fp_emb,encoding="bytes")
        else:
            self.all_net_embeddings = None

        

    def __getitem__(self,index):
    
        feavalue_value = []
        for idx in self.all_sequences[index][:1000]:
            feavalue_value.append(self.lookup_matrix[idx])

        feavalue_value =  np.stack(feavalue_value)

        label = self.all_labels[index]
        label = np.array(label,dtype=np.float32)
        gos = self.all_gos[index]

        no_zero_index = self.inter_pro[index]
        interpro_info = np.zeros(35020,dtype=np.float32)
        interpro_info[no_zero_index] = 1


        if self.all_net_embeddings:

            embedding = np.array(self.all_net_embeddings[index],dtype=np.float32)
            return feavalue_value,embedding,interpro_info,label,gos

        else:

            return feavalue_value, label,gos

                

    def __len__(self):
    
        return len(self.all_sequences)
        
        
        
        
class dataSet_V3(data.Dataset):
    def __init__(self,label_file=None,gos_gile=None, PPI_embedding_file=None,interpro_file=None):
        super(dataSet_V3,self).__init__()

        with open(label_file,"rb") as fp_label:
            self.all_labels  = pickle.load(fp_label)

        with open(gos_gile, "rb") as fp_gos:
            self.all_gos = pickle.load(fp_gos)

        with open(interpro_file,"rb") as fp_pro:
            self.inter_pro = pickle.load(fp_pro)


        if PPI_embedding_file:
            with open(PPI_embedding_file,"rb") as fp_emb:
                self.all_net_embeddings = pickle.load(fp_emb,encoding="bytes")
        else:
            self.all_net_embeddings = None

        

    def __getitem__(self,index):

        feavalue_value = [[0],[0]]
        feavalue_value = np.stack(feavalue_value)
        label =  self.all_labels[index]
        label = np.array(label,dtype=np.float32)


        gos = self.all_gos[index]

        no_zero_index = self.inter_pro[index]
        interpro_info = np.zeros(35020,dtype=np.float32)
        interpro_info[no_zero_index] = 1
        if self.all_net_embeddings:
            embedding = np.array(self.all_net_embeddings[index],dtype=np.float32)
            return feavalue_value,embedding,interpro_info,label,gos
        else:
            return feavalue_value, label,gos

                

    def __len__(self):
    
        return len(self.all_labels)
