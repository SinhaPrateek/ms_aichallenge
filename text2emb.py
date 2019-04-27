import re
import argparse
import random
import os
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
import time
from nltk import word_tokenize
#from nltk.corpus import stopwords
#import bz2
"""
    Converts each query and passage pair into glove embeddings
"""
re_not_alpha = re.compile("[^a-zA-Z]+")

class txt2emb:
    def __init__(self, embeddings_dim=50, source = "data/all_data.tsv", embeddings_file= "glove.6B.50d.txt", mode= "debug"):
        self.embeddings_dim = embeddings_dim
        self.embeddings = {}
        self.column_index = {"query":"1","passage":"2","passage_label":"3","passage_index":"4","query_index":"index"}
        #self.stopwords = set(stopwords.words('english'))     
        self.embeddingFileName = embeddings_file
        self.source = source
        self.mode = mode
        self._loadEmbeddings()
        
        
    # The following method takes Embedding file and stores all words and their embeddings in a dictionary
    def _loadEmbeddings(self):
        with open(self.embeddingFileName,"r",encoding="utf-8") as fe:
            for line in fe:
                tokens = line.strip().split()
                word = tokens[0]
                vec = [float(x) for x in tokens[1:]]
                self.embeddings[word] = vec
        self.embeddings["zerovec"] = [0.0]*self.embeddings_dim

        
    def get_vec(self, res_text, max_words=50):
        res_text = res_text.lower()
        words = word_tokenize(res_text)
        #words = [x for x in words if not x in self.stopwords]
        words = [re_not_alpha.sub("", w) for w in words]
        words = [x for x in words if x]  # to remove empty words

        remaining = max_words - len(words)
        if remaining > 0:
            words += ["zerovec"] * remaining  # Pad zero vecs if the word count is less than max__words
        else:
            words = words[:max_words]  # trim extra words

        # create Feature vector
        zerovec = self.embeddings["zerovec"]
        feature_vector = [self.embeddings[w] if w in self.embeddings else zerovec for w in words] # Adds zerovec for OOV terms

        return feature_vector
    
    
    def embed(self, rows):
        res = {}
        idx = rows[self.column_index["query_index"]].unique()[0]
        query = rows[self.column_index["query"]].unique()[0]
        query_embed = self.get_vec(query, max_words=12)

        cnt = 1 if self.mode == "test" else 2
        for _, row in rows.iterrows():
            p_label = 2 if(self.mode == 'test') else row[self.column_index["passage_label"]] 
            p_index = row[self.column_index["passage_index"]]
            p_embed = self.get_vec(row[self.column_index["passage"]])
            if p_label == 1:
                # positive case
                res[str(idx)+"_"+"1"] = {"query": query_embed,"passage": p_embed, \
                    "passage_index": p_index}
            else:
                # wrong/unknown case
                res[str(idx)+"_"+str(cnt)] = {"query":query_embed,"passage":p_embed,"passage_index":p_index}
                cnt+=1
        return res


    def convert(self, idx_file=None, dest_folder=None, suffix="debug"):
        '''
            Reads the query indices and selects all the query-passage pairs for that index
        '''
        if not idx_file:
            raise ValueError("File with query indices not provided")

        # Read all the query indices to be processed from the idx_file
        with open(idx_file, "r") as f:
            print("Reading indices")
            query_indices = [int(line.strip()) for line in f]

        txt_df = pd.read_csv(self.source, sep="\t")
        txt_df = txt_df.loc[txt_df['index'].isin(query_indices)]


        if not os.path.isdir(dest_folder):
            os.mkdir(dest_folder)

        embed_data = {}
        for q in tqdm(query_indices):
            temp_df = txt_df.loc[txt_df["index"] == q]
            temp_res = self.embed(temp_df)
            #with bz2.BZ2File("{}/test_single/{}.pkl".format(dest_folder,str(q)),"w") as g:
                #pickle.dump(temp_res,g)
                #import json
                #json.dump(temp_res,g)
            embed_data.update(temp_res)

        f_data = dest_folder + "/data_" + str(self.embeddings_dim) + "_" + suffix + "_stop"+".pkl"

        with open(f_data, "wb") as f:
            pickle.dump(embed_data, f)


if __name__ == "__main__":
    '''
        Create dataset in pkl format with embeddings
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "val", "debug","test"], type=str, required=True)
    parser.add_argument("--indices_file", type=str, required=True, help="file containing indices based on mode")
    parser.add_argument("--dest_folder", type=str, required=True)
    parser.add_argument("--text_data", type=str, default="data/all_data.tsv", help="all data from which train,val and debug indices are taken ")
    parser.add_argument("--embeddings_file", type=str, required=True)
    parser.add_argument("--embeddings_dim", type=int, required=True)
    args = parser.parse_args()
    
    if args.mode == 'test':
        args.text_data = 'data/test_data.tsv'
    t2e = txt2emb(source=args.text_data, embeddings_dim=args.embeddings_dim, embeddings_file=args.embeddings_file, mode=args.mode)
    t2e.convert(idx_file=args.indices_file, dest_folder=args.dest_folder, suffix=args.mode)

