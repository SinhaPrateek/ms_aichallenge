import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import random


'''
    Helper for managing dataset operations
'''

ID_FILES = {"train":"data/query_ids/train.ids", "val":"data/query_ids/val.ids", "debug":"data/query_ids/debug.ids", "test":"data/query_ids/test.ids"}


class DataSetReader(Dataset):
    def __init__(self, data_path, mis_class, mode, embed_dim):
        print("Reading data from {} in {} mode".format(data_path, mode))
        with open(data_path, "rb") as df:
            self._data = pickle.load(df)
        self._embed_dim = embed_dim
        indices_file = ID_FILES[mode]
        print("Reading indices from {}".format(indices_file))
        with open(indices_file, "r") as f:
            self._indexes = [int(line.strip()) for line in f]
        
        key_suffix = list(range(2, 11))
        if mode == "train" or mode == "debug":
            self.mis_class = mis_class
            key_suffix = [1] + random.sample(key_suffix, self.mis_class)
        else: 
            self.mis_class = 9
            key_suffix = [1] + key_suffix

        self._data_keys = []
        for i in self._indexes:
            for s in key_suffix:
                self._data_keys.append("{:d}_{:d}".format(i, s))


    def __len__(self):
        return len(self._data_keys)


    def __getitem__(self, idx):
        sample_key = self._data_keys[idx]
        sample = self._data[sample_key]

        query = torch.FloatTensor(sample['query']).view(1, 12, self._embed_dim)
        passage = torch.FloatTensor(sample['passage']).view(1, 50, self._embed_dim)

        idx_label = sample_key.split("_")
        query_index = torch.IntTensor([int(idx_label[0])])
        passage_index = torch.IntTensor([sample['passage_index']])
        passage_label = float(idx_label[-1])
        
        if passage_label > 1.0:
            # nagative. In test mode, this key is never checked, so it doesn't matter then
            passage_label = [0.0,1.0]
        else:
            passage_label = [1.0,0.0]
        passage_label = torch.FloatTensor(passage_label)

        return {"query_index":query_index, 
            "query":query,
            "passage":passage,
            "passage_index": passage_index,
            "passage_label":passage_label
            }

class DataSetReader_Siamese(Dataset):
    def __init__(self, data_path, mis_class, mode, embed_dim):
        print("Reading data from {} in {} mode".format(data_path, mode))
        with open(data_path, "rb") as df:
            self._data = pickle.load(df)

        self._embed_dim = embed_dim
        indices_file = ID_FILES[mode]
        print("Reading indices from {}".format(indices_file))
        with open(indices_file, "r") as f:
            self._indexes = [int(line.strip()) for line in f]
        
        key_suffix = list(range(2, 11))
        if mode == "train" or mode == "debug":
            self.mis_class = mis_class
            key_suffix = [1] + random.sample(key_suffix, self.mis_class)
        else: 
            self.mis_class = 9
            key_suffix = [1] + key_suffix



    def __len__(self):
        return len(self._indexes)


    def __getitem__(self, idx):
        sample_key = self._indexes[idx]
        key_suffix = list(range(2,11))
        key_suffix = [1] + random.sample(key_suffix,self.mis_class)
        
        data_keys = []
        
        for s in key_suffix:
            data_keys.append("{:d}_{:d}".format(sample_key, s))

        
        sample = self._data[data_keys[0]]
        query_index = torch.IntTensor([sample_key])

        query = torch.FloatTensor(sample['query']).view(1, 12, self._embed_dim)
        passages = []
        passage_indexes = []
        passage_labels = []
        for ks in data_keys:
            passage = torch.FloatTensor(self._data[ks]['passage']).view(1, 50, self._embed_dim)
            passage_index = torch.IntTensor([self._data[ks]['passage_index']])
            k = int(ks.split("_")[1])
            if k > 1:
                passage_label = torch.FloatTensor([0.0,1.0])
            else:
                passage_label = torch.FloatTensor([1.0,0.0])
            passage_indexes.append(passage_index)
            passage_labels.append(passage_label)
            passages.append(passage)
        
        return {"query_index":query_index, 
            "query":query,
            "passage":torch.stack(passages),
            "passage_index": torch.stack(passage_indexes),
            "passage_label":torch.stack(passage_labels)
            }



def get_dataloader(data_path, mode, batch_size=16, mis_class=9, num_workers=4, shuffle=True, embed_dim = 50):
    if mode=="train" or mode=="debug":
        dataset = DataSetReader(data_path=data_path,mis_class=mis_class,mode=mode,embed_dim = embed_dim)
    else:
        dataset = DataSetReader(data_path=data_path, mis_class=mis_class, mode=mode,embed_dim = embed_dim)
    
    print("Loading dataset with:\nbatch_size={}".format(batch_size))
    loader  = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)

    print("Dataset Loaded")
    return loader,len(dataset)

