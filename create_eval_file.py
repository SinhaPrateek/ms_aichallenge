import torch
import model as m
import argparse
import data_helper as dh
import numpy as np

def load_model(model_path):
    #model_obj = m.Simp_Model_100_LSTM_EncDec()
    model_obj = m.Simp_Model_100_GRU_EncDec_samehidden_Attn()
    checkpoint_dict = torch.load(model_path)
    model_obj.load_state_dict(checkpoint_dict['model_state_dict'])
    return model_obj

def calculate_scores(loader_test,model,device):
    scores = torch.FloatTensor([]).to(device=device)
    passages = torch.FloatTensor([]).to(device=device)
    query_index = torch.IntTensor([]).to(device=device)
    passage_index = torch.IntTensor([]).to(device=device)
    model.eval()
    with torch.no_grad():
        for t, test_batch in enumerate(loader_test):
            for k, v in test_batch.items():
                test_batch[k] = v.to(device=device)
            scores = torch.cat((scores, model(test_batch['query'], test_batch['passage']))) 
            passages = torch.cat((passages, test_batch['passage']))
            query_index = torch.cat((query_index, test_batch['query_index']))
            passage_index = torch.cat((passage_index, test_batch['passage_index']))

    return scores, passages, query_index, passage_index.reshape((-1))

def create_final_file(eval_data_path,model_path):
    if not args.no_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        raise ValueError("GPU not available. Aborting.")
    print('using device:', device)
    model = load_model(model_path).to(device=device)

    loader_test,_ = dh.get_dataloader(eval_data_path, mode="test", batch_size=1024, mis_class=9, num_workers = 16, shuffle=False,embed_dim = 100)
    
    scores, passages, query_indices, passage_indices = calculate_scores(loader_test,model,device)
    unique_indices = torch.IntTensor(np.unique(query_indices.data)).to(device=device)
    
    f = open("/".join(model_path.split("/")[:-1]) + "/answer.tsv","w+")
    for q in unique_indices:
        q_slice = (query_indices == q).reshape((-1))
        q_score = scores[q_slice][:,0]
        passage_index = passage_indices[q_slice]
        passage_index, sort_idx = torch.sort(passage_index, descending=False)
        q_score = q_score[sort_idx]
        f.write("{:8d}".format(q))
        for score in q_score:
            f.write("\t{:.2f}".format(score.item()))
        f.write("\n")
    f.close()

if __name__=='__main__':
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_data_path", type=str, default='data/pkl/data_100_test_stop.pkl')
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--no_gpu", default=False, action='store_true')
    args = parser.parse_args()

    create_final_file(args.eval_data_path,args.model_path)
