import os
import argparse
import json
import time
from tqdm import tqdm
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import data_helper as dh
import model as m
from tensorboardX import SummaryWriter

def save_model(path, epoch, iteration, model, optim, optim_method, batch_size, mis_class):
    torch.save({
            "model_type": "".join(model.__class__.__name__.split("_")),
            'epoch': epoch,
            'iteration': iteration,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            'optim_method': optim_method,
            'batch_size':batch_size, 
            'mis_class':mis_class
            }, path)

def loss_siamese(pred,target):
    losses = []
    for i in range(args.mis_class):
        p = pred[:,i,:]
        t = target[:,i,:]
        l = -1 * (t*torch.log(p))
        if i!=0:
            l = float(1/args.mis_class) * l
        losses.append(l)
    losses = torch.stack(losses,1)
    losses = losses.view(pred.shape[0],args.mis_class*2)
    losses = torch.sum(losses,1)
    loss = torch.mean(losses)
    return loss
            

def train(model, num_epochs, mis_class, batch_size, model_name, optim_name="adam", optimizer=None, print_every=10, device=None, use_shuffle=True, checkpoint=None):
    """
    Inputs:
    - model: A PyTorch Module giving the model to train.
    - optimizer: An Optimizer object we will use to train the model
    - epochs: (Optional) The number of epochs to train for
    """
    logger = SummaryWriter("runs/logs/" + model_name)
    if not args.no_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        raise ValueError("GPU not available. Aborting.")
    print('using device:', device)
    model = model.cuda()  # move the model parameters to CPU/GPU

    if args.mode == "train":
        loader,samples = dh.get_dataloader("data/pkl/data_{}_train_stop.pkl".format(args.embed_dim), mode="train", batch_size=batch_size, mis_class=mis_class, num_workers=args.num_workers, shuffle=use_shuffle, embed_dim = args.embed_dim)
    else:
        loader,samples = dh.get_dataloader("data/pkl/data_{}_debug_negstop.pkl".format(args.embed_dim), mode="debug", batch_size=batch_size, mis_class=mis_class, num_workers=args.num_workers, shuffle=use_shuffle, embed_dim = args.embed_dim)

    param = list(model.named_parameters())
    
    print("Number of samples : {}".format(samples))

    steps_per_epoch = np.ceil(samples / args.batch_size).astype(np.int32)
    num_total_steps = num_epochs * steps_per_epoch
    print("Steps or batches per epoch is : {}".format(steps_per_epoch))
    print("Total number of steps or batches are : {}".format(num_total_steps))


    num_training_params = 0
    for n,p in param:
        if p.requires_grad:
            num_training_params += np.array(p.size()).prod()
    print("Number of trainable parameters: {}".format(num_training_params))

    print("Start training")
    it = 0
    pbar = tqdm(range(args.from_epoch, args.from_epoch + num_epochs))
    val_mrr = 0
    epoch_loss = 0
    epoch_acc = 0
    if args.retrain:
        f = open("models/{}/train_logs.txt".format(model_name),"w+") if not args.mode == "debug" else None
    else:
        f = open("models/{}/train_logs.txt".format(model_name),"a") 

    for e in pbar:
        #print("epoch\tbatch\tloss\ttrain_accuracy\tnum_correct/num_samples")
        #print("Val Accuracy {:.2f}".format(val_accuracy(loader_val,model,device)[0]))
        model.train()  # put model to training mode
        for num_iter, samp_batch in enumerate(loader):
            if checkpoint and num_iter < checkpoint['iteration']: continue
            for k, v in samp_batch.items():
                samp_batch[k] = v.to(device=device)
            loss_multiplier = (samp_batch['passage_label'][:,0].clone()) 
            loss_multiplier[loss_multiplier==0] = 1/mis_class
            loss_multiplier = loss_multiplier.view((-1,1))
            #samp_batch["passage_label"] = samp_batch["passage_label"].reshape((-1))
            
            scores = model(samp_batch["query"], samp_batch["passage"])
            
            if args.penalize:
                loss = F.binary_cross_entropy(scores, samp_batch["passage_label"],weight = loss_multiplier)
                #loss = loss_siamese(scores,samp_batch["passage_label"])
            else:
                loss = F.binary_cross_entropy(scores, samp_batch["passage_label"])
            # Zero out all of the gradients for the variables which the optimizer
            # will update.
            optimizer.zero_grad()

            # This is the backwards pass: compute the gradient of the loss with
            # respect to each  parameter of the model.
            loss.backward()
            #print(scores.shape)
            # Actually update the parameters of the model using the gradients
            # computed by the backwards pass.
            optimizer.step()
            if num_iter % print_every == 0:
                batch_acc  = get_accuracy_softmax(scores, samp_batch["passage_label"])
                epoch_acc  = (epoch_acc * num_iter + batch_acc["accuracy"]) / (num_iter + 1) # this is f-score
                epoch_loss = (epoch_loss * num_iter + loss.item()) / (num_iter + 1)
                logger.add_scalar("per_iter/accuracy", batch_acc["accuracy"], it)
                logger.add_scalar("per_iter/loss", loss.item(), it)
                logger.add_scalar("per_iter/true_pos", batch_acc["tp"], it)
                logger.add_scalar("per_iter/true_neg", batch_acc["tn"], it)
                if f:
                    f.write("Epoch {:2d}\t Iterations {:3d}\t Loss {:.2f}\t Batch_acc{:.2f}\t True_Positive {}\t True_Negative {}\t Val_Accuracy {:.4f} \n".format(e+1, num_iter+1, loss, batch_acc["accuracy"],batch_acc['tp'], batch_acc['tn'], val_mrr))
                pbar.set_description("Epoch {:2d} Iterations {:3d} Loss {:.2f} Batch_acc {:.2f} True_Positive {} True_Negative {} Val_Accuracy {:.4f}".format(e+1, num_iter+1, loss, batch_acc["accuracy"], batch_acc['tp'], batch_acc['tn'], val_mrr))

            it += 1
             
            path = "models/{}/model_{}_{}.pb".format(model_name,e+1,num_iter)
            if num_iter % 2000==0:
                save_model(path,e+1,it,model,optimizer,optim_name,batch_size, mis_class)
        
        if (e+1)%5==0 and args.decay_factor:
            adjust_learning_rate(optimizer,(e+1)/5)
        # print("--------------------------------------")
        # print("Epoch Summary:\nEpoch Number = {:3d}\nValidation MRR = {:.2f}".format(e+1, val_accuracy(loader_val, model, device)[0]))
        # print("--------------------------------------")
        param_new = list(model.named_parameters())
        
        #for i in range(len(param)):
         #     if np.array_equal(param[i][1].data.cpu().numpy(),param_new[i][1].data.cpu().numpy()):
          #       print(param[i][0],param_new[i][0])
        path = "models/{}/model_{}.pb".format(model_name,e+1)
        save_model(path,e,it,model,optimizer,optim_name,batch_size, mis_class)

        val_mrr, f_measures = val_accuracy(model, device)
        logger.add_scalar("per_epoch/validation_mrr", val_mrr, e+1)
        logger.add_scalar("per_epoch/train_accuracy", epoch_acc, e+1)
        logger.add_scalar("per_epoch/train_loss", epoch_loss, e+1)
        logger.add_scalar("per_epoch/true_pos", f_measures["tp"], e+1)
        logger.add_scalar("per_epoch/true_neg", f_measures["tn"], e+1)
        pbar.set_description("Epoch {:2d}  Iterations {:3d} Loss {:.2f} Batch_acc {:.2f} True_Positive {} True_Negative {} Val_Accuracy {:.2f}".format(e+1, num_iter+1, loss, epoch_acc, f_measures['tp'], f_measures['tn'],val_mrr))
        if f: f.write("Epoch Summary:\nEpoch Number = {:3d}\nValidation MRR = {:.2f}\nTrue_Positive {} || True_Negative {}".format(e+1, val_mrr, f_measures['tp'], f_measures['tn']))
    if f: f.close()


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every epoch"""
    if args.decay_factor:
        lr = args.learning_rate * args.decay_factor
    else:
        lr = args.learning_rate * (5)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_accuracy(scores, target):
    preds = torch.round(scores)
    confusion_vector = (preds.reshape((-1)))/(target.reshape((-1)))
    tp = torch.sum(confusion_vector == 1).item()
    fp = torch.sum(confusion_vector == float('inf')).item()
    tn = torch.sum(torch.isnan(confusion_vector)).item()
    fn = torch.sum(confusion_vector == 0).item() 
    precision = tp/(tp+fp) if tp+fp != 0 else 0.0
    recall = tp/(tp+fn)
    f_score = 2*(precision*recall)/(precision+recall) if (precision+recall)!=0.0 else 0.0
    return {"tp": tp, "tn": tn, "accuracy": f_score}
    #num_correct = (preds == target).sum()
    #num_samples = preds.size(0)
    #accuracy = float(num_correct) / num_samples
    #return {"num_correct": num_correct, "num_samples": num_samples, "accuracy": 100.0 * accuracy}
    


def get_accuracy_softmax(scores,target):
    preds = torch.round(scores)
    preds = preds[:,0]
    target = target[:,0]
    #temp_target = target.clone()
    #temp_target[target==0] = 1
    #temp_target[target!=0] = 0
    #target = temp_target
    confusion_vector = (preds.reshape((-1)))/(target.reshape((-1)))
    tp = torch.sum(confusion_vector == 1).item()
    fp = torch.sum(confusion_vector == float('inf')).item()
    tn = torch.sum(torch.isnan(confusion_vector)).item()
    fn = torch.sum(confusion_vector == 0).item() 
    precision = tp/(tp+fp) if tp+fp != 0 else 0.0
    recall = tp/(tp+fn) if tp+fn != 0 else 0.0
    f_score = 2*(precision*recall)/(precision+recall) if (precision+recall)!=0.0 else 0.0
    return {"tp": tp, "tn": tn, "accuracy": f_score}

def get_accuracy_siamese(scores,target):
    preds = torch.round(scores)
    preds = preds[:,:,0].view(preds.shape[0]*(args.mis_class+1))
    target = target[:,:,0].view(target.shape[0]*(args.mis_class+1))
    #temp_target = target.clone()
    #temp_target[target==0] = 1
    #temp_target[target!=0] = 0
    #target = temp_target
    confusion_vector = (preds.reshape((-1)))/(target.reshape((-1)))
    tp = torch.sum(confusion_vector == 1).item()
    fp = torch.sum(confusion_vector == float('inf')).item()
    tn = torch.sum(torch.isnan(confusion_vector)).item()
    fn = torch.sum(confusion_vector == 0).item() 
    precision = tp/(tp+fp) if tp+fp != 0 else 0.0
    recall = tp/(tp+fn) if tp+fn != 0 else 0.0
    f_score = 2*(precision*recall)/(precision+recall) if (precision+recall)!=0.0 else 0.0
    return {"tp": tp, "tn": tn, "accuracy": f_score}



def val_accuracy(model, device):
    scores = torch.FloatTensor([]).to(device=device)
    target = torch.FloatTensor([]).to(device=device)
    query_index = torch.IntTensor([]).to(device=device)
    loader_val,_ = dh.get_dataloader("data/pkl/data_{}_val_stop.pkl".format(args.embed_dim), mode="val", batch_size=1024, mis_class=9, num_workers=args.num_workers, shuffle=False, embed_dim = args.embed_dim)
    model.eval()
    with torch.no_grad():
        for t, val_batch in enumerate(loader_val):
            for k, v in val_batch.items():
                val_batch[k] = v.to(device=device)


            scores = torch.cat((scores, model(val_batch['query'], val_batch['passage'])))
            target = torch.cat((target, val_batch['passage_label']))
            query_index = torch.cat((query_index, val_batch['query_index']))
    del loader_val
    return get_mrr_softmax(scores, target, query_index, device), get_accuracy_softmax(scores, target)

def val_accuracy_siamese(model, device):
    scores = torch.FloatTensor([]).to(device=device)
    target = torch.FloatTensor([]).to(device=device)
    query_index = torch.IntTensor([]).to(device=device)
    loader_val,_ = dh.get_dataloader("data/pkl/data_{}_val_stop.pkl".format(args.embed_dim), mode="val", batch_size=1024, mis_class=9, num_workers=args.num_workers, shuffle=False, embed_dim = args.embed_dim)
    model.eval()
    with torch.no_grad():
        for t, val_batch in enumerate(loader_val):
            for k, v in val_batch.items():
                val_batch[k] = v.to(device=device)

            scores = torch.cat((scores, model(val_batch['query'], val_batch['passage'].view(val_batch['passage'].shape[0],1,1,50,50)).view(val_batch['passage'].shape[0],2)))
            target = torch.cat((target, val_batch['passage_label']))
            query_index = torch.cat((query_index, val_batch['query_index']))
    del loader_val
    return get_mrr_softmax(scores, target, query_index, device), get_accuracy_softmax(scores, target)



def get_mrr(scores, targets, query_indices, device):
    sort_per_query = {}
    mrr = 0.0
    num_q = 0.0
    unique_indices = torch.IntTensor(np.unique(query_indices.data)).to(device=device)
    for q in unique_indices:
        q_slice = (query_indices == q).reshape((-1))
        q_score = scores[q_slice]
        q_target = targets[q_slice]
        q_score, sort_idx = torch.sort(q_score, descending=True)
        q_target = q_target[sort_idx]
        rank = q_target.nonzero()[0][0].float() + 1.0
        q_mrr = torch.reciprocal(rank)
        mrr += q_mrr.item()
        num_q += 1

    mrr = mrr/num_q

    return mrr


def get_mrr_softmax(scores,target,query_indices,device):
    sort_per_query = {}
    mrr = 0.0
    num_q = 0.0
    unique_indices = torch.IntTensor(np.unique(query_indices.data)).to(device=device)
    for q in unique_indices:
       q_slice = (query_indices == q).reshape((-1))
       q_score = scores[q_slice][:,0]
       q_target = target[q_slice][:,0]
       q_score, sort_idx = torch.sort(q_score, descending=True)
       q_target = q_target[sort_idx]
       rank = q_target.nonzero()[0][0].float() + 1.0
       q_mrr = torch.reciprocal(rank)
       mrr += q_mrr.item()
       num_q += 1

    mrr = mrr/num_q

    return mrr


def get_optimizer(params, name="adam", checkpoint=None):
    if checkpoint:
        name = checkpoint['optim_method']
        if name == "sgd":
            optimizer = optim.SGD(params)
        elif name == "adam":
            #print(checkpoint['optimizer_state_dict'])
            optimizer = torch.optim.Adam(params)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.learning_rate
    else:    
        if name == "sgd":
            optimizer = optim.SGD(params, lr=args.learning_rate, momentum=args.momentum, nesterov=True)
        elif name == "adam":
            optimizer = torch.optim.Adam(params, lr=args.learning_rate, weight_decay=args.weight_decay)
        else:
            raise ValueError("Invalid Optimizer name.")

    return optimizer


if __name__ == "__main__":
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--optimizer", choices=["sgd", "adam"], type=str)
    parser.add_argument("--no_shuffle", default=False, action='store_true')
    parser.add_argument("--mis_class", type=int, default=9)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--penalize", default=False, action='store_true')
    parser.add_argument("--mode", choices=["train", "debug"], type=str, required=True)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--print_every", type=int, default=5)
    parser.add_argument("--no_gpu", default=False, action='store_true')
    parser.add_argument("--embed_dim", type=int, default=50)
    parser.add_argument("--retrain", default=False, action='store_true')
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--from_epoch", type=int, default=0)
    parser.add_argument("--decay_factor", type=float, default=0)


    # TODO: Subparsers for optim_params

    args = parser.parse_args()

    if args.embed_dim == 100:
        #model = m.Simp_Model_100_LSTM()
        #model = m.Simp_Model_100_LSTM_EncDec()
        #model = m.Simp_Model_100_GRU_EncDec()
        #model = m.Simp_Model_100_GRU_EncDec_samehidden()
        model = m.Simp_Model_100_GRU_EncDec_samehidden_Attn()
    elif args.embed_dim == 300:
        model = m.Simp_Model_300_LSTM()
    else:
        #model = m.Simp_Model_LSTM()
        model = m.Simp_Model_LSTM_EncDec()

    #model = m.Comp_Model_1()
    if not os.path.exists("models") and not args.mode == "debug":
        os.mkdir("models")
    
    #torch.cuda.set_device(1) 
    
    if args.penalize:
        p = "penalize"
    else:
        p = ""
    config_dict = {"start_lr":args.learning_rate,"momentum":args.momentum,"batch_size":args.batch_size,"mis_class":args.mis_class, "weight_decay":args.weight_decay,"penalize":args.penalize}
    if not args.retrain:
        model_name = "expt_" + "".join(model.__class__.__name__.split("_")) + "_lr" + str(args.learning_rate) + "_mcl" + str(args.mis_class) + "_bs" + str(args.batch_size) + "_wd" + str(args.weight_decay) + "_" + p + "_" + str(time.time())
        optimizer = get_optimizer(model.parameters(), name=args.optimizer)
    else:
        checkpoint = torch.load(args.model_path)
        if checkpoint:
            model_name = args.model_path.split("/")[-2]
            model.load_state_dict(checkpoint['model_state_dict'])

            optimizer = get_optimizer(model.parameters(), name=None, checkpoint=checkpoint)
            print(optimizer)
            #args.batch_size = checkpoint['batch_size']
            args.optimizer = checkpoint['optim_method']
            if 'mis_class' in checkpoint:
                args.mis_class = checkpoint['mis_class']

    model_name = "Testing Cuda"
    if args.mode=="debug":
        model_name = "expt_" +  "".join(model.__class__.__name__.split("_")) + "_debug" 
        if not os.path.exists("models/{}".format(model_name)):
            os.mkdir("models/{}".format(model_name))
    if not os.path.exists("models/{}".format(model_name)) and not args.mode == "debug":
        os.mkdir("models/{}".format(model_name))
    if not args.mode == "debug":
        with open("models/{}/train_config.json".format(model_name), "w") as f:
            json.dump(config_dict,f)
    train(model, num_epochs=args.num_epochs, mis_class=args.mis_class, optim_name=args.optimizer, optimizer=optimizer, batch_size=args.batch_size, model_name=model_name, print_every=args.print_every, use_shuffle=(not args.no_shuffle), checkpoint=None)
