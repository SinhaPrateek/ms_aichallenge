import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.autograd import Variable

class Simp_Model(nn.Module):
    def __init__(self):
        
        super().__init__()
        
        def init_weights(m):
            if type(m) == nn.Conv2d or type(m) == nn.Linear:
                
                torch.nn.init.xavier_uniform(m.weight)                  
                m.bias.data.fill_(0)

        
        class Flatten(nn.Module):
            def forward(self, x):
                N = x.shape[0] # read in N, C, H, W
                return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image
                
        self.seq_module_q = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=(3,10)), #4*10*41    #Conv2D(input_channel,out_channel,kernel,padding)
            nn.Tanh(),                           #nn.ReLU()
            nn.MaxPool2d((2,3),(2,3)),       #4*5*13                          #MaxPool2d(kernel_size, stride=None, padding=0)
            nn.Conv2d(4, 2, kernel_size=(2,4), padding=0),    #2*4*10
            nn.Tanh(),  # nn.ReLU()
            nn.MaxPool2d((2,2),(2,2)),  #2*2*5
            Flatten(),
            nn.Linear(2*10, 2*2),      #2*2           #torch.nn.Linear(in_features, out_features, bias=True)
            nn.Tanh()                           #nn.ReLU()
        )
        self.seq_module_q.apply(init_weights)
        
        self.seq_module_p = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=(5, 10), padding=0),  #4*46*41  #Conv2D(input_channel,out_channel,kernel,padding)
            nn.Tanh(),                           #nn.ReLU()
            nn.MaxPool2d((5,5),(5,5)),                    #4*9*8             #MaxPool2d(kernel_size, stride=None, padding=0)
            nn.Conv2d(4, 2, kernel_size=(3,3), padding=0), #2*7*6
            nn.Tanh(),                           #nn.ReLU()
            nn.MaxPool2d((2,2),(2,2)),                 #2*3*3
            Flatten(),
            nn.Linear(2*9, 2*2) ,   #2*2      #torch.nn.Linear(in_features, out_features, bias=True)
            nn.Tanh()  # nn.ReLU()
        )
        self.seq_module_p.apply(init_weights)
        
        self.self_module_final = nn.Sequential(
            nn.Linear(4,2),
            nn.Softmax(dim=1)
        )
        self.self_module_final.apply(init_weights)
        
    def forward(self, q, p):
        for module in self.seq_module_q:
            q = module(q)
        for module in self.seq_module_p:
            p = module(p)
        scores = torch.mul(q, p)

        for module in self.self_module_final:
            scores = module(scores)
        return scores

class Simp_Model_Connected(nn.Module):
    def __init__(self):
        
        super().__init__()
        def init_weights(m):
            if type(m) == nn.Conv2d or type(m) == nn.Linear:
                torch.nn.init.xavier_uniform(m.weight)                  
                m.bias.data.fill_(0)
        
        class Flatten(nn.Module):
            def forward(self, x):
                N = x.shape[0] # read in N, C, H, W
                return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image
                
        self.seq_module_q = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=(3,10)), #4*10*41    #Conv2D(input_channel,out_channel,kernel,padding)
            nn.Tanh(),                           #nn.ReLU()
            nn.MaxPool2d((2,3),(2,3)),       #4*5*13                          #MaxPool2d(kernel_size, stride=None, padding=0)
            nn.Conv2d(4, 2, kernel_size=(2,4), padding=0),    #2*4*10
            nn.Tanh(),  # nn.ReLU()
            nn.MaxPool2d((2,2),(2,2)),  #2*2*5
            Flatten(),
            nn.Linear(2*10, 2*2),      #2*2           #torch.nn.Linear(in_features, out_features, bias=True)
            nn.Tanh()                           #nn.ReLU()
        )
        self.seq_module_q.apply(init_weights)
        
        self.seq_module_p = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=(5, 10), padding=0),  #4*46*41  #Conv2D(input_channel,out_channel,kernel,padding)
            nn.Tanh(),                           #nn.ReLU()
            nn.MaxPool2d((5,5),(5,5)),                    #4*9*8             #MaxPool2d(kernel_size, stride=None, padding=0)
            nn.Conv2d(4, 2, kernel_size=(3,3), padding=0), #2*7*6
            nn.Tanh(),                           #nn.ReLU()
            nn.MaxPool2d((2,2),(2,2)),                 #2*3*3
            Flatten(),
            nn.Linear(2*9, 2*2) ,   #2*2      #torch.nn.Linear(in_features, out_features, bias=True)
            nn.Tanh()  # nn.ReLU()
        )
        self.seq_module_p.apply(init_weights)
        
        self.q_p_cat = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(2,3), padding = (0,1)),  #1*1*4
            nn.Tanh(),
            Flatten(),
        )

        self.q_p_cat.apply(init_weights)

        self.self_module_final = nn.Sequential(
            nn.Linear(4,2),
            nn.Softmax(dim=1)
        )
        self.self_module_final.apply(init_weights)
        
    def forward(self, q, p):
        for module in self.seq_module_q:
            q = module(q)
        for module in self.seq_module_p:
            p = module(p)
        q_p_cat = torch.cat((q.unsqueeze(1),p.unsqueeze(1)),1)
        q_p_cat = q_p_cat.unsqueeze(1)
        p_new = self.q_p_cat(q_p_cat)
        scores = torch.mul(q, p_new)

        for module in self.self_module_final:
            scores = module(scores)
        return scores

class Simp_Model_Siamese(nn.Module):
    def __init__(self):
        
        super().__init__()
        def init_weights(m):
            if type(m) == nn.Conv2d or type(m) == nn.Linear:
                torch.nn.init.xavier_uniform(m.weight)                  
                m.bias.data.fill_(0)
        
        class Flatten(nn.Module):
            def forward(self, x):
                N = x.shape[0] # read in N, C, H, W
                return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image
                
        self.seq_module_q = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=(3,10)), #4*10*41    #Conv2D(input_channel,out_channel,kernel,padding)
            nn.Tanh(),                           #nn.ReLU()
            nn.MaxPool2d((2,3),(2,3)),       #4*5*13                          #MaxPool2d(kernel_size, stride=None, padding=0)
            nn.Conv2d(4, 2, kernel_size=(2,4), padding=0),    #2*4*10
            nn.Tanh(),  # nn.ReLU()
            nn.MaxPool2d((2,2),(2,2)),  #2*2*5
            Flatten(),
            nn.Linear(2*10, 2*2),      #2*2           #torch.nn.Linear(in_features, out_features, bias=True)
            nn.Tanh()                           #nn.ReLU()
        )
        self.seq_module_q.apply(init_weights)
        
        self.seq_module_p = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=(5, 10), padding=0),  #4*46*41  #Conv2D(input_channel,out_channel,kernel,padding)
            nn.Tanh(),                           #nn.ReLU()
            nn.MaxPool2d((5,5),(5,5)),                    #4*9*8             #MaxPool2d(kernel_size, stride=None, padding=0)
            nn.Conv2d(4, 2, kernel_size=(3,3), padding=0), #2*7*6
            nn.Tanh(),                           #nn.ReLU()
            nn.MaxPool2d((2,2),(2,2)),                 #2*3*3
            Flatten(),
            nn.Linear(2*9, 2*2) ,   #2*2      #torch.nn.Linear(in_features, out_features, bias=True)
            nn.Tanh()  # nn.ReLU()
        )
        self.seq_module_p.apply(init_weights)
        
        self.self_module_final = nn.Sequential(
            nn.Linear(4,2),
            nn.Softmax(dim=1)
        )
        self.self_module_final.apply(init_weights)
        
    def forward(self, q, passages):
        for module in self.seq_module_q:
            q = module(q)
        mis_class = list(passages.shape)[1]
        n = list(passages.shape)[0]
        scores = []

        for i in range(mis_class):
            p_eval = passages[:,i,:,:,:]
            for module in self.seq_module_p:
                p_eval = module(p_eval)
            score = torch.mul(q,p_eval)
            score = self.self_module_final(score)
            scores.append(score)
        
        return torch.stack(scores,1)


class Simp_Model_LSTM(nn.Module):
    def __init__(self):
        
        super().__init__()
        def init_weights(m):
            for n,p in m.named_parameters():
                
                if 'weight' in n:
                    torch.nn.init.xavier_uniform(p)                  
                else:
                    p.data.fill_(0)
        
        class Flatten(nn.Module):
            def forward(self, x):
                N = x.shape[0] # read in N, C, H, W
                return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image
        
        self.seq_module_q = nn.Sequential(
            nn.LSTM(50,25,batch_first = True),
            nn.LSTM(25,10,batch_first = True),
            nn.LSTM(10,4,batch_first = True)
        )

        self.seq_module_q.apply(init_weights)
        
        self.seq_module_p = nn.Sequential(
            nn.LSTM(50,25,batch_first = True),
            nn.LSTM(25,10,batch_first = True),
            nn.LSTM(10,4,batch_first = True)
        )
        self.seq_module_p.apply(init_weights)
        
        self.self_module_final = nn.Sequential(
            nn.Linear(4,2),
            nn.Softmax(dim=1)
        )
        self.self_module_final.apply(init_weights)
        
    def forward(self, q, p):
        q = q.view(-1,12,50)
        for module in self.seq_module_q:
            q,(h_q,c_q) = module(q)
        p = p.view(-1,50,50)
        for module in self.seq_module_p:
            p,(h_p,c_p) = module(p)
        scores = torch.mul(h_q.view(-1,4), h_p.view(-1,4))

        for module in self.self_module_final:
            scores = module(scores)
        return scores

class Simp_Model_LSTM_EncDec(nn.Module):
    def __init__(self):
        
        super().__init__()
        def init_weights(m):
            for n,p in m.named_parameters():
                
                if 'weight' in n:
                    torch.nn.init.xavier_uniform(p)                  
                else:
                    p.data.fill_(0)
        
        class Flatten(nn.Module):
            def forward(self, x):
                N = x.shape[0] # read in N, C, H, W
                return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image
        
        self.seq_module_q = nn.Sequential(
            nn.LSTM(50,25,batch_first = True),
            nn.LSTM(25,10,batch_first = True),
            nn.LSTM(10,4,batch_first = True)
        )

        self.seq_module_q.apply(init_weights)
        
        self.seq_module_p = nn.Sequential(
            nn.LSTM(50,25,batch_first = True),
            nn.LSTM(25,10,batch_first = True),
            nn.LSTM(10,4,batch_first = True)
        )
        self.seq_module_p.apply(init_weights)
        
        self.self_module_final = nn.Sequential(
            nn.Linear(4,2),
            nn.Softmax(dim=1)
        )
        self.self_module_final.apply(init_weights)
        
    def forward(self, q, p):
        q = q.view(-1,12,50)
        h_q_list = []
        c_q_list = []
        for module in self.seq_module_q:
            q,(h_q,c_q) = module(q)
            h_q_list.append(h_q)
            c_q_list.append(c_q)
        p = p.view(-1,50,50)
        i = 0
        for module in self.seq_module_p:
            p,(h_p,c_p) = module(p,(h_q_list[i],c_q_list[i]))
            i+=1
        
        scores = h_p.view(-1,4)  
        for module in self.self_module_final:
            scores = module(scores)

        return scores



class Simp_Model_100_LSTM(nn.Module):
    def __init__(self):
        
        super().__init__()
        def init_weights(m):
            for n,p in m.named_parameters():
                
                if 'weight' in n:
                    torch.nn.init.xavier_uniform(p)                  
                else:
                    p.data.fill_(0)
        
        class Flatten(nn.Module):
            def forward(self, x):
                N = x.shape[0] # read in N, C, H, W
                return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image
        
        self.seq_module_q = nn.Sequential(
            nn.LSTM(100,50,batch_first = True),
            nn.LSTM(50,25,batch_first = True),
            nn.LSTM(25,10,batch_first = True),
            nn.LSTM(10,4,batch_first = True)
        )

        self.seq_module_q.apply(init_weights)
        
        self.seq_module_p = nn.Sequential(
            nn.LSTM(100,50,batch_first = True),
            nn.LSTM(50,25,batch_first = True),
            nn.LSTM(25,10,batch_first = True),
            nn.LSTM(10,4,batch_first = True)
        )
        self.seq_module_p.apply(init_weights)
        
        self.self_module_final = nn.Sequential(
            nn.Linear(4,2),
            nn.Softmax(dim=1)
        )
        self.self_module_final.apply(init_weights)
        
    def forward(self, q, p):
        q = q.view(-1,12,100)
        for module in self.seq_module_q:
            q,(h_q,c_q) = module(q)
        p = p.view(-1,50,100)
        for module in self.seq_module_p:
            p,(h_p,c_p) = module(p)
        scores = torch.mul(h_q.view(-1,4), h_p.view(-1,4))

        for module in self.self_module_final:
            scores = module(scores)
        return scores

class Simp_Model_100_LSTM_EncDec(nn.Module):
    def __init__(self):
        
        super().__init__()
        def init_weights(m):
            for n,p in m.named_parameters():
                
                if 'weight' in n:
                    torch.nn.init.xavier_uniform(p)                  
                else:
                    p.data.fill_(0)
        
        class Flatten(nn.Module):
            def forward(self, x):
                N = x.shape[0] # read in N, C, H, W
                return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image
        
        self.seq_module_q = nn.Sequential(
            nn.LSTM(100,50,batch_first = True),
            nn.LSTM(50,25,batch_first = True),
            nn.LSTM(25,10,batch_first = True),
            nn.LSTM(10,4,batch_first = True)
        )

        self.seq_module_q.apply(init_weights)
        
        self.seq_module_p = nn.Sequential(
            nn.LSTM(100,50,batch_first = True),
            nn.LSTM(50,25,batch_first = True),
            nn.LSTM(25,10,batch_first = True),
            nn.LSTM(10,4,batch_first = True)
        )
        self.seq_module_p.apply(init_weights)
        
        self.self_module_final = nn.Sequential(
            nn.Linear(4,2),
            nn.Softmax(dim=1)
        )
        self.self_module_final.apply(init_weights)
        
    def forward(self, q, p):
        q = q.view(-1,12,100)
        h_q_list = []
        c_q_list = []
        for module in self.seq_module_q:
            q,(h_q,c_q) = module(q)
            h_q_list.append(h_q)
            c_q_list.append(c_q)
        p = p.view(-1,50,100)
        i = 0
        for module in self.seq_module_p:
            p,(h_p,c_p) = module(p,(h_q_list[i],c_q_list[i]))
            i+=1
        #scores = torch.mul(h_q.view(-1,4), h_p.view(-1,4))
        scores = h_p.view(-1,4)  
        for module in self.self_module_final:
            scores = module(scores)
        return scores

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size, self.hidden_size)
        #self.v = nn.Parameter(torch.rand(hidden_size))
        #stdv = 1. / math.sqrt(self.v.size(0))
        #self.v.data.uniform_(-stdv, stdv)
        self.v = nn.Parameter(torch.FloatTensor(1,hidden_size))

    def forward(self, hidden, encoder_outputs):
        timestep = encoder_outputs.size(1)
        #h = hidden.repeat(timestep, 1, 1).transpose(0,1)
        #encoder_outputs = encoder_outputs  # [B*T*H]
        attn_energies = []
        for i in range(12):
            attn_energies.append(self.score(hidden,encoder_outputs[:,i,:]))
        out = F.softmax(torch.stack(attn_energies).transpose(0,1),dim=1).unsqueeze(1)
        return out

    def score(self, hidden, encoder_output):
        # [B*T*2H]->[B*T*H]
        energy = self.attn(encoder_output)
        #energy = energy.transpose(1, 2)  # [B*H*T]
        #v = self.v.repeat(encoder_output.size(0),1)  # [B*1*H]
        #energy = torch.bmm(v, energy)  # [B*1*T]
        energy = torch.sum(hidden*energy,dim=1)
        return energy

class Simp_Model_100_GRU_EncDec_samehidden_Attn(nn.Module):
    def __init__(self):
        
        super().__init__()
        def init_weights(m):
            for n,p in m.named_parameters():
                
                if 'weight' in n:
                    torch.nn.init.xavier_uniform(p)                  
                else:
                    p.data.fill_(0)
        
        self.attn_dec = Attention(100)
        
        self.emb_layer = nn.Linear(100,50)
        self.emb_layer.apply(init_weights)

        self.seq_module_q = nn.GRU(100,100,2,batch_first = True)
            

        self.seq_module_q.apply(init_weights)
        
        self.seq_module_p = nn.GRU(200,100,2,batch_first = True)
        self.seq_module_p.apply(init_weights)
        
        self.out_context = nn.Linear(200,50)

        self.self_module_final = nn.Sequential(
            nn.Linear(50,25),
            nn.ReLU(),
            nn.Linear(25,10),
            nn.ReLU(),
            nn.Linear(10,4),
            nn.ReLU(),
            nn.Linear(4,2),
            nn.Softmax(dim=1)
        )

        self.self_module_final.apply(init_weights)
        
    def forward(self, q, p):
        q = q.view(-1,12,100)
        #p = p.view(-1,50,100)
        #q_input = []
        #for i in range(q.shape[1]):
         #   q_input.append(self.emb_layer(q[:,i,:]))
        #q_input = torch.stack(q_input).view(-1,q.shape[1],50)
        #p_input = []
        #print(p.shape)
        #for i in range(p.shape[1]):
         #   p_input.append(self.emb_layer(p[:,i,:]))
        #p_input = torch.stack(p_input).view(-1,p.shape[1],50)
        #print(q_input.shape)
        #print(p_input.shape)
        q,h_q = self.seq_module_q(q)
        encoder_outputs = q.clone()
        last_hidden = h_q.clone()
        outputs = []
        p = p.view(-1,50,100)
        for i in range(50):
            input_word = p[:,i,:]
            attn_weights = self.attn_dec(last_hidden[-1],encoder_outputs)
            context = attn_weights.bmm(q)
            #print("Context",context.shape)
            #print("Input_word",input_word.shape)
            gru_p_input = torch.cat([input_word.unsqueeze(1),context],2)
            output,hidden = self.seq_module_p(gru_p_input,last_hidden)
            output = output.squeeze(1)
            context = context.squeeze(1)
            output = self.out_context(torch.cat([output,context],1))
            output = F.relu(output)
            #print("output",output.shape)
            last_hidden = hidden
            outputs.append(output)

        #print(outputs[0].shape)
        #outputs = torch.stack(output)
        scores = outputs[-1]
        for module in self.self_module_final:
            scores = module(scores)

        return scores


class Simp_Model_100_GRU_EncDec(nn.Module):
    def __init__(self):
        
        super().__init__()
        def init_weights(m):
            for n,p in m.named_parameters():
                
                if 'weight' in n:
                    torch.nn.init.xavier_uniform(p)                  
                else:
                    p.data.fill_(0)
        
        class Flatten(nn.Module):
            def forward(self, x):
                N = x.shape[0] # read in N, C, H, W
                return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image
        
        self.seq_module_q = nn.Sequential(
            nn.GRU(100,50,batch_first = True),
            nn.GRU(50,25,batch_first = True),
            nn.GRU(25,10,batch_first = True),
            nn.GRU(10,4,batch_first = True)
        )

        #self.seq_module_q.apply(init_weights)
        
        self.seq_module_p = nn.Sequential(
            nn.GRU(100,50,batch_first = True),
            nn.GRU(50,25,batch_first = True),
            nn.GRU(25,10,batch_first = True),
            nn.GRU(10,4,batch_first = True)
        )
        #self.seq_module_p.apply(init_weights)
        
        self.self_module_final = nn.Sequential(
            nn.Linear(4,2),
            nn.Softmax(dim=1)
        )
        self.self_module_final.apply(init_weights)
        
    def forward(self, q, p):
        q = q.view(-1,12,100)
        h_q_list = []
        for module in self.seq_module_q:
            q,h_q = module(q)
            h_q_list.append(h_q)
        p = p.view(-1,50,100)
        i = 0
        for module in self.seq_module_p:
            p,h_p = module(p,h_q_list[i])
            i+=1
        #scores = torch.mul(h_q.view(-1,4), h_p.view(-1,4))
        scores = h_p.view(-1,4)  
        for module in self.self_module_final:
            scores = module(scores)
        return scores

class Simp_Model_100_GRU_EncDec_samehidden(nn.Module):
    def __init__(self):
        
        super().__init__()
        def init_weights(m):
            for n,p in m.named_parameters():
                
                if 'weight' in n:
                    torch.nn.init.xavier_uniform(p)                  
                else:
                    p.data.fill_(0)
        
        class Flatten(nn.Module):
            def forward(self, x):
                N = x.shape[0] # read in N, C, H, W
                return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image
       
        self.seq_module_q = nn.Sequential(
            nn.GRU(100,50,batch_first = True),
            nn.GRU(50,50,batch_first = True),
        )

        self.seq_module_q.apply(init_weights)
        
        self.seq_module_p = nn.Sequential(
            nn.GRU(100,50,batch_first = True),
            nn.GRU(50,50,batch_first = True),
        )
        self.seq_module_p.apply(init_weights)
        
        self.self_module_final = nn.Sequential(
            nn.Linear(50,25),
            nn.ReLU(),
            nn.Linear(25,10),
            nn.ReLU(),
            nn.Linear(10,4),
            nn.ReLU(),
            nn.Linear(4,2),
            nn.Softmax(dim=1)
        )
        self.self_module_final.apply(init_weights)
        
    def forward(self, q, p):
        q = q.view(-1,12,100)
        h_q_list = []
        for module in self.seq_module_q:
            q,h_q = module(q)
            h_q_list.append(h_q)
        p = p.view(-1,50,100)
        i = 0
        for module in self.seq_module_p:
            p,h_p = module(p,h_q_list[i])
            i+=1
        #scores = torch.mul(h_q.view(-1,4), h_p.view(-1,4))
        scores = h_p.view(-1,50)  
        for module in self.self_module_final:
            scores = module(scores)
        return scores



class Simp_Model_300_LSTM(nn.Module):
    def __init__(self):
        
        super().__init__()
        def init_weights(m):
            for n,p in m.named_parameters():
                
                if 'weight' in n:
                    torch.nn.init.xavier_uniform(p)                  
                else:
                    p.data.fill_(0)
        
        class Flatten(nn.Module):
            def forward(self, x):
                N = x.shape[0] # read in N, C, H, W
                return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image
        
        self.seq_module_q = nn.Sequential(
            nn.LSTM(300,200,batch_first = True),
            nn.LSTM(200,100,batch_first = True),
            nn.LSTM(100,50,batch_first = True),
            nn.LSTM(50,25,batch_first = True),
            nn.LSTM(25,10,batch_first = True),
            nn.LSTM(10,4,batch_first = True)
        )

        self.seq_module_q.apply(init_weights)
        
        self.seq_module_p = nn.Sequential(
            nn.LSTM(300,200,batch_first = True),
            nn.LSTM(200,100,batch_first = True),
            nn.LSTM(100,50,batch_first = True),
            nn.LSTM(50,25,batch_first = True),
            nn.LSTM(25,10,batch_first = True),
            nn.LSTM(10,4,batch_first = True)
        )
        self.seq_module_p.apply(init_weights)
        
        self.self_module_final = nn.Sequential(
            nn.Linear(4,2),
            nn.Softmax(dim=1)
        )
        self.self_module_final.apply(init_weights)
        
    def forward(self, q, p):
        q = q.view(-1,12,300)
        for module in self.seq_module_q:
            q,(h_q,c_q) = module(q)
        p = p.view(-1,50,300)
        for module in self.seq_module_p:
            p,(h_p,c_p) = module(p)
        scores = torch.mul(h_q.view(-1,4), h_p.view(-1,4))

        for module in self.self_module_final:
            scores = module(scores)
        return scores


class Simp_Model_LSTM_Siamese(nn.Module):
    def __init__(self):
        
        super().__init__()
        def init_weights(m):
            for n,p in m.named_parameters():
                
                if 'weight' in n:
                    torch.nn.init.xavier_uniform(p)                  
                else:
                    p.data.fill_(0)
        
        class Flatten(nn.Module):
            def forward(self, x):
                N = x.shape[0] # read in N, C, H, W
                return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image
        
        self.seq_module_q = nn.Sequential(
            nn.LSTM(50,25,batch_first = True),
            nn.LSTM(25,10,batch_first = True),
            nn.LSTM(10,4,batch_first = True)
        )

        self.seq_module_q.apply(init_weights)
        
        self.seq_module_p = nn.Sequential(
            nn.LSTM(50,25,batch_first = True),
            nn.LSTM(25,10,batch_first = True),
            nn.LSTM(10,4,batch_first = True)
        )
        self.seq_module_p.apply(init_weights)
        
        self.self_module_final = nn.Sequential(
            nn.Linear(4,2),
            nn.Softmax(dim=1)
        )
        self.self_module_final.apply(init_weights)
        
    def forward(self, q, passages):
        q = q.view(-1,12,50)
        for module in self.seq_module_q:
            q,(h_q,c_q) = module(q)
        
        mis_class = list(passages.shape)[1]

        scores = []
        for i in range(mis_class):
            p_eval = passages[:,i,:,:,:].view(-1,50,50)
            for module in self.seq_module_p:
                p_eval,(h_p_eval,c_p_eval) = module(p_eval)
            score = torch.mul(h_q.view(-1,4),h_p_eval.view(-1,4))
            score = self.self_module_final(score)
            scores.append(score)

        return torch.stack(scores,1)



class Simp_Model_100(nn.Module):
    def __init__(self):
        
        super().__init__()
        def init_weights(m):
            if type(m) == nn.Conv2d or type(m) == nn.Linear:
                torch.nn.init.xavier_uniform(m.weight)                  
                m.bias.data.fill_(0)
        
        class Flatten(nn.Module):
            def forward(self, x):
                N = x.shape[0] # read in N, C, H, W
                return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image
                
        self.seq_module_q = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=(3,10)),  #Conv2D(input_channel,out_channel,kernel,padding)
            nn.Tanh(),                           #nn.ReLU()
            nn.MaxPool2d((2,3),(2,3)),                                 #MaxPool2d(kernel_size, stride=None, padding=0)
            nn.Conv2d(4, 2, kernel_size=(2,4), padding=0),
            nn.Tanh(),  # nn.ReLU()
            nn.MaxPool2d((2,2),(2,2)),
            Flatten(),
            nn.Linear(2*26, 2*5),    #torch.nn.Linear(in_features, out_features, bias=True)
            nn.Tanh(),
            nn.Linear(2*5,2*2),
            nn.Tanh()                           #nn.ReLU()
        )
        self.seq_module_q.apply(init_weights)
        
        self.seq_module_p = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=(5, 10), padding=0),  #Conv2D(input_channel,out_channel,kernel,padding)
            nn.Tanh(),                           #nn.ReLU()
            nn.MaxPool2d((5,5),(5,5)),                                 #MaxPool2d(kernel_size, stride=None, padding=0)
            nn.Conv2d(4, 2, kernel_size=(3,3), padding=0),
            nn.Tanh(),                           #nn.ReLU()
            nn.MaxPool2d((2,2),(2,2)),
            Flatten(),
            nn.Linear(2*24, 2*5) ,                  #torch.nn.Linear(in_features, out_features, bias=True)
            nn.Tanh(),  # nn.ReLU()
            nn.Linear(2*5,2*2),
            nn.Tanh()         
        )
        self.seq_module_p.apply(init_weights)
        
        self.self_module_final = nn.Sequential(
            nn.Linear(4,2),
            nn.Softmax(dim=1)
        )
        self.self_module_final.apply(init_weights)
        
    def forward(self, q, p):
        for module in self.seq_module_q:
            q = module(q)
        for module in self.seq_module_p:
            p = module(p)
        scores = torch.mul(q, p)

        for module in self.self_module_final:
            scores = module(scores)
        return scores





class Comp_Model_1(nn.Module):
    def __init__(self):
        
        super().__init__()
        def init_weights(m):
            if type(m) == nn.Conv2d or type(m) == nn.Linear:
                torch.nn.init.xavier_uniform(m.weight)                  
                m.bias.data.fill_(0)
        
        class Flatten(nn.Module):
            def forward(self, x):
                N = x.shape[0] # read in N, C, H, W
                return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image
                
        self.seq_module_q = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=(3,7)),  #6*10*44  #Conv2D(input_channel,out_channel,kernel,padding)
            nn.Tanh(),                           #nn.ReLU()
            nn.MaxPool2d((2,2),(2,2)),           #6*5*22                      #MaxPool2d(kernel_size, stride=None, padding=0)
            nn.Conv2d(6, 4, kernel_size=(2,4), padding=0),       #4*4*19
            nn.Tanh(),  # nn.ReLU()
            nn.MaxPool2d((2,2),(2,2)),             #4*2*9
            nn.Conv2d(4, 2, kernel_size=(1,2), padding=0),       #2*2*8
            nn.Tanh(),  # nn.ReLU()
            nn.MaxPool2d((1,2),(1,2)),             #2*2*4
            Flatten(),
            nn.Linear(2*8, 2*2),                 #torch.nn.Linear(in_features, out_features, bias=True)
            nn.Tanh()                  #2*2         #nn.ReLU()
        )
        self.seq_module_q.apply(init_weights)
        
        self.seq_module_p = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=(3, 7), padding=0), #6*48*44  #Conv2D(input_channel,out_channel,kernel,padding)
            nn.Tanh(),                           #nn.ReLU()
            nn.MaxPool2d((3,3),(3,3)),      #6*16*14                           #MaxPool2d(kernel_size, stride=None, padding=0)
            nn.Conv2d(6, 4, kernel_size=(3,3), padding=0),    #4*14*12
            nn.Tanh(),                           #nn.ReLU()
            nn.MaxPool2d((2,2),(2,2)),     #4*7*6
            nn.Conv2d(4, 2, kernel_size=(2,2), padding=0),       #2*6*5
            nn.Tanh(),  # nn.ReLU()
            nn.MaxPool2d((2,2),(2,2)),             #2*3*2
            Flatten(),
            nn.Linear(2*6, 2*2) ,                  #torch.nn.Linear(in_features, out_features, bias=True)
            nn.Tanh()  # nn.ReLU()   #2*2
        )
        self.seq_module_p.apply(init_weights)
        
        self.self_module_final = nn.Sequential(
            nn.Linear(4,3),
            nn.Linear(3,2),
            nn.Softmax(dim=1)
        )
        self.self_module_final.apply(init_weights)
        
    def forward(self, q, p):
        for module in self.seq_module_q:
            q = module(q)
        for module in self.seq_module_p:
            p = module(p)
        scores = torch.mul(q, p)

        for module in self.self_module_final:
            scores = module(scores)
        return scores


class Comp_Model_100(nn.Module):
    def __init__(self):
        
        super().__init__()
        def init_weights(m):
            if type(m) == nn.Conv2d or type(m) == nn.Linear:
                torch.nn.init.xavier_uniform(m.weight)                  
                m.bias.data.fill_(0)
        
        class Flatten(nn.Module):
            def forward(self, x):
                N = x.shape[0] # read in N, C, H, W
                return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image
                
        self.seq_module_q = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=(3,7)),  #6*10*95  #Conv2D(input_channel,out_channel,kernel,padding)
            nn.Tanh(),                           #nn.ReLU()
            nn.MaxPool2d((2,2),(2,2)),           #6*5*47                      #MaxPool2d(kernel_size, stride=None, padding=0)
            nn.Conv2d(6, 4, kernel_size=(2,4), padding=0),       #4*4*44
            nn.Tanh(),  # nn.ReLU()
            nn.MaxPool2d((2,2),(2,2)),             #4*2*22
            nn.Conv2d(4, 2, kernel_size=(1,2), padding=0),       #2*2*21
            nn.Tanh(),  # nn.ReLU()
            nn.MaxPool2d((1,2),(1,2)),             #2*2*10
            Flatten(),
            nn.Linear(2*20, 2*10),                 #torch.nn.Linear(in_features, out_features, bias=True)
            nn.Tanh(),           #2*2         #nn.ReLU()
            nn.Linear(2*10, 2*5),                 #torch.nn.Linear(in_features, out_features, bias=True)
            nn.Tanh(),
            nn.Linear(2*5, 2*2),                 #torch.nn.Linear(in_features, out_features, bias=True)
            nn.Tanh()             
        )
        self.seq_module_q.apply(init_weights)
        
        self.seq_module_p = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=(3, 7), padding=0), #6*48*95  #Conv2D(input_channel,out_channel,kernel,padding)
            nn.Tanh(),                           #nn.ReLU()
            nn.MaxPool2d((3,3),(3,3)),      #6*16*31                           #MaxPool2d(kernel_size, stride=None, padding=0)
            nn.Conv2d(6, 4, kernel_size=(3,3), padding=0),    #4*14*28
            nn.Tanh(),                           #nn.ReLU()
            nn.MaxPool2d((2,2),(2,2)),     #4*7*14
            nn.Conv2d(4, 2, kernel_size=(2,2), padding=0),       #2*6*13
            nn.Tanh(),  # nn.ReLU()
            nn.MaxPool2d((2,2),(2,2)),             #2*3*6
            Flatten(),
            nn.Linear(2*18, 2*9) ,                  #torch.nn.Linear(in_features, out_features, bias=True)
            nn.Tanh(),  # nn.ReLU()   #2*2
            nn.Linear(2*9, 2*4),                 #torch.nn.Linear(in_features, out_features, bias=True)
            nn.Tanh(),
            nn.Linear(2*4, 2*2),                 #torch.nn.Linear(in_features, out_features, bias=True)
            nn.Tanh()             
        )
        self.seq_module_p.apply(init_weights)
        
        self.self_module_final = nn.Sequential(
            nn.Linear(4,2),
            nn.Softmax(dim=1)
        )
        self.self_module_final.apply(init_weights)
        
    def forward(self, q, p):
        for module in self.seq_module_q:
            q = module(q)
        for module in self.seq_module_p:
            p = module(p)
        scores = torch.mul(q, p)

        for module in self.self_module_final:
            scores = module(scores)
        return scores

