import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

class G_network(nn.Module):
    def __init__(self, hidden_size, gpu=False):
        super(G_network, self).__init__()
        self.hidden_size = hidden_size
        self.gpu_ = gpu

        self.recurrent = nn.GRU(1, hidden_size)
        self.dense = nn.Linear(hidden_size, 1)

    def init_hidden(self, batch_size=1):
        h = torch.zeros(1, batch_size, self.hidden_size, dtype=torch.float64)
        if self.gpu_:
            return h.cuda()
        else:
            return h

    def forward(self, inp, hidden):
        if self.gpu_:
            inp = inp.cuda()
        out, hidden = self.recurrent(inp, hidden)
        out = F.relu(self.dense(out.view(-1, self.hidden_size))) #out = F.relu(self.dense(hidden))
        return out, hidden

    def sample(self, num_samples, max_length, start_value):
        samples = torch.zeros(num_samples, max_length).type(torch.DoubleTensor)
        h = self.init_hidden(num_samples)
        inp = torch.DoubleTensor([[start_value] * num_samples]).resize_(1,num_samples,1)
        print(inp.dtype)

        if self.gpu_:
            samples = samples.cuda()
            inp = inp.cuda()

        for i in range(max_length):
            out, h = self.forward(inp, h)
            samples[:, i] = out.view(-1).data
            inp = out.view(1,num_samples,1)

        return samples

    def PGLoss(self, inp, target, reward):
        batch_size, seq_len, _ = inp.size()
        inp = inp.permute(1, 0, 2)
        target = target.permute(1, 0, 2)
        h = self.init_hidden(batch_size)
        loss = 0

        for i in range(seq_len):
            out, h = self.forward(inp[i].view(1,50,1), h)
            for j in range(batch_size):
                loss += (out[j] - target[i,j])**2 * reward[j]

        return loss / batch_size

class D_network(nn.Module):
    def __init__(self, dict_size, embedding_dim, hidden_size, gpu=False):
        super(D_network, self).__init__()

        self.hidden_size = dict_size, embedding_dim, hidden_size
        self.gpu_ = gpu

        self.recurrent = nn.GRU(1, hidden_size, num_layers=2, bidirectional=True)
        self.gru2hidden = nn.Linear(2*2*hidden_size, hidden_size)
        self.dropout = nn.Dropout(p=0.3)
        self.dense = nn.Linear(hidden_size, 1)

    def init_hidden(self, batch_size):
        h = autograd.Variable(torch.zeros(2*2*1, batch_size, self.hidden_size))
        if self.gpu_:
            return h.cuda()
        else:
            return h

    def forward(self, x, h):
        if self.gpu_:
            x = x.cuda()
        _, h = self.recurrent(x, h)
        h = h.permute(1,0,2).contiguous()
        out = self.gru2hidden(h.view(-1, 4*self.hidden_size))
        out = torch.tanh(out)
        out = self.dropout(out)
        out = self.dense(out)
        out = torch.sigmoid(out)
        return out

    def batchClassify(self, x):
        h = self.init_hidden(x.size()[0])
        out = self.forward(x, h)
        return out.view(-1)

    def batchBCELoss(self, x, target):
        loss_fn = nn.BCELoss()
        h = self.init_hidden(x.size()[0])
        out = self.forward(x, h)
        return loss_fn(out, target)
