import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

def getLastOutputs(outputs, lens):
    lasts = []
    for data, length in zip(outputs, lens):
        lasts.append(data[length-1])
    return torch.stack(lasts)

class SentenceEncoder(torch.nn.Module):
    def __init__(self, word_size, output_size, em_size=256, hidden_size=256, dropout=0.1, last_output = True):
        super(SentenceEncoder, self).__init__()

        self.last_output = last_output
        self.padding_value = 0
        
        self.embedding = torch.nn.Embedding(word_size, em_size, padding_idx=self.padding_value)
        self.rnn = torch.nn.GRU(em_size, hidden_size, batch_first=True)
        self.out = torch.nn.Linear(hidden_size, output_size)
        self.dropout = torch.nn.Dropout(dropout)
        
    def forward(self, inputs, hiddens=None):
        #get sort order 
        lens = torch.LongTensor([x.size(0) for x in inputs])
        sortedLen, fwdOrder = torch.sort(lens, descending=True)
        _, backOrder = torch.sort(fwdOrder)
        
        
        pad_seq = pad_sequence(inputs, batch_first=True, padding_value=self.padding_value)
        sort_seq = pad_seq[fwdOrder]
        
        sort_seq = self.embedding(sort_seq)
        sort_seq = self.dropout(sort_seq)
        packed_seq_input = pack_padded_sequence(sort_seq, lengths=sortedLen, batch_first=True)
        self.rnn.flatten_parameters()
        outputs, hiddens = self.rnn(packed_seq_input, hiddens)
        outputs, outLens = pad_packed_sequence(outputs, batch_first=True)
        
        if self.last_output:
            outputs = getLastOutputs(outputs, outLens)
        outputs = self.out(outputs)
        outputs = self.dropout(outputs)
        outputs = outputs[backOrder]
        return outputs, hiddens
    
class SentenceDecoder(torch.nn.Module):
    def __init__(self, word_size, feature_size, em_size=256, hidden_size=256, dropout=0.1, last_output = False):
        super(SentenceDecoder, self).__init__()

        self.last_output = last_output
        self.padding_value = 0
        
        self.embedding = torch.nn.Embedding(word_size, em_size, padding_idx=self.padding_value)
        self.rnn = torch.nn.GRU(em_size+feature_size, hidden_size, batch_first=True)
        self.out = torch.nn.Linear(hidden_size, word_size)
        self.dropout = torch.nn.Dropout(dropout)
        
    def forward(self, inputs, feature, hiddens=None):
        #get sort order 
        lens = torch.LongTensor([x.size(0) for x in inputs])
        sortedLen, fwdOrder = torch.sort(lens, descending=True)
        _, backOrder = torch.sort(fwdOrder)
        
        pad_seq = pad_sequence(inputs, batch_first=True, padding_value=self.padding_value)
        pad_seq = self.embedding(pad_seq)
        
        feature = torch.cat(feature, dim=1).unsqueeze(1).repeat(1, pad_seq.size(1), 1)
        pad_seq = torch.cat([pad_seq, feature], dim=2)
        
        sort_seq = pad_seq[fwdOrder]
        sort_seq = self.dropout(sort_seq)
        packed_seq_input = pack_padded_sequence(sort_seq, lengths=sortedLen, batch_first=True)
        self.rnn.flatten_parameters()
        outputs, hiddens = self.rnn(packed_seq_input, hiddens)
        outputs, outLens = pad_packed_sequence(outputs, batch_first=True)
        
        if self.last_output:
            outputs = getLastOutputs(outputs, outLens)
        outputs = self.out(outputs)
        outputs = self.dropout(outputs)
        outputs = outputs[backOrder]
        return outputs, hiddens    
    
class ImageEncoder(torch.nn.Module):
    def __init__(self, output_size, cnn_hidden=1024, dropout=0.1, cnn_type=models.resnet50, pretrained=False):
        super(ImageEncoder, self).__init__()
        self.cnn = cnn_type(pretrained=pretrained)
        if cnn_type == models.resnet50:
            self.cnn.fc = torch.nn.Linear(self.cnn.fc.in_features, cnn_hidden)
        self.out = torch.nn.Linear(cnn_hidden, output_size)
        self.dropout = torch.nn.Dropout(dropout)
        
    def forward(self, imgs):
        output = self.cnn(imgs)
        output = F.relu(self.dropout(output))
        output = self.out(output)
        
        output = F.relu(output)
        return output
    
cnnTransforms = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], 
                                                         [0.229, 0.224, 0.225])])

class Gesd(torch.nn.Module):
    def __init__(self, gamma=1, c=1, dim=1):
        super(Gesd, self).__init__()
        self.gamma = gamma
        self.c = c
        self.dim = dim

    def forward(self, f1, f2):
        l2_norm = ((f1-f2) ** 2).sum(dim=self.dim)
        euclidean = 1 / (1 + l2_norm)
        sigmoid  = 1 / (1 + torch.exp(-1 * self.gamma * ((f1*f2).sum(dim=self.dim) + self.c)))
        output = euclidean * sigmoid

        return output