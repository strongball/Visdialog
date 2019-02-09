import torch
from torch.nn.utils.rnn import pad_sequence

def setData(data, lang, device):
    seqs_t = []
    for seq in data:
        seqs_t.append(torch.LongTensor(seq).to(device))
        
    decode_t = trainSentence(seqs_t, lang["<SOS>"], lang["<EOS>"], device)
    decode_t["in"] = pad_sequence(decode_t["in"], batch_first=True)
    decode_t["out"] = pad_sequence(decode_t["out"], batch_first=True)
    encode_t = pad_sequence(seqs_t, batch_first=True)
    return encode_t, decode_t

def collate_fn(batch):
    return batch

def trainSentence(sents, SOS, EOS, device):
    train_sents = {}
    train_sents["in"] = []
    train_sents["out"] = []    
    
    for sent in sents:
        train_sents["in"].append(torch.cat([sent.new([SOS]), sent]).to(device))
        train_sents["out"].append(torch.cat([sent, sent.new([EOS])]).to(device))
    return train_sents

def predit(model, device, lang, image, question):    
    image_t = image.unsqueeze(0).to(device)
    question_t = torch.LongTensor(question).unsqueeze(0).to(device)
    feature = model.makeContext(image_t, question_t)
    
    inputs = torch.LongTensor([[lang["<SOS>"]]]).to(device)
    hidden = None
    ans = []
    probs = []
    for i in range(20):
        outputs, hidden = model.decode(inputs, feature, hidden)
        prob, outputs = outputs.topk(1)
        if(outputs.item() == lang["<EOS>"]):
            break
        ans.append(outputs.item())
        probs.append(prob[0])

        inputs = outputs.squeeze(1).detach()
    return lang.vectorToSentence(ans), probs


class Beam():
    def __init__(self, seq, end, scores=None, state=None):
        self.seq = seq
        self.state = None
        self.scores = scores
        self.end = end
        
    def getInput(self):
        return self.seq[:, -1:], self.state
        
    def addState(self, next_seq, score, state):
        seq = torch.cat([self.seq, next_seq], 1)
        scores = torch.cat([self.scores, score], 1) if self.scores is not None else score
        
        return Beam(seq, 
                    self.end,
                    scores, 
                    state)
    def isEnd(self):
        return self.seq[0, -1] == self.end
    
    def score(self):
        if self.scores is None:
            return -1
        if self.isEnd():
            score = self.scores[:-1].mean() + 10
        else:
            score = self.scores[:-1].mean()
        return  score
    
    def __lt__(self, other):
        return self.score() < other.score()
    
def beamPredit(model, device, lang, image, question, beamSize, MAX_LEN=20): 
    image_t = image.unsqueeze(0).to(device)
    question_t = torch.LongTensor(question).unsqueeze(0).to(device)
    feature = model.makeContext(image_t, question_t)
    beams = [Beam(torch.LongTensor([[lang["<SOS>"]]]).to(device), lang["<EOS>"])]
    for _ in range(MAX_LEN):
        newBeams = []
        endSeq = 0
        for beam in beams:
            if beam.isEnd():
                newBeams.append(beam)
                endSeq += 1
            else:
                pre_inputs, hidden = beam.getInput()
                next_outputs, hidden = model.decode(pre_inputs, feature, hidden)

                probs, next_outputs = next_outputs.topk(beamSize)
                for i in range(probs.size(2)):
                    newBeams.append(beam.addState(next_outputs[:,:,i].detach(), probs[:,:,i].detach(), hidden))
        newBeams.sort(reverse=True)
        beams = newBeams[:beamSize]
        if endSeq >= beamSize:
            break
    answers = []
    probs = []
    for beam in beams:
        if beam.isEnd():
            ans = lang.vectorToSentence(beam.seq[0, 1:-1].cpu().numpy())
            score = beam.score() - 10
        else:
            ans = lang.vectorToSentence(beam.seq[0, 1:].cpu().numpy())
            score = beam.score()
        answers.append(ans)
        probs.append(score.item())
    return answers, probs