import torch

class BeamSearch():
    def __init__(self, encode_fn, decode_fn, lang, device, beamSize, MAX_LEN=20, softmax=True):
        self.encode = encode_fn
        self.decode = decode_fn
        self.beamSize = beamSize
        self.MAX_LEN = MAX_LEN
        self.lang = lang
        self.device = device
        self.softmax = softmax
        
    def makeFeature(self, **args):
        out = self.encode(**args)
        if not isinstance(out, list):
            out = [out]
        return out
    
    def featureDecode(self, pre_inputs, feature, hidden):
        return self.decode(pre_inputs, feature, hidden)
    
    def predit(self, **inputs):
        feature = self.makeFeature(**inputs)
        pre_inputs = torch.LongTensor([[self.lang["<SOS>"]]]).to(self.device)
        hidden = None
        ans = []
        probs = []
        
        for i in range(self.MAX_LEN):
            next_outputs, hidden = self.featureDecode(pre_inputs, feature, hidden)
            prob, next_outputs = next_outputs.topk(1)
            if(next_outputs.item() == self.lang["<EOS>"]):
                break
            ans.append(next_outputs.item())
            probs.append(prob[0])

            pre_inputs = next_outputs.squeeze(1).detach()
        return self.lang.vectorToSentence(ans), probs
        
    def beamPredit(self, **inputs):
        feature = self.makeFeature(**inputs)
        
        beams = [Beam(torch.LongTensor([[self.lang["<SOS>"]]]).to(self.device), self.lang["<EOS>"])]
        for _ in range(self.MAX_LEN):
            newBeams = []
            endSeq = 0
            for beam in beams:
                if beam.isEnd():
                    newBeams.append(beam)
                    endSeq += 1
                else:
                    pre_inputs, hidden = beam.getInput()
                    next_outputs, hidden = self.featureDecode(pre_inputs, feature, hidden)
                    if self.softmax:
                        next_outputs = torch.nn.functional.softmax(next_outputs, dim=2)
                    probs, next_outputs = next_outputs.topk(self.beamSize)
                    for i in range(probs.size(2)):
                        newBeams.append(beam.addState(next_outputs[:,:,i].detach(), probs[:,:,i].detach(), hidden))
            newBeams.sort(reverse=True)
            beams = newBeams[:self.beamSize]
            if endSeq >= self.beamSize:
                break
        answers = []
        probs = []
        for beam in beams:
            if beam.isEnd():
                ans = self.lang.vectorToSentence(beam.seq[0, 1:-1].cpu().numpy())
                score = beam.score() - 10
            else:
                ans = self.lang.vectorToSentence(beam.seq[0, 1:].cpu().numpy())
                score = beam.score()
            answers.append(ans)
            probs.append(score.item())
        return answers, probs
        
        
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
            score = self.scores[:,:-1].mean() + 10
        else:
            score = self.scores[:,:-1].mean()
        return  score
    
    def __lt__(self, other):
        return self.score() < other.score()