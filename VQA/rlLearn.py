import torch
class RLhelper():
    def __init__(self, model, criterion):
        self.model = model
        self.criterion = criterion
        
    def getRLSample(self, images_t, questions_t, eos, maxLength=20):

        # create new tensor as same as image device
        inputs = images_t.new_tensor([[sos]], dtype=torch.long)
        inputs = inputs.repeat(images_t.size(0), 1)
        hidden = None
        sampleAns = []
        
        feature = self.model.makeContext(images_t, questions_t)
        for i in range(maxLength):
            outputs, hidden = self.model.decode(inputs, feature, hidden)
            inputs = torch.multinomial(outputs[:,0,:], 1).detach()
            sampleAns.append(inputs)
        sampleAns = torch.stack(sampleAns).squeeze(2).transpose(0,1)
        sampleAns = cutWithEOS(sampleAns, eos)
        return sampleAns
    
    def reward(self, outputs):
        rewards = []
        for output in outputs:
            rewards.append(1-1/(output.size(0)+1))
        return rewards
    
    def RLLoss(self, predits, targets, rewards):
        loss = 0
        batch = len(rewards)
        for batch in range(len(rewards)):
            loss += self.criterion(predits[batch, :targets[batch].size(0)], targets[batch]) * rewards[batch]
        return loss / batch
    
def cutWithEOS(batch, eos):
    cutter = []
    lens = batch.size()
    for sent_l in range(lens[0]):
        for word_l in range(lens[1]):
            if batch[sent_l, word_l] == eos or word_l+1 == lens[1]:
                cutter.append(batch[sent_l, 0:word_l])
                break
    return cutter