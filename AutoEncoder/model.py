import torch
import torch.nn.functional as F
from model.model import SentenceEncoder, SentenceDecoder

class AutoModel(torch.nn.Module):
    def __init__(self, encoder_setting, decoder_setting, padding_idx=0):
        super(AutoModel, self).__init__()

        self.encodeModel = SentenceEncoder(**encoder_setting)
        decoder_setting["feature_size"] = encoder_setting["output_size"]
        self.decoderModel = SentenceDecoder(**decoder_setting)
        
    def forward(self, encode_seqs, input_seqs, hidden=None):
        context = self.makeContext(encode_seqs)
        output, hidden = self.decoderModel(input_seqs, context, hidden)
        return output, hidden
    
    def makeContext(self, encode_seqs):
        en_out, _ = self.encodeModel(encode_seqs)
        return [en_out]
    
    def decode(self, input_seq, context, hidden=None):
        output, hidden = self.decoderModel(input_seq, context, hidden)
        output = F.softmax(output, dim=2)
        return output, hidden