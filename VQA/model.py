import torch
import torch.nn.functional as F
from model.model import SentenceEncoder, SentenceDecoder, ImageEncoder

class VQAModel(torch.nn.Module):
    def __init__(self, image_setting, question_setting, answer_setting, padding_idx=0):
        super(VQAModel, self).__init__()

        self.imageModel = ImageEncoder(**image_setting)
        self.questionModel = SentenceEncoder(**question_setting)
        answer_setting["feature_size"] = image_setting["output_size"] + question_setting["output_size"]
        self.answerModel = SentenceDecoder(**answer_setting)
        
    def forward(self, input_imgs, input_questions, input_answers, hidden=None):
        context = self.makeContext(input_imgs, input_questions)
        output, hidden = self.answerModel(input_answers, context, hidden)
        return output, hidden
    
    def makeContext(self, input_imgs, input_questions):
        img_out = self.imageModel(input_imgs)
        q_out, _ = self.questionModel(input_questions)
        return [img_out, q_out]
    
    def decode(self, input_seq, context, hidden=None):
        output, hidden = self.answerModel(input_seq, context, hidden)
        output = F.softmax(output, dim=2)
        return output, hidden