import torch
import torch.nn.functional as F
from model.model import SentenceEncoder, SentenceDecoder, ImageEncoder

class ImageCaption(torch.nn.Module):
    def __init__(self, image_setting, question_setting, answer_setting, padding_idx=0):
        super(ImageCaption, self).__init__()

        self.imageModel = ImageEncoder(**image_setting)
        answer_setting["feature_size"] = image_setting["output_size"]
        self.answerModel = SentenceDecoder(**answer_setting)
        
    def forward(self, input_imgs, input_answers, hidden=None):
        context = self.makeContext(input_imgs, input_questions)
        output, hidden = self.decode(input_answers, context)
        return output, hidden
    
    def makeContext(self, input_imgs):
        img_out = self.imageModel(input_imgs)
        return [img_out]
    
    def decode(self, input_seq, context, hidden=None):
        output, hidden = self.answerModel(input_seq, context, hidden)
        output = F.softmax(output, dim=2)
        return output, hidden