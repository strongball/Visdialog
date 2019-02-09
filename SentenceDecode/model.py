import torch
import torch.nn.functional as F
from model.model import SentenceEncoder, SentenceDecoder, ImageEncoder, Gesd

class VQAFeatureModel(torch.nn.Module):
    def __init__(self, image_setting, question_setting, answer_setting, padding_idx=0):
        super(VQAFeatureModel, self).__init__()

        self.imageModel = ImageEncoder(**image_setting)
        self.questionModel = SentenceEncoder(**question_setting)
        answer_setting["feature_size"] = image_setting["output_size"] + question_setting["output_size"]
        self.answerModel = torch.nn.Linear(answer_setting["feature_size"], answer_setting["output_size"])
    
    def forward(self, input_imgs, input_questions):
        img_out = self.imageModel(input_imgs)
        q_out, _ = self.questionModel(input_questions)
        feature = torch.cat([img_out, q_out], dim=1)
        outputs = self.answerModel(feature)
        return outputs
    
class VQADualModel(torch.nn.Module):
    def __init__(self, image_setting, sentence_setting, concat_dropout=0.1, padding_idx=0):
        super(VQADualModel, self).__init__()

        self.imageModel = ImageEncoder(**image_setting)
        self.questionModel = SentenceEncoder(**sentence_setting)
        
        concat_size = image_setting["output_size"] + sentence_setting["output_size"]
        self.concatLayer = model = torch.nn.Sequential(
            torch.nn.Linear(concat_size, concat_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(concat_dropout),
            torch.nn.Linear(concat_size, sentence_setting["output_size"]),
        )
        
        self.answerModel = SentenceEncoder(**sentence_setting)
        self.gesd = Gesd()
    
    def forward(self, input_imgs, input_questions, input_answer):
        iqFeature = self.imageQuestion(input_imgs, input_questions)
        aFeature = self.answer(input_answer)
        sim = self.gesd(iqFeature, aFeature)
        return sim
    
    def imageQuestion(self, input_imgs, input_questions):
        img_out = self.imageModel(input_imgs)
        q_out, _ = self.questionModel(input_questions)
        feature = torch.cat([img_out, q_out], dim=1)
        feature = self.concatLayer(feature)
        return feature
    
    def answer(self, input_answer):
        a_out, _ = self.answerModel(input_answer)
        return a_out