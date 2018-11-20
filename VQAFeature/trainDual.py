import torch 
from torch import optim
import torch.utils.data
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from utils.tool import Average
from tqdm import tqdm

from model.model import cnnTransforms, Gesd
from VQAFeature.model import VQADualModel
from VQAFeature.utils import setDualData

from dataset import VisDialDataset
from utils.token import Lang

import os
import argparse
def addArgparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epoch', help="Epoch to Train", type=int, default=10)
    parser.add_argument('-b', '--batch', help="Batch size", type=int, default=30)
    parser.add_argument('-lr', help="Loss to Train", type=float, default = 1e-4)
    parser.add_argument('-m', '--model', help="model dir", required=True)
    parser.add_argument('-d', '--data', help="Data loaction", default="/home/ball/dataset/mscoco/visdialog/visdial_1.0_train.json")
    parser.add_argument('-l', '--lang', help="Lang file", default="dataset/lang.pkl")
    parser.add_argument('-f', '--feature', help="Feature file", default="visdial_train.h5")
    parser.add_argument('-c', '--coco', help="coco image location", default="/home/ball/dataset/mscoco")
    return parser

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def trainer(args):
    if not os.path.isdir(args.model):
        os.makedirs(args.model, )
    print("Use Device: {}".format(DEVICE))
    
    lang = Lang.load(args.lang)
    dataset = VisDialDataset(dialFile = args.data,
                             cocoDir = args.coco, 
                             sentTransform = torch.LongTensor,
                             imgTransform = cnnTransforms,
                             convertSentence = lang.sentenceToVector)
    
    loader = torch.utils.data.DataLoader(dataset, 
                                         batch_size=args.batch, 
                                         shuffle=True, 
                                         num_workers=0, 
                                         collate_fn=VisDialDataset.collate_fn)

    image_setting = {
        "output_size": 1024,
        "pretrained": True
    }
    sentence_setting = {
        "word_size": len(lang),
        "output_size": 512
    }

    model = VQADualModel(image_setting, sentence_setting).to(DEVICE)

    model.train()
    
    criterion = torch.nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    recLoss = Average()
    
    print("\nStart trainning....\nEpoch\t:{}\nBatch\t:{}\nDataset\t:{}\n".format(args.epoch, args.batch, len(dataset)))
    
    for epoch in range(args.epoch):
        pbar = tqdm(loader)
        pbar.set_description("Epoch: {}, Loss: {:.4f}".format(epoch, 0))
        for i, data in enumerate(pbar, 0):
            loss = step(model=model, 
                        data=data, 
                        criterion=criterion, 
                        optimizer=optimizer,
                        lang=lang,
                        device=DEVICE)
            recLoss.addData(loss.item())
            pbar.set_description("Ep: {}, Loss: {:.5f}".format(epoch, loss.item()))
            if i % 10 == 0:
                pass
                #pbar.set_description("Epoch: {}, Loss: {:.5f}".format(epoch, recLoss.getAndReset()))
        torch.save(model, os.path.join(args.model, "VQAmodel.{}.pth".format(epoch)))    

def step(model, criterion, optimizer, **args):
    images_t, questions_t, answers_t, label_t = setDualData(**args)
    
    scores = model(images_t, questions_t, answers_t)

    loss = criterion(scores, label_t)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss
    
    
if __name__ == "__main__":
    args = addArgparse().parse_args()
    trainer(args)