import torch 
from torch import optim
import torch.utils.data
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from utils.tool import Average
from tqdm import tqdm

from model.model import cnnTransforms
from SentenceDecode.model import SentenceDecoder
from SentenceDecode.utils import setData, collate_fn

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
    parser.add_argument('-d', '--data', help="Data loaction", default="/home/ball/dataset/mscoco/visdialog/visdial_1.0_val.json")
    parser.add_argument('-l', '--lang', help="Lang file", default="dataset/lang.pkl")
    parser.add_argument('-c', '--coco', help="coco image location", default="/home/ball/dataset/mscoco")
    parser.add_argument('-f', '--encode', help="encode model")
    return parser

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def trainer(args):
    if not os.path.isdir(args.model):
        os.makedirs(args.model, )
    print("Decode feature")
    print("Use Device: {}".format(DEVICE))
    
    lang = Lang.load(args.lang)
    dataset = VisDialDataset(dialFile = args.data,
                             cocoDir = args.coco, 
                             sentTransform = torch.LongTensor,
                             imgTransform = cnnTransforms,
                             convertSentence = lang.sentenceToVector)
    
    loader = torch.utils.data.DataLoader(dataset.data["answers"],
                                         batch_size=args.batch, 
                                         shuffle=True, 
                                         num_workers=4, 
                                         collate_fn=collate_fn)

    encodeModel = torch.load(args.encode).to(DEVICE).eval()
    
    sentence_setting = {
        "word_size": len(lang),
        "feature_size": 512,
    }
    model = SentenceDecoder(**sentence_setting).to(DEVICE)

    model.train()
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    recLoss = Average()
    
    print("\nStart trainning....\nEpoch\t:{}\nBatch\t:{}\nDataset\t:{}\n".format(args.epoch, args.batch, len(dataset)))
    
    for epoch in range(args.epoch):
        pbar = tqdm(loader)
        pbar.set_description("Epoch: {}, Loss: {:.4f}".format(epoch, 0))
        for i, data in enumerate(pbar, 0):
            loss = step(model=model, 
                        encodeModel=encodeModel,
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
        torch.save(model, os.path.join(args.model, "DecodeModel.{}.pth".format(epoch)))    

def step(model, encodeModel, criterion, optimizer, **args):
    en_seq, de_seq = setData(**args)
    
    feature = encodeModel.answer(en_seq)
    sent, _ = model(de_seq["in"], [feature.detach()])

    loss = criterion(sent.view(-1, sent.size(2)), de_seq["out"].view(-1))
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss
    
    
if __name__ == "__main__":
    args = addArgparse().parse_args()
    trainer(args)