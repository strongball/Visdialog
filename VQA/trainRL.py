import torch 
from torch import optim
import torch.utils.data
import torch.nn.functional as F


from utils.tool import Average
from tqdm import tqdm

from model.model import cnnTransforms
from VQA.model import VQAModel
from VQA.utils import setData, trainSentence, collate_fn
from VQA.rlLearn import RLhelper

from dataset import VisDialDataset
from utils.token import Lang

import os
import argparse
def addArgparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epoch', help="Epoch to Train", type=int, default=10)
    parser.add_argument('-b', '--batch', help="Batch size", type=int, default=30)
    parser.add_argument('-lr', help="Loss to Train", type=float, default = 1e-4)
    parser.add_argument('-p', '--pretrain', help="pretrain model dir", required=True)
    parser.add_argument('-m', '--model', help="model dir", required=True)
    parser.add_argument('-d', '--data', help="Data loaction", default="/home/ball/dataset/mscoco/visdialog/visdial_1.0_train.json")
    parser.add_argument('-l', '--lang', help="Lang file", default="dataset/lang.pkl")
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
                                         num_workers=4, 
                                         collate_fn=collate_fn)


    model = torch.load(args.pretrain).to(DEVICE).train()

    model.train()
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    recLoss = Average()
    
    helper = RLhelper(model, criterion)
    print("\nStart trainning....\nEpoch\t:{}\nBatch\t:{}\nDataset\t:{}\n".format(args.epoch, args.batch, len(dataset)))
    
    for epoch in range(args.epoch):
        pbar = tqdm(loader)
        pbar.set_description("Ep: {}, Loss: {:.4f}".format(epoch, 0))
        for i, data in enumerate(pbar, 0):
            loss = step(model=model, 
                        data=data, 
                        helper=helper, 
                        optimizer=optimizer,
                        lang=lang,
                        device=DEVICE)
            recLoss.addData(loss.item())
            pbar.set_description("Ep: {}, Loss: {:.2f}".format(epoch, loss.item()))
            if i % 10 == 0:
                pass
                #pbar.set_description("Epoch: {}, Loss: {:.5f}".format(epoch, recLoss.getAndReset()))
        print("Epoch: {}, Loss: {:.4f}".format(epoch, recLoss.getAndReset()))
        torch.save(model, os.path.join(args.model, "VQAmodel.{}.pth".format(epoch)))    

def step(model, helper, optimizer, lang, data, device):
    images_t, questions_t, answers_t = setData(data, lang, device)
    
    if torch.rand(1) > 0.9:
        samples = helper.getRLSample(images_t, questions_t, lang["<EOS>"])
        rewards = helper.reward(samples)
        answers_t = trainSentence(samples, lang["<SOS>"], lang["<EOS>"], device)
    else:
        rewards = [1] * images_t.size(0)
    outputs, hidden = model(images_t, questions_t, answers_t["in"])
    
    loss = helper.RLLoss(outputs, answers_t["out"], rewards)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss
    
    
if __name__ == "__main__":
    args = addArgparse().parse_args()
    trainer(args)