import torch 
from torch import optim
import torch.utils.data
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from utils.tool import Average
from tqdm import tqdm

from model.model import SentenceEncoder, SentenceDecoder, ImageEncoder, cnnTransforms
from dataset import VisDialDataset
from utils.token import Lang

import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epoch', help="Epoch to Train", type=int, default=10)
parser.add_argument('-b', '--batch', help="Batch size", type=int, default=30)
parser.add_argument('-lr', help="Loss to Train", type=float, default = 1e-4)
parser.add_argument('-m', '--model', help="model dir", required=True)
parser.add_argument('-d', '--data', help="Data loaction", default="/home/ball/dataset/mscoco/visdialog/visdial_1.0_train.json")
parser.add_argument('-l', '--lang', help="Lang file", default="dataset/lang.pkl")
parser.add_argument('-c', '--coco', help="coco image location", default="/home/ball/dataset/mscoco")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def trainer(args):
    if not os.path.isdir(args.model):
        os.mkdir(args.model)
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
    imgcnn = ImageEncoder(1024, 1024, pretrained=True).to(DEVICE)
    sentDecoder = SentenceDecoder(word_size=len(lang), em_size=256, hidden_size=256, feature_size=1024).to(DEVICE)

    imgcnn.train()
    sentDecoder.train()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(list(imgcnn.parameters())+list(sentDecoder.parameters()), lr=args.lr)
    recLoss = Average()
    for epoch in range(args.epoch):
        pbar = tqdm(loader)
        pbar.set_description("Epoch: {}, Loss: {:.4f}".format(epoch, 0))
        for i, data in enumerate(pbar, 0):
            loss = step(imageModel=imgcnn, 
                        sentModel=sentDecoder, 
                        data=data, 
                        criterion=criterion, 
                        optimizer=optimizer,
                        lang=lang)
            recLoss.addData(loss.item())
            pbar.set_description("Epoch: {}, Loss: {:.4f}".format(epoch, loss.item()))
            if i % 10 == 0:
                pass
                #pbar.set_description("Epoch: {}, Loss: {:.5f}".format(epoch, recLoss.getAndReset()))
        torch.save(imgcnn, os.path.join(args.model, "image{}.pkl".format(epoch)))
        torch.save(sentDecoder, os.path.join(args.model, "sentence{}.pkl".format(epoch)))
    

def step(imageModel, sentModel,criterion, optimizer, **args):
    images, in_seq, out_seq = setData(**args)
    imgout = imageModel(images)
    sentout, senthidden = sentModel(in_seq, imgout)
    
    loss = criterion(sentout.view(-1, sentout.size(2)), out_seq.view(-1))
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss
    
def setData(data, lang):
    in_seq = []
    out_seq = []
    for cap in data["captions"]:
        in_seq.append(torch.cat([torch.LongTensor([lang["<SOS>"]]), cap]).to(DEVICE))
        out_seq.append(torch.cat([cap, torch.LongTensor([lang["<EOS>"]])]).to(DEVICE))
    out_seq = pad_sequence(out_seq, batch_first=True)
    images = data["images"].to(DEVICE)
    return images, in_seq, out_seq
    
def collate_fn(batch):
    images = []
    captions = []
    for row in batch:
        images.append(row["image"])
        captions.append(row["caption"])
    images = torch.stack(images)
    return {
        "images": images,
        "captions": captions
    }
    
    
    
if __name__ == "__main__":
    args = parser.parse_args()
    trainer(args)