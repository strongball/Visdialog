import torch
import json, os, glob
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset

class VisDialDataset(Dataset):
    def __init__(self, dialFile, cocoDir, imgTransform=None, sentTransform=None, convertSentence=None):
        with open(dialFile, 'r') as f:
            self.data = json.load(f)
            self.data = self.data["data"]
        
        self.cocoDir = cocoDir
        self.imageFile = {}
        for image_path in tqdm(glob.iglob(os.path.join(self.cocoDir, '*', '*.jpg')), desc="Preparing image paths with image_ids"):
            self.imageFile[int(image_path[-12:-4])] = image_path
        
        self.imgTransform = imgTransform
        self.sentTransform = sentTransform if sentTransform else lambda s: s
        self.convertSentence(convertSentence)
    
    def convertSentence(self, fn):
        if fn: 
            for kType in ["questions", "answers"]:
                for idx in range(len(self.data[kType])):
                    self.data[kType][idx] = fn(self.data[kType][idx])
                    
            for idx in range(len(self.data["dialogs"])):
                self.data["dialogs"][idx]["caption"] = fn(self.data["dialogs"][idx]["caption"])
            
    def __len__(self):
        return len(self.data["dialogs"])

    def getImage(self, image_id):
        file = self.imageFile[image_id]
        if os.path.isfile(file):
            img = Image.open(file)
            if self.imgTransform:
                img = self.imgTransform(img)
                if img.size(0) != 3:
                    return [], False
            return img, True
        else:
            return [], False
        
    def __getitem__(self, idx):
        if isinstance(idx, slice) :
            #Get the start, stop, and step from the slice
            return [self[ii] for ii in range(*idx.indices(len(self)))]
        
        row = self.data["dialogs"][idx]
        item = {} 
        item["index"] = idx
        item["caption"] = self.sentTransform(row["caption"])
        item["image"], sucess = self.getImage(row["image_id"])
        item["questions"] = []
        item["answers"] = []
        for i in range(10):
            item["questions"].append(self.sentTransform(self.data["questions"][row["dialog"][i]["question"]]))
            item["answers"].append(self.sentTransform(self.data["answers"][row["dialog"][i]["answer"]]))
        if not sucess:
            #print("Error Image: {}".format(idx))
            return self[idx-1] if idx > 0 else self[idx+1]
        return item
    
    def getAllSentences(self):
        sentences = []
        for dialog in self.data["dialogs"]:
            sentences.append(dialog["caption"])
        sentences += self.data["questions"]
        sentences += self.data["answers"]
        return sentences
    
    def collate_fn(batch):
        images = []
        questions = []
        answers = []
        for i in range(len(batch[0]["questions"])):
            for row in batch:
                images.append(row["image"])
                questions.append(row["questions"][i])
                answers.append(row['answers'][i])
        images = torch.stack(images)
        return {
            "images": images,
            "questions": questions,
            "answers": answers
        }