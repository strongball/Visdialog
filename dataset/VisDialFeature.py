import torch
import json, os, glob
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset
import h5py

class VisDialFeature(Dataset):
    def __init__(self, dialFile, cocoDir, sentFeatureFile, featureType=["answers"], imgTransform=None, sentTransform=None, featureTransform=None):
        with open(dialFile, 'r') as f:
            self.data = json.load(f)
            self.data = self.data["data"]
        
        self.cocoDir = cocoDir
        self.imageFile = {}
        for image_path in tqdm(glob.iglob(os.path.join(self.cocoDir, '*', '*.jpg')), desc="Preparing image paths with image_ids"):
            self.imageFile[int(image_path[-12:-4])] = image_path
        self.sentFeature = h5py.File(sentFeatureFile, "r")
        self.featureType = featureType
        self.imgTransform = imgTransform
        self.sentTransform = sentTransform
        self.featureTransform = featureTransform
            
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
        
    def getSentence(self, idx, dtype):
        if dtype in self.featureType:
            if self.featureTransform:
                sent = self.featureTransform(self.sentFeature[dtype][idx])
            else:
                sent = self.sentFeature[dtype][idx]
        else:
            if self.sentTransform:
                sent = self.sentTransform(self.data[dtype][idx])
            else:
                sent = self.data[dtype][idx]
        return sent
        
    def __getitem__(self, idx):
        if isinstance(idx, slice) :
            #Get the start, stop, and step from the slice
            return [self[ii] for ii in range(*idx.indices(len(self)))]
        
        row = self.data["dialogs"][idx]
        item = {} 
        item["index"] = idx
        item["caption"] = self.sentFeature["caption"][idx]
        item["image"], sucess = self.getImage(row["image_id"])
        item["questions"] = []
        item["answers"] = []
        for i in range(10):
            item["questions"].append(self.getSentence(row["dialog"][i]["question"], "questions"))
            item["answers"].append(self.getSentence(row["dialog"][i]["answer"], "answers"))
        if not sucess:
            #print("Error Image: {}".format(idx))
            return self[idx-1] if idx > 0 else self[idx+1]
        return item
    
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
        answers = torch.stack(answers)

        return {
            "images": images,
            "questions": questions,
            "answers": answers
        }