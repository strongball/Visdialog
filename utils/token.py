import jieba
import pickle

class Lang:
    def __init__(self, name, split="jieba"):
        self.name = name
        self.split = split
        self.word2index = {}
        self.word2count = {}
        self.index2word = {}
        self.n_words = 0
        
        self.addWord("<PAD>")
        self.addWord("<SOS>")
        self.addWord("<EOS>")
        self.addWord("<UNK>")
        
    def  __getitem__(self, key):
        if isinstance(key, str):
            if key in self.word2index:
                return self.word2index[key]
            else:
                return self.word2index["<UNK>"]
        elif isinstance(key, int):
            if key < self.n_words:
                return self.index2word[key]
        return None
    
    def __len__(self):
        return self.n_words
    
    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1
    def splitSentence(self, s):
        if self.split == "":
            return s
        if self.split == "jieba":
            return jieba.cut(s)
        else:
            return s.split(self.split)
    def addSentance(self, sent):
        for w in self.splitSentence(sent):
            self.addWord(w)
            
    def sentenceToVector(self, s):
        tokens = []
        for w in self.splitSentence(s):
            if w in self.word2index:
                tokens.append(self.word2index[w])
            else:
                tokens.append(self.word2index["<UNK>"])
        return tokens
    
    def vectorToSentence(self, v):
        sp = " " if self.split == " " else ""
        s = sp.join(self.index2word[i] for i in v)
        return s
    
    def save(self, langFile):
        with open(langFile, 'wb') as f:
            pickle.dump(self, f)
            
    def load(langFile):
        with open(langFile, 'rb') as f:
            lang = pickle.load(f)
            print("Load lang model: {}. Word size: {}".format(langFile, len(lang)))
        return lang