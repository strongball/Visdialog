import argparse
from tqdm import tqdm

from dataset import VisDialDataset
from utils.token import Lang


parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data', help="Visdial dataset", required=True)
parser.add_argument('-o', '--output', help="Output location", required=True)
parser.add_argument('-l', '--lang', help="Old lang file")
parser.add_argument('-s', '--split', help="Split type", default=' ', choices=["", " ", 'jieba'])

if __name__ == "__main__":
    args = parser.parse_args()
    
    if args.lang:
        print("Load lang file: {}".format(args.lang))
        lang = Lang.load(args.lang)
    else:
        lang = Lang("Visdial", split = args.split)
        
    print("Loading dataset......")
    dataset = VisDialDataset(args.data, "")
    print("Dataset size: {}".format(len(dataset)))
    
    for i, sent in enumerate(tqdm(dataset.getAllSentences(), desc="Create Dict")):
        lang.addSentance(sent)
        
    lang.save(args.output)
    print("Saved lang: {}, Word size: {}".format(args.output, len(lang)))
    
    