from VQAFeature.trainDual import addArgparse, trainer
#from SentenceDecode.train import addArgparse, trainer


if __name__ == "__main__":
    args = addArgparse().parse_args()
    trainer(args)