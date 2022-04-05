# import imp
from asyncio.log import logger
# import logging
import time
import torch

# import utility
# import data
import model
from model.DemSR import DemSR
from loss.losses import Loss
from data.Demdataset import Demdataset
from torch.utils.data import DataLoader

from option import args
from trainer import Trainer

torch.manual_seed(args.seed)
# checkpoint = utility.checkpoint(args)

def main():
    global model
    
    # if checkpoint.ok:
    traindataset = Demdataset(args.traindataset_path, mode="train",scale=96, reverse=True)
    valdataset = Demdataset(args.valdataset_path, mode="val",scale=192)
    traindataloader = DataLoader(traindataset,batch_size=args.batch_size,num_workers=args.workers,drop_last=True,shuffle=True,pin_memory=False)
    valdataloader = DataLoader(valdataset, batch_size=args.test_batch_size,num_workers=args.workers,drop_last=True,shuffle=False)
    loader = {
        "loader_train": traindataloader,
        "loader_test": valdataloader,
        "num_train": traindataset.__len__(),
        "num_val": valdataset.__len__(),
    }
    _model = DemSR(args)
    # print(_model)
    _loss = Loss(weight=[1,1])
    t = Trainer(args, loader, _model, _loss)
    for epoch in range(args.epochs):
        train_start = time.time()
        t.train(epoch=epoch)
        val_start = time.time()
        print("train time: ",val_start- train_start)
        t.test(epoch=epoch)
        print("val time: ", time.time()- val_start)
    t.writer.close()
        # checkpoint.done()

if __name__ == '__main__':
    main()
