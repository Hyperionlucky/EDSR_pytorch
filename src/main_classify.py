# import imp
# from asyncio.log import logger
# import logging
import time

import torch
from thop import profile
from thop import clever_format
from torch.utils.data import DataLoader

import model
from data.Demdataset import Demdataset
from loss.losses_flow import Loss
from option import args

torch.manual_seed(args.seed)
# checkpoint = utility.checkpoint(args)


def main():
    global model

    traindataset_path = args.dataset_dir + "train_" + str(args.scale) +"x_flow.txt"
    valdataset_path = args.dataset_dir + "val_" + str(args.scale) +"x_flow.txt"
    traindataset = Demdataset(traindataset_path,
                              mode="train", crop_size=args.patch_size, scale=args.scale, reverse=True)
    valdataset = Demdataset(valdataset_path, mode="val", crop_size=192, scale = args.scale)
    traindataloader = DataLoader(traindataset, batch_size=args.batch_size,
                                 num_workers=args.workers, drop_last=True, shuffle=True, pin_memory=False)
    valdataloader = DataLoader(valdataset, batch_size=args.test_batch_size,
                               num_workers=args.workers, drop_last=True, shuffle=False)
    loader = {
        "loader_train": traindataloader,
        "loader_test": valdataloader,
        "num_train": traindataset.__len__(),
        "num_val": valdataset.__len__(),
    }
    _model = model.Model(args=args)
    _loss = Loss(weight=args.loss_weight)
    from train_classify import Trainer
    t = Trainer(args, loader, _model, _loss)
    current_epoch = t.current_epoch
    if not args.test_only:
        for epoch in range(current_epoch, 100):
            train_start = time.time()
            t.train(epoch=epoch)
            val_start = time.time()
            print("train time: ", val_start - train_start)
            t.test(epoch=epoch)
            print("val time: ", time.time() - val_start) 
    else:
        t.test()
    # t.writer.add_graph(t.model, (torch.randn(1,1,48,48).cuda(),torch.randn(1,1,48,48).cuda()))
    t.writer.close()
    # checkpoint.done()


if __name__ == '__main__':
    main()
