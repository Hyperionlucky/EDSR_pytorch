# import imp
# from asyncio.log import logger
# import logging
import time

import torch
from thop import profile
from thop import clever_format
from torch.utils.data import DataLoader

import model
from data.DemDataset import DemDataset
from loss.losses_flow import Loss
from option import args

torch.manual_seed(args.seed)
# checkpoint = utility.checkpoint(args)


def main():
    global model

    train_dataset_path = args.dataset_dir + "train_" + str(args.scale) +"x_terrain.txt"
    val_dataset_path = args.dataset_dir + "val_" + str(args.scale) +"x_terrain.txt"
    train_dataset = DemDataset(train_dataset_path,
                              mode="train", crop_size=args.patch_size, scale=args.scale, reverse=True)
    val_dataset = DemDataset(val_dataset_path, mode="val", crop_size=192, scale = args.scale)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                 num_workers=args.workers, drop_last=True, shuffle=True, pin_memory=False)
    val_dataloader = DataLoader(val_dataset, batch_size=args.test_batch_size,
                               num_workers=args.workers, drop_last=True, shuffle=False)
    loader = {
        "loader_train": train_dataloader,
        "loader_test": val_dataloader,
        "num_train": train_dataset.__len__(),
        "num_val": val_dataset.__len__(),
    }
    _model = model.Model(args=args)
    _loss = Loss(weight=args.loss_weight,scale=args.scale)
    if args.isDual:
        from trainer_Dual import Trainer
        t = Trainer(args, loader, _model, _loss)
    else:
        from trainer import Trainer
        t =  Trainer(args, loader, _model, _loss)
    current_epoch = t.current_epoch
    if not args.test_only:
        for epoch in range(current_epoch, args.epochs):
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
