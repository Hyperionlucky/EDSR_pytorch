# import imp
# from asyncio.log import logger
# import logging
import time

import torch
from thop import profile
from thop import clever_format
from torch.utils.data import DataLoader

# import utility
# import data
import model
from data.Demdataset import Demdataset
from loss.losses_flow import Loss
from option import args
from trainer_Dual import Trainer

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
    # traindataset_extend_path = args.dataset_dir + "train_" + str(args.scale) +"x_flow_extend.txt"
    # traindataset_extend = Demdataset(traindataset_extend_path, mode="train", crop_size=args.patch_size, scale=args.scale, reverse=True)
    # train_extend_dataloader = DataLoader(traindataset_extend, batch_size=args.batch_size,
    #                              num_workers=args.workers, drop_last=True, shuffle=True, pin_memory=False)
    loader = {
        "loader_train": traindataloader,
        "loader_test": valdataloader,
        "num_train": traindataset.__len__(),
        "num_val": valdataset.__len__(),
    }
    if args.model == "EDSR_Slope":
        from model.DemSR import DemSR
        _model = DemSR(args)
    if args.model == "DRN":
        from model.drn import DRN
        _model = DRN(args=args)
    if args.model == "EDSR":
        from model.Edsr import EDSR
        _model = EDSR(args)
    if args.model in ["nearest","bilinear","bicubic"]:
        from model.benchmark import Benchmark
        _model = Benchmark(args=args)
    if args.model == "RFAN":
        from model.rfan import RFAN
        _model = RFAN(args=args) 
    # print(_model)
    _loss = Loss(weight=args.loss_weight)
    t = Trainer(args, loader, _model, _loss)
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
