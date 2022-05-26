import os

import numpy as np
from utils.saver import Saver
import logging
import datetime
from utils.metrics_classify import Evaluator
from data import prefetcher
from utils.summaries import TensorboardSummary
import torch
from thop import profile
from thop import clever_format

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class Trainer():
    def __init__(self, args, loader, my_model, my_loss):
        self.args = args
        self.scale = args.scale
        self.saver = Saver(self.args)
        self.saver.save_experiment_config()
        # self.ckp = ckp
        self.loader_train = loader['loader_train']           #获取训练数据集
        self.loader_test = loader['loader_test']
        self.num_trainImage = loader["num_train"]
        self.num_valImage = loader["num_val"]             #获取测试数据
        #print(loader.loader_test)
        self.model = my_model                              #神经网络结构
        self.loss = my_loss
        self.current_epoch = 0
        self.optimizer = None
        self.lr_scheduler = None
        if not args.test_only:                                #损失函数
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr,betas=[args.beta1,args.beta2], eps=args.epsilon)
            # self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,args.epochs,args.eta_min)
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,milestones=[40,80],gamma=0.5)            
        if args.resume is not None:
            checkpoint = torch.load(args.resume, map_location='cpu')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.current_epoch = checkpoint['epoch']
            if self.optimizer is not None:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.lr_scheduler.load_state_dict(checkpoint["scheduler"])
                for state in self.optimizer.state.values():
                    for k,v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.cuda()

        self.best_pred = 0.0
        self.summary = TensorboardSummary(directory=self.saver.experiment_dir)
        logging.basicConfig(filename=os.path.join(self.saver.experiment_dir, "train.log"), filemode="w", level=logging.INFO, format="%(levelname)s:%(asctime)s:%(message)s")
        self.writer = self.summary.create_summart()
        self.train_iters_epoch = len(self.loader_train)
        self.val_iters_epoch = len(self.loader_test)
        self.train_evaluator = Evaluator(3)
        self.val_evaluator = Evaluator(3)

        if torch.cuda.is_available():
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            self.model = self.model.cuda()
            torch.backends.cudnn.benchmark = True 

    def train(self,epoch):

        train_loss = 0.0
        # train_L1_loss = 0.0
        self.train_evaluator.reset()
        self.model.train()                                         #设置train属性为true
        train_prefetcher = prefetcher.DataPrefetcher(self.loader_train)
        hr,_,flow = train_prefetcher.next()
        i = 1
        while hr is not None:
            self.optimizer.zero_grad()
            hr_seg = self.model(hr)
            total_loss = self.loss.CrossEntropyLoss(hr_seg, flow)
            

            # total_loss = self.args.loss_weight[0] * l1_loss + self.args.loss_weight[1]* slope_loss
            total_loss.backward()
            self.optimizer.step()
            train_loss += total_loss.item()
            # train_L1_loss += l1_loss.item()


            global_step = i + self.train_iters_epoch * epoch
            
            if global_step % 50 == 0:
                self.writer.add_scalar("train/train_loss", train_loss/i, i + self.train_iters_epoch * epoch)
                msg = "%s | Epoch: %d | global_step: %d | lr: %.8f | Train loss_average: %.4f " %(datetime.datetime.now(), epoch, global_step, self.optimizer.param_groups[0]["lr"], train_loss/i)
                print(msg)
                logging.info(msg=msg)
            i += 1
            hr_seg = hr_seg.data.cpu().numpy()
            hr_seg = np.argmax(hr_seg, axis=1)
            hr_seg = hr_seg[:, np.newaxis, :, :]
            flow = flow.data.cpu().numpy()
            self.train_evaluator.add_batch(label=flow, pred=hr_seg)
            hr,_,flow = train_prefetcher.next()

            
        mIoU_class, mIoU = self.train_evaluator.Mean_Intersection_over_Union()
        precision, recall, F1_class, F1 = self.train_evaluator.F1_score()

        self.writer.add_scalar("train/learning_rate", self.optimizer.param_groups[0]["lr"], epoch)
        self.writer.add_scalar("train/loss_epoch", train_loss/(i+1), epoch)
        self.writer.add_scalar("train/mIoU", mIoU, epoch)
        self.writer.add_scalar("train/F1", F1, epoch)
        # 计算参数量
        params_num = sum(p.numel() for p in self.model.parameters())


        print("Train:")
        print('[Epoch: %d, numImages: %5d]' % (epoch, self.num_trainImage))
        print("mIoU:{}, mIoU_class:{}".format(mIoU, mIoU_class))
        print("F1:{}, F1_class:{}".format(F1, F1_class))
        print('Loss: %.3f' % (train_loss/i))
        print("Params: %.2fM" %(params_num / 1e6))
        logging.info("Train:")
        logging.info('[Epoch: %d, numImages: %5d]' % (epoch, self.num_trainImage))
        logging.info("mIoU:{}, mIoU_class:{}".format(mIoU, mIoU_class))
        logging.info("F1:{}, F1_class:{}".format(F1, F1_class))
        logging.info('Loss: %.3f' %(train_loss / i))
        logging.info("Params: %.2fM" %(params_num / 1e6))

        self.lr_scheduler.step()


    def test(self, epoch=0):
        val_loss = 0.0
        # val_L1_loss = 0.0
        # val_slope_loss = 0.0
        self.val_evaluator.reset()
        self.model.eval()                                            #设置train为false

        val_prefetcher = prefetcher.DataPrefetcher(self.loader_test)
        hr,_,flow = val_prefetcher.next()
        # flops,params = profile(self.model, inputs=(lr,))
        # flops,params = clever_format([flops, params], "%.3f")
        i = 1
        while hr is not None:
            with torch.no_grad():
                hr_seg = self.model(hr)
                val_loss = self.loss.CrossEntropyLoss(hr_seg, flow)
            hr = hr.data.cpu().numpy()
            flow = flow.data.cpu().numpy()
            self.val_evaluator.add_batch(label=flow, pred=hr_seg)
                
            i += 1
            hr,_,flow = val_prefetcher.next()

        mIoU_class, mIoU = self.val_evaluator.Mean_Intersection_over_Union()
        precision, recall, F1_class, F1 = self.val_evaluator.F1_score()
        print("Validation:")
        print('[Epoch: %d, numImages: %5d]' % (epoch, self.num_valImage))
        print("mIoU:{}, mIoU_class:{}".format(mIoU, mIoU_class))
        print("F1:{}, F1_class:{}".format(F1, F1_class))
        logging.info("Validation:")
        logging.info('[Epoch: %d, numImages: %5d]' % (epoch, self.num_valImage))
        logging.info("mIoU:{}, mIoU_class:{}".format(mIoU, mIoU_class))
        logging.info("F1:{}, F1_class:{}".format(F1, F1_class))
        # print('\nFLOPs: ',flops, 'Params: ', params)
        # logging.info("FLOPs:{}, Params: {}".format(flops,params))
        if not self.args.test_only:
            self.writer.add_scalar("val/loss_epoch", val_loss/(i+1), epoch)
            self.writer.add_scalar("val/mIoU", mIoU, epoch)
            self.writer.add_scalar("val/F1", F1, epoch)
            # 计算参数量
            params_num = sum(p.numel() for p in self.model.parameters())
            print('Loss: %.4f' % (val_loss/i))
            print("Params: %.2fM" %(params_num / 1e6))
            logging.info('Loss: %.3f' %(val_loss / i))
            logging.info("Params: %.2fM" %(params_num / 1e6))
            new_pred = mIoU
            if new_pred > self.best_pred:
                is_best = True
                self.best_pred = new_pred
                self.saver.save_checkpoint({'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.lr_scheduler.state_dict(),
                'best_pred': self.best_pred}, is_best=is_best)

