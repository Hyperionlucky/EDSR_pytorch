import os
from utils.saver import Saver
import logging
import datetime
from utils.metrics import Evaluator
from data import prefetcher
from utils.summaries import TensorboardSummary
import torch

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
        if not args.test_only:                                #损失函数
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
            # self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=self.optimizer, milestones=self.args.milestones, gamma=self.args.gamma)     #优化策略
        if args.resume is not None:
            checkpoint = torch.load(args.resume, map_location='cpu')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.current_epoch = checkpoint['epoch']
            if self.optimizer is not None:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                for state in self.optimizer.state.values():
                    for k,v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.cuda()
            # self.optimizer = self.optimizer.cuda()
        
        # if self.args.load != '':
        #     self.optimizer.load(ckp.dir, epoch=len(ckp.log))          #获取log文档

        self.best_pred = 1e6

        self.summary = TensorboardSummary(directory=self.saver.experiment_dir)
        logging.basicConfig(filename=os.path.join(self.saver.experiment_dir, "train.log"), filemode="w", level=logging.INFO, format="%(levelname)s:%(asctime)s:%(message)s")
        self.writer = self.summary.create_summart()
        self.train_iters_epoch = len(self.loader_train)
        self.val_iters_epoch = len(self.loader_test)
        self.total_iters = 500 * self.train_iters_epoch
        self.train_evaluator = Evaluator(self.args.batch_size,self.args.rgb_range)
        self.val_evaluator = Evaluator(self.args.test_batch_size,self.args.rgb_range)

        if torch.cuda.is_available():
            self.model = self.model.cuda()

    def train(self,epoch):

        # self.ckp.write_log(
        #     '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(learning_rate))
        # )
        # self.loss.start_log()
        learning_rate = self.optimizer.param_groups[0]['lr']
        train_loss = 0.0
        train_L1_loss = 0.0
        train_slope_loss = 0.0
        self.train_evaluator.reset()
        self.model.train()                                         #设置train属性为true
        # timer_data, timer_model = utility.timer(), utility.timer()
        # TEMP
        train_prefetcher = prefetcher.DataPrefetcher(self.loader_train)
        hr,lr,slope = train_prefetcher.next()
        # self.loader_train.dataset.set_scale(0)                     #设置训练尺度
        i = 1
        while hr is not None:
            # timer_data.hold()
            # timer_model.tic()
            self.optimizer.zero_grad()
            # with torch.autograd.set_detect_anomaly(True):

            sr = self.model(lr, slope)
            l1_loss,slope_loss = self.loss.L1Loss(sr, hr),self.loss.SlopeLoss(sr,hr)
            

            total_loss = self.args.loss_weight[0] * l1_loss + self.args.loss_weight[1]* slope_loss
            total_loss.backward()
            self.optimizer.step()
            # train_loss = loss
            # if self.args.gclip > 0:
            #     utils.clip_grad_value_(
            #         self.model.parameters(),
            #         self.args.gclip
            #     )
            # self.optimizer.step()

            # timer_model.hold()

            # add different loss every step
            train_loss += total_loss.item()
            train_L1_loss += l1_loss.item()
            train_slope_loss += slope_loss.item()


            global_step = i + self.train_iters_epoch * epoch
            if epoch < 500:
                learning_rate = self.args.lr * (1 - global_step/self.total_iters) ** 0.9
                self.optimizer.param_groups[0]["lr"] = learning_rate if learning_rate > self.args.lr * 0.1 else 1e-5
            else:
                self.optimizer.param_groups[0]["lr"] = self.args.lr * 0.1
            if global_step % 50 == 0:
                self.summary.visualize_image(self.writer, hr, lr, slope, sr, global_step, mode="train")
                self.writer.add_scalar("train/train_loss", train_loss/i, i + self.train_iters_epoch * epoch)
                self.writer.add_scalar("train/train_L1_loss", train_L1_loss/i, i + self.train_iters_epoch * epoch)
                self.writer.add_scalar("train/train_slope_loss", train_slope_loss/i, i + self.train_iters_epoch * epoch)
                msg = "%s | Epoch: %d | global_step: %d | lr: %.8f | Train loss_average: %.4f | L1 loss_average: %.4f | slope loss_average: %.4f" %(datetime.datetime.now(), epoch, global_step, self.optimizer.param_groups[0]["lr"], train_loss/i, train_L1_loss/i, train_slope_loss/i)
                print(msg)
                logging.info(msg=msg)
            i += 1
            sr = sr.data.cpu().numpy()
            hr = hr.data.cpu().numpy()
            self.train_evaluator.add_batch(sr=sr, hr=hr)
            # total_loss = l1_loss + slope_loss
            # del total_loss,sr,l1_loss,slope_loss,hr,lr
            # torch.cuda.empty_cache()
            hr,lr,slope = train_prefetcher.next()

            
        mse, mae, rmse, e_max, slope_mae,psnr  = self.train_evaluator.score(self.num_trainImage)
        self.writer.add_scalar("train/learning_rate", self.optimizer.param_groups[0]["lr"], epoch)
        self.writer.add_scalar("train/loss_epoch", train_loss/(i+1), epoch)
        self.writer.add_scalar("train/MSE", mse, epoch)
        self.writer.add_scalar("train/PSNR", psnr, epoch)
        self.writer.add_scalar("train/MAE", mae, epoch)
        self.writer.add_scalar("train/RMSE", rmse, epoch)
        self.writer.add_scalar("train/E_MAX", e_max, epoch)
        self.writer.add_scalar("train/Slope_MAE", slope_mae, epoch)

        # 计算参数量
        params_num = sum(p.numel() for p in self.model.parameters())


        print("Train:")
        print('[Epoch: %d, numImages: %5d]' % (epoch, self.num_trainImage))
        print("MSE:{}, MAE:{}, RMSE:{}, E_MAX:{}, SLOPE_MAE:{}, PSNR:{}".format(mse, mae, rmse, e_max, slope_mae, psnr))
        print('Loss: %.3f' % (train_loss/i))
        print("\nParams: %.2fM" %(params_num / 1e6))

        logging.info("Train:")
        logging.info('[Epoch: %d, numImages: %5d]' % (epoch, self.num_trainImage))
        logging.info("MSE:{}, MAE:{}, RMSE:{}, E_MAX:{}, SLOPE_MAE:{}, PSNR:{}".format(mse, mae, rmse, e_max, slope_mae, psnr))
        logging.info('Loss: %.3f' %(train_loss / i))
        logging.info("\nParams: %.2fM" %(params_num / 1e6))

        # self.lr_scheduler.step()

        # torch.cuda.empty_cache()

    def test(self, epoch=0):
        val_loss = 0.0
        val_L1_loss = 0.0
        val_slope_loss = 0.0
        self.val_evaluator.reset()
        # epoch = self.current_epochs
        # self.ckp.write_log('\nEvaluation:')
        # self.ckp.add_log(
            # torch.zeros(1, len(self.loader_test), len(self.scale))
        # )
        self.model.eval()                                            #设置train为false

        # timer_test = utility.timer()
        # if self.args.save_results: self.ckp.begin_background()
        val_prefetcher = prefetcher.DataPrefetcher(self.loader_test)
        hr,lr,slope = val_prefetcher.next()
        i = 1
        while hr is not None:
            with torch.no_grad():
                sr = self.model(lr,slope)
                # save_list = [sr]
                l1_loss,slope_loss = self.loss.L1Loss(sr, hr),self.loss.SlopeLoss(sr,hr)
                # total_loss = l1_loss + slope_loss
            # val_loss += total_loss.item()
            val_L1_loss += l1_loss.item()
            val_slope_loss += slope_loss.item()
            # val_loss = val_L1_loss + val_slope_loss
            val_loss = self.args.loss_weight[0] * val_L1_loss + self.args.loss_weight[1]* val_slope_loss
            global_step = i + self.val_iters_epoch * epoch
            if global_step % 50 == 0:
                self.summary.visualize_image(self.writer, hr, lr, slope, sr, global_step, mode="val")
            sr = sr.data.cpu().numpy()
            hr = hr.data.cpu().numpy()
            # lr = lr.data.cpu().numpy()
            self.val_evaluator.add_batch(sr=sr, hr=hr)
                
            i += 1
            hr,lr,slope = val_prefetcher.next()

        mse, mae, rmse, e_max, slope_mae,psnr = self.val_evaluator.score(self.num_valImage)
        print("Validation:")
        print('[Epoch: %d, numImages: %5d]' % (epoch, self.num_valImage))
        print("MSE:{}, MAE:{}, RMSE:{}, E_MAX:{}, SLOPE_MAE:{}, PSNR:{}".format(mse, mae, rmse, e_max, slope_mae, psnr))
        logging.info("Validation:")
        logging.info('[Epoch: %d, numImages: %5d]' % (epoch, self.num_valImage))
        logging.info("MSE:{}, MAE:{}, RMSE:{}, E_MAX:{}, SLOPE_MAE:{}, PSNR:{}".format(mse, mae, rmse, e_max, slope_mae, psnr))
        if not self.args.test_only:
            self.writer.add_scalar("val/loss_epoch", val_loss/(i+1), epoch)
            self.writer.add_scalar("val/MSE", mse, epoch)
            self.writer.add_scalar("val/PSNR", psnr, epoch)
            self.writer.add_scalar("val/MAE", mae, epoch)
            self.writer.add_scalar("val/RMSE", rmse, epoch)
            self.writer.add_scalar("val/E_MAX", e_max, epoch)
            self.writer.add_scalar("val/Slope_MAE", slope_mae, epoch)
            # 计算参数量
            params_num = sum(p.numel() for p in self.model.parameters())
            print('Loss: %.3f' % (val_loss/i))
            print("\nParams: %.2fM" %(params_num / 1e6))
            # self.loss.end_log(len(self.loader_train))
            # self.error_last = self.loss.log[-1, -1]
            logging.info('Loss: %.3f' %(val_loss / i))
            logging.info("\nParams: %.2fM" %(params_num / 1e6))
            new_pred = mae
            if new_pred < self.best_pred:
                is_best = True
                self.best_pred = new_pred
                self.saver.save_checkpoint({'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred}, is_best=is_best)

        # self.writer.close()
        # torch.set_grad_enabled(True)

