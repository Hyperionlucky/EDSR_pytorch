import glob
import os
from collections import OrderedDict

import torch
class Saver(object):
    def __init__(self,args) -> None:
        self.args = args
        self.directory = os.path.join('experiments_USA', 'X'+str(args.scale),args.model.upper())
        self.runs = sorted(glob.glob(os.path.join(self.directory, 'experiment_*')))
        run_id = 0
        if self.runs:
            for i in self.runs:
                run_id = max(run_id, int(i.split('_')[-1]))
            run_id += 1
        
        self.experiment_dir = os.path.join(self.directory,'experiment_{}'.format(str(run_id)))
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)
    def save_checkpoint(self, state, is_best, filename='checkpoint.pth',best_filename="best_val.txt"):
        filename = os.path.join(self.experiment_dir,filename)
        torch.save(state, filename)
        if is_best:
            best_pred = state['best_pred']
            epoch = state['epoch']
            with open(os.path.join(self.experiment_dir,best_filename),"a") as f:
                f.write(str(best_pred)+"--------------"+str(epoch)+"\n")

    def save_experiment_config(self):
        log_file = os.path.join(self.experiment_dir, "parameters.txt")
        log_file = open(log_file, "w")
        p = OrderedDict()
        p['lr'] = self.args.lr
        p['optimizer'] = self.args.optimizer
        p['batch_size'] = self.args.batch_size
        p['test_batch_size'] = self.args.test_batch_size
        p['workers'] = self.args.workers
        p['epochs'] = self.args.epochs
        p['scale'] = self.args.scale
        p['loss'] = str(self.args.loss_weight[0]) + ' *L1 Loss   ' + str(self.args.loss_weight[1]) + ' *Slopeloss'
        p['patch_size'] = self.args.patch_size
        for key,val in p.items():
            log_file.write(key+":"+str(val)+"\n")
        log_file.close()


