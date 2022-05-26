import numpy as np
# np.seterr(divide='ignore',invalid='ignore')
import math

class Evaluator(object):
    def __init__(self,num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)
    def Pixel_Accuracy(self):
        acc = np.diag(self.confusion_matrix).sum()/self.confusion_matrix.sum()
        return acc

    def pixel_Accuracy_class(self):
        acc_class = np.diag(self.confusion_matrix)/self.confusion_matrix.sum(axis=1)
        return acc_class

    def Mean_Intersection_over_Union(self):
        TP = np.diag(self.confusion_matrix)
        FP_FN_TP = np.sum(self.confusion_matrix,axis=1) + np.sum(self.confusion_matrix,axis=0)-np.diag(self.confusion_matrix)
        MIoU_class  = TP/FP_FN_TP
        MIoU = np.nanmean(MIoU_class)
        return MIoU_class,MIoU

    def F1_score(self):
        TP = np.diag(self.confusion_matrix)
        FP = np.sum(self.confusion_matrix,axis=1)-TP
        FN = np.sum(self.confusion_matrix,axis=0)-TP

        precision = TP/(TP+FP+1e-9)
        recall = TP/(TP+FN+1e-9)
        F1_class = 2*precision*recall/(precision+recall+1e-9)
        F1 = np.nanmean(F1_class)

        return precision,recall,F1_class,F1

    def cal_kappa(self,hist):
        if hist.sum() == 0:
            po = 0
            pe = 1
            kappa = 0
        else:
            po = np.diag(hist).sum() / hist.sum()
            pe = np.matmul(hist.sum(1), hist.sum(0).T) / hist.sum() ** 2
            if pe == 1:
                kappa = 0
            else:
                kappa = (po - pe) / (1 - pe)
        return kappa

    def score(self):
        hist = self.confusion_matrix
        hist_fg = self.confusion_matrix[1:,1:]
        c2hist = np.zeros((2, 2))
        c2hist[0][0] = hist[0][0]
        c2hist[0][1] = hist.sum(1)[0] - hist[0][0]
        c2hist[1][0] = hist.sum(0)[0] - hist[0][0]
        c2hist[1][1] = hist_fg.sum()
        hist_n0 = hist.copy()
        hist_n0[0][0] = 0
        kappa_n0 = self.cal_kappa(hist_n0)
        iu = np.diag(c2hist) / (c2hist.sum(1) + c2hist.sum(0) - np.diag(c2hist))
        IoU_fg = iu[1]
        IoU_mean = (iu[0] + iu[1]) / 2
        Sek = (kappa_n0 * math.exp(IoU_fg)) / math.e
        Score = 0.3 * IoU_mean + 0.7 * Sek

        return IoU_mean,Sek,Score

    def _generate_matrix(self,label,pred):
        mask = (label>=0) & (label<self.num_class)
        label = self.num_class * label[mask].astype('int') + pred[mask]
        count = np.bincount(label,minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class,self.num_class)
        return confusion_matrix

    def add_batch(self,label,pred):
        # pred = pred.astype(np.uint8)
        if label.shape != pred.shape:
            label = np.squeeze(label,axis=1)
        assert label.shape == pred.shape
        self.confusion_matrix += self._generate_matrix(label,pred)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,)*2)
if __name__ == '__main__':
    ev = Evaluator(6)
    pass