import torch

class DataPrefetcher(object):
    def __init__(self,loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_hr,self.next_lr,self.next_slope = next(self.loader)
        except StopIteration:
            self.next_hr = None
            self.next_lr = None
            self.next_slope = None
            return
        with torch.cuda.stream(self.stream):
            self.next_hr = self.next_hr.cuda(non_blocking=True)
            self.next_lr = self.next_lr.cuda(non_blocking=True)
            self.next_slope = self.next_slope.cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        hr = self.next_hr
        lr = self.next_lr
        slope = self.next_slope
        self.preload()
        return hr,lr,slope