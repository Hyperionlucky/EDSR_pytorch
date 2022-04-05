import numpy as np
import random
import torch

def setup_seed(seed):
    np.random.seed(seed=seed)
    random.seed(seed)
    torch.manual_seed(seed=seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = True