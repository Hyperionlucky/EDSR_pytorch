import argparse
import template

parser = argparse.ArgumentParser(description='EDSR and MDSR')

parser.add_argument('--debug', action='store_true',
                    help='Enables debug mode')
parser.add_argument('--template', default='.',
                    help='You can set various templates in option.py')

# Hardware specifications
parser.add_argument('--n_threads', type=int, default=6,
                    help='number of threads for data loading')
parser.add_argument('--n_GPUs', type=int, default=1,
                    help='number of GPUs')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')

# Data specifications
#parser.add_argument('--dir_data', type=str, default='home/corn/SRdataset/DIV2K',

parser.add_argument('--scale', type=str, default='2',
                    help='super resolution scale')
parser.add_argument('--patch_size', type=int, default=96,
                    help='output patch size')
parser.add_argument('--rgb_range', type=int, default=65535,
                    help='maximum value of RGB')
parser.add_argument('--n_channels', type=int, default=1,
                    help='number of color channels to use')
parser.add_argument('--chop', action='store_true',
                    help='enable memory-efficient forward')
parser.add_argument('--no_augment', action='store_true',
                    help='do not use data augmentation')
parser.add_argument('--traindataset_path',type=str, default="/home/cgd/DEM/ESDR_Pytorch/EDSR_pytorch/Dataset/train_2x_slope.txt")
parser.add_argument('--valdataset_path',type=str, default="/home/cgd/DEM/ESDR_Pytorch/EDSR_pytorch/Dataset/val_2x_slope.txt")


# Model specifications
parser.add_argument('--model', default='EDSR',
                    help='model name')

parser.add_argument('--act', type=str, default='relu',
                    help='activation function')
parser.add_argument('--pre_train', type=str, default='',
                    help='pre-trained model directory')
parser.add_argument('--extend', type=str, default='.',
                    help='pre-trained model directory')
parser.add_argument('--n_resblocks', type=int, default=32,
                    help='number of residual blocks')
parser.add_argument('--n_features', type=int, default=256,
                    help='number of feature maps')
parser.add_argument('--res_scale', type=float, default=1,
                    help='residual scaling')
parser.add_argument('--shift_mean', default=True,
                    help='subtract pixel mean from the input')
parser.add_argument('--dilation', action='store_true',
                    help='use dilated convolution')
parser.add_argument('--precision', type=str, default='single',
                    choices=('single', 'half'),
                    help='FP precision for test (single | half)')


# Training specifications

parser.add_argument('--epochs', type=int, default=1000,
                    help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=16,
                    help='input batch size for training')
parser.add_argument('--test_batch_size', type=int, default=8,
                    help='input batch size for val')
parser.add_argument('--workers', type=int, default=8)


# Optimization specifications
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate')
parser.add_argument('--milestones', type=list, default=[200,300],
                    help='learning rate decay type')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='learning rate decay factor for step decay')

parser.add_argument('--epsilon', type=float, default=1e-8,
                    help='ADAM epsilon for numerical stability')


# Loss specifications
parser.add_argument('--loss_weight', type=list, default=[1,1],
                    help='loss function weight')

# Log specifications

parser.add_argument('--resume', type=int, default=0,
                    help='resume from specific checkpoint')

parser.add_argument('--checkpoint_name', type=str, default='EDSR')
args = parser.parse_args()
template.set_template(args)

args.scale = list(map(lambda x: int(x), args.scale.split('+')))




for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False

