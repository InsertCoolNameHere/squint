import numpy as np
import os
import os.path as osp
import random
import sys
import torch
import yaml
from utils.dotdict import DotDict
from data.ImageDataset import ImageDataset
import utils.im_manipulation as ImageManipulator
import matplotlib.pyplot as plt
from torchvision.utils import save_image, make_grid
from utils.Phase import Phase
import utils as utils
from torch.utils.data import DataLoader

configFile="../config/prosrgan.yaml" #Config File
n_cpu=8 #number of cpu threads to use during batch generation


def modelling(args):
    seed = args.xtra.seed
    print(args.xtra.out_path)
    print(args.test.dataset.path)
    #multiplier between lr and hr
    current_scale_idx=2



def meth(num_init_features, bn_size,
                 growth_rate, ps_woReLU, level_config, level_compression,
                 res_factor, max_num_feature, max_scale, **kwargs):
    print("HERE")



if __name__ == '__main__':

    with open(configFile) as file:
        try:
            params = yaml.load(file, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            print(exc)
            sys.exit(0)

    params = DotDict(params)

    if not osp.isdir(params.xtra.out_path):
        os.makedirs(params.xtra.out_path)

    #SAVING PARAMETERS FOR RESTART....NEEDS WORK
    #np.save(osp.join(params.ip.out_path, 'params'), params)

    experiment_id = osp.basename(params.xtra.out_path)

    print('experiment ID: {}'.format(experiment_id))

    '''if args.visdom:
        from lib.prosr.visualizer import Visualizer
        visualizer = Visualizer(experiment_id, port=args.visdom_port)'''

    #pprint(params)
    modelling(params)
    params.G.max_scale = max(params.data.scale)
    meth(**params.G)