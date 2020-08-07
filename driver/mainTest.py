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
from model.trainer import HierarchicalSRTrainer
from collections import defaultdict
from time import  time

configFile="../config/prosrgan.yaml" #Config File
n_cpu=8 #number of cpu threads to use during batch generation
file_separator = "/"


def modelling(args):

    training_dataset = ImageDataset(phase=Phase.TRAIN,
                                    img_dir= args.train.dataset.path,
                                    img_type=args.xtra.img_type,
                                    mean = args.train.dataset.mean,
                                    stddev = args.train.dataset.stddev,
                                    scales = args.data.scale,
                                    high_res=args.xtra.high_res)

    dataloader = DataLoader(
        training_dataset,
        batch_size=args.train.batch_size,
        shuffle=True,
        num_workers=n_cpu,
    )

    test_dataset = ImageDataset(phase=Phase.TEST,
                                    img_dir=args.test.dataset.path,
                                    img_type=args.xtra.img_type,
                                    mean=args.test.dataset.mean,
                                    stddev=args.train.dataset.stddev,
                                    scales=args.data.scale,
                                    high_res=args.xtra.high_res)

    testing_data_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=True,
    )

    # checkpoint is the directory where checkpoints are read from and stored
    trainer = HierarchicalSRTrainer(args, dataloader, training_dataset, save_dir=args.xtra.out_path)

    best_saved = 0

    steps_per_epoch = len(training_dataset)
    errors_accum = defaultdict(list)
    errors_accum_prev = defaultdict(lambda: 0)
    total_steps = 0

    trainer.testcuda()

    print("RIKI: BEGIN EPOCHS....")

    epoch = 0

    #for epoch in range(0, args.train.epochs):
    while epoch < args.train.epochs:
        iter_start_time = time()
        trainer.set_train()

        intermediate_steps = 0
        recalibrating = False

        for i, imgs in enumerate(dataloader):
            '''if (i+1)% freq == 0:
                print("RIKI: EPOCH:%d BATCH: %d SCALE: %d BLEND: %f" % ( epoch, i, int(imgs['scale'][0].item()), trainer.blend))'''

            trainer.set_input(imgs['lr'], imgs['hr'], imgs['bicubic'], imgs['scale'][0].item())

            # GENERATOR MAKING HR IMAGE
            gen_hr = trainer.forward()

            # SCALE SWITCHING HAPPENS HERE IF #EPOCHS IS REACHED
            # L1 LOSS ALSO CALCULATED HERE - INSIDE backward()
            ret = trainer.optimize_parameters()

            if ret:
                print("SWITCHING GEARS...RESTARTING EPOCH "+str(epoch))
                total_steps-=intermediate_steps
                intermediate_steps = 0
                epoch-=1
                recalibrating = True
                break

            #dict: eg. 'l1_x2'
            errors = trainer.get_current_errors()
            for key, item in errors.items():
                errors_accum[key].append(item)

            total_steps += 1
            intermediate_steps += 1


            # HOW OFTEN TO PRINT ERROR STATUS TO CONSOLE: print_errors_freq
            if total_steps % args.train.io.print_errors_freq == 0:
                for key, item in errors.items():
                    if len(errors_accum[key]):
                        errors_accum[key] = np.nanmean(errors_accum[key])
                    if np.isnan(errors_accum[key]):
                        errors_accum[key] = errors_accum_prev[key]
                message = 'RIKI: EPOCH:%d BATCH: %d SCALE: %d BLEND: %f ERRORS: %s' % (epoch, i, int(imgs['scale'][0].item()), trainer.blend, str(errors_accum[key]))
                print(message)
                errors_accum_prev = errors_accum
                #t = time() - iter_start_time
                #iter_start_time = time()
                errors_accum = defaultdict(list)

            if total_steps % args.train.io.save_img_freq == 0:
                key='l1_x'+str(trainer.model_scale)

                # SAVING IMAGES IN BETWEEN EPOCHS
                op = ImageManipulator.combine_images(imgs['lr'], imgs['hr'], gen_hr, trainer.model_scale)
                fname = args.xtra.save_path+file_separator+"opimg_x"+str(trainer.model_scale)+"_"+str(epoch)+"_"+str(total_steps)+".jpg"
                print('RIKI: SAVE TO ' + fname)
                save_image(op, fname, normalize=True)

        epoch += 1
        if recalibrating:
            continue

        ################# update learning rate  #################
        # IF THE BEST EPOCH WAS SEEN A WHILE BACK...IT'S TIME TO UPDATE THE LEARNING RATE BY 1/2
        if (epoch - trainer.best_epoch) > args.train.lr_schedule_patience:
            #trainer.save('last_lr_%g' % trainer.lr, epoch, trainer.lr)
            print("RIKI: UPDATING LR...")
            trainer.update_learning_rate()

        #test_dataset.set_scales(trainer.current_scale_id)

        print("RIKI: BEGIN EVALUATION... "+str(trainer.current_scale_id)+" "+str(training_dataset.current_scale_id)+" "+str(training_dataset.current_scale))
        if len(testing_data_loader)>0:
            with torch.no_grad():

                # use validation set
                trainer.set_eval()
                trainer.reset_eval_result()

                test_dataset.copy_scales(training_dataset.current_scale)
                #print("RIKI: CHECK "+str(test_dataset.current_scale))
                for i, imgs in enumerate(testing_data_loader):
                    trainer.set_input(imgs['lr'],imgs['hr'],imgs['bicubic'],imgs['scale'])
                    #PSNR COMPUTATIONS HERE
                    trainer.evaluate()

                test_result = trainer.get_current_eval_result()

                trainer.update_best_eval_result(epoch, test_result)
                print(
                    'RIKI: eval at epoch %d : ' % epoch + ' | '.join([
                        '{}: {:.07f}'.format(k, v)
                        for k, v in test_result.items()
                    ]))

                print(
                    'RIKI: best so far %d : ' % trainer.best_epoch + ' | '.join([
                        '{}: {:.07f}'.format(k, v)
                        for k, v in trainer.best_eval.items()
                    ]))

                # IF CURRENT EPOCH IS THE BEST EPOCH SO FAR
                if trainer.best_epoch == epoch:
                    if len(trainer.best_eval) > 1:

                        # select only upto current training scale
                        best_key = ["psnr_x%d" % trainer.opt.data.scale[s_idx]
                                    for s_idx in range(trainer.current_scale_id + 1)]
                        best_key = [k for k in best_key
                                    if trainer.best_eval[k] == test_result[k]]

                    else:
                        best_key = list(trainer.best_eval.keys())
                    print('RIKI: BEST KEY: '+str(epoch) + '__'+ str(best_key))

                    # COPY trainer to best_trainer
                    trainer.best_trainer = trainer.save(epoch, trainer.lr, trainer.current_scale_id, False)

        if epoch % args.train.io.save_model_freq == 0:
            print('RIKI: STORING THE MODEL....AFTER A WHILE')
            trainer.store()





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

    #pprint(params)
    modelling(params)