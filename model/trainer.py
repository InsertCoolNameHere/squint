
import os
import torch
import numpy as np
from collections import OrderedDict
from utils.im_manipulation import tensor2im,eval_psnr_and_ssim
from model.gen_dis import HierarchicalGenerator
from data import ImageDataset
import utils.im_manipulation as ImageManipulator
from torchvision.utils import save_image, make_grid

class HierarchicalSRTrainer(object):
    def __init__(self,opt, data_loader, dataset:ImageDataset, save_dir="data/checkpoints"):
        super(HierarchicalSRTrainer, self).__init__()


        self.input = torch.zeros(
            opt.train.batch_size, 3, 48, 48,
            dtype=torch.float32)
        self.label = torch.zeros_like(
            self.input, dtype=torch.float32)
        self.interpolated = torch.zeros_like(
            self.label, dtype=torch.float32)

        self.opt = opt
        self.save_dir = save_dir

        self.dataset = dataset
        # NUMBER OF BATCHES POSSIBLE IN 1 RUN OF DATASET
        self.dataLen = len(data_loader)
        self.start_epoch = 0
        self.progress = self.start_epoch / opt.train.epochs
        self.blend = 1
        # INDEX OF THE CURRENT MODEL SCALE
        self.current_scale_id=0
        self.model_scale = self.opt.data.scale[self.current_scale_id]

        # Dictionary of scalewise best evaluated scores
        self.best_eval = OrderedDict([('psnr_x%d' % s, 0.0) for s in opt.data.scale])
        # Dictionary of scalewise all evaluated scores
        self.eval_dict = OrderedDict([('psnr_x%d' % s, []) for s in opt.data.scale])

        # TENSOR TO NUMPY ARRAY
        self.tensor2im = lambda t: tensor2im(t, mean=opt.train.dataset.mean,stddev=opt.train.dataset.stddev)

        opt.G.max_scale = max(opt.data.scale)

        # INITIALIZING THE GENERATOR
        self.net_G = HierarchicalGenerator(**opt.G).cuda()
        self.best_epoch = 0

        self.optimizer_G = torch.optim.Adam(
            [p for p in self.net_G.parameters() if p.requires_grad],
            lr=self.opt.train.lr,
            betas=(0.9, 0.999),
            eps=1.0e-08)
        self.lr = self.opt.train.lr

        self.l1_criterion = torch.nn.L1Loss()

        self.best_trainer = self.save(0, 0, 0, False)

        # INITIALIZATION OF SCALE & NETWORK PARAMETERS
        #self.reset_curriculum_for_dataloader()

    def testcuda(self):
        cuda = torch.cuda.is_available()
        print("GPU AVAILABLE:" + str(cuda))
        self.input = self.input.cuda(non_blocking=True)
        self.label = self.label.cuda(non_blocking=True)
        self.interpolated = self.interpolated.cuda(non_blocking=True)
        self.net_G = self.net_G.cuda()
        self.l1_criterion = self.l1_criterion.cuda()

    # THE LR AND THE BICUBIC INTERPOLATED HR IMAGE AS INPUT
    # RETURNS THE GENERATED HR OUTPUT
    def forward(self):
        # GETTING THE CURRENT BLEND VALUE
        if self.current_scale_id != 0:
            lo, hi = self.opt.train.growing_steps[self.current_scale_id * 2 - 2 : self.current_scale_id * 2]
            self.blend = min((self.progress - lo) / (hi - lo), 1)
            assert self.blend >= 0 and self.blend <= 1
        else:
            self.blend = 1

        self.output = self.net_G(self.input, upscale_factor=self.model_scale, blend = self.blend) + self.interpolated
        return self.output

    def set_input(self, lr, hr, bic, scale):
        self.input.resize_(lr.size()).copy_(lr)
        self.label.resize_(hr.size()).copy_(hr)
        self.interpolated.resize_(bic.size()).copy_(bic)
        #self.model_scale = scale

    def set_train(self):
        self.net_G.train()
        self.isTrain = True

    def backward(self):
        self.compute_loss()
        self.loss.backward()

    # GENERATOR LOSS
    def compute_loss(self):
        self.loss = 0
        self.l1_loss = self.l1_criterion(self.output, self.label)
        self.loss += self.l1_loss

    def optimize_parameters(self):
        self.optimizer_G.zero_grad()
        self.backward()
        self.optimizer_G.step()
        # update progress
        ret = self.increment_training_progress()
        return ret

    # THIS HELPS JUMP THE SCALE DURING TRAINING
    def increment_training_progress(self):
        ret = False
        """increment self.progress and D, G scale_idx"""
        # 1/2/3 = 1/(2*3) = .166
        # PITSTOPS AT 12%, 25%, 45%, 60%, 100%
        self.progress += 1 / self.dataLen / self.opt.train.epochs

        # IF THE #EPOCHS HAS HIT A BREAK-POINT - THIS MEANS A SCALE CHANGE IS TO BE INITIATED
        if self.progress > self.opt.train.growing_steps[self.current_scale_id * 2]:
            if self.current_scale_id < len(self.opt.data.scale) - 1:
                # SAVING CURRENT BEST
                print('RIKI: STORING CURRENT BEST MODEL....DUE TO SCALE UPDATE ')
                self.store()


                print('RIKI: trainer TIME TO INCREASE TRAINING SCALE')
                self.current_scale_id += 1
                self.net_G.current_scale_idx = self.current_scale_id
                self.model_scale*=2

                # SO THAT THE DATA LOADER LOADS THE CORRECT IMAGES
                self.dataset.set_scales(self.current_scale_id)
                # REPEAT self.opt.data.scale[i] FOR self.current_scale_idx + 1 TIMES
                training_scales = [
                    self.opt.data.scale[i]
                    for i in range(self.current_scale_id + 1)
                ]

                print('RIKI: UPDATED TRAINING SCALES: {}'.format(str(training_scales)))
                ret = True
            elif self.current_scale_id == len(self.opt.data.scale) - 1:
                print("FINISHED ALL EPOCHS AND SCALES...")

        return ret


    def get_current_errors(self):
        d = OrderedDict()
        if hasattr(self, 'l1_loss'):
            d['l1_x%d' % self.model_scale] = self.l1_loss.item()

        return d

    def set_learning_rate(self, lr, optimizer):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def update_learning_rate(self):
        """update learning rate with exponential decay"""
        lr = self.lr * self.opt.train.lr_decay
        if lr < self.opt.train.smallest_lr:
            return
        self.set_learning_rate(lr, self.optimizer_G)
        print('update learning rate: %f -> %f' % (self.lr, lr))
        self.lr = lr

    # MODEL EVALUATION
    def evaluate(self):

        self.output = self.net_G(self.input, upscale_factor=self.model_scale,
                                 blend=self.blend) + self.interpolated

        im1 = self.tensor2im(self.label)
        im2 = self.tensor2im(self.output)
        eval_res = {
            'psnr_x%d' % self.model_scale:
                eval_psnr_and_ssim(im1, im2, self.model_scale)[0]
        }
        for k, v in eval_res.items():
            self.eval_dict[k].append(v)
        return eval_res

    # TEST EVALUATION AND IMAGE GENERATION
    def evaluate_and_generate(self, act_fname, output_image_path):

        self.output = self.net_G(self.input, upscale_factor=self.model_scale,
                                 blend=self.blend) + self.interpolated

        im1 = self.tensor2im(self.label)
        im2 = self.tensor2im(self.output)

        err = eval_psnr_and_ssim(im1, im2, self.model_scale)[0]
        eval_res = {
            'psnr_x%d' % self.model_scale: err
        }

        err_str = '{:.02f}'.format(err)
        op = ImageManipulator.combine_images_test(self.input, self.label, self.interpolated, self.output, self.model_scale)
        fname = output_image_path + self.opt.xtra.file_sep + act_fname+"_x" + str(self.model_scale) + "_"+err_str+ ".jpg"
        print('RIKI: SAVE TO ' + fname)
        save_image(op, fname, normalize=True)

        return eval_res

    def set_eval(self):
        self.net_G.eval()
        self.isTrain = False

    def reset_eval_result(self):
        for k in self.eval_dict:
            self.eval_dict[k].clear()

    # AVERAGING CURRENT PSNRs FROM eval_dict
    def get_current_eval_result(self):
        eval_result = OrderedDict()
        for k, vs in self.eval_dict.items():
            eval_result[k] = 0
            if vs:
                for v in vs:
                    eval_result[k] += v
                eval_result[k] /= len(vs)
        return eval_result

    def update_best_eval_result(self, epoch, current_eval_result=None):
        if current_eval_result is None:
            eval_result = self.get_current_eval_result()
        else:
            eval_result = current_eval_result
        is_best_sofar = any(
            [np.round(eval_result[k],2) > np.round(v,2) for k, v in self.best_eval.items()])
        #print("RIKI: trainer IS BEST SO FAR: "+str(is_best_sofar))
        if is_best_sofar:
            self.best_epoch = epoch
            self.best_eval = {
                k: max(self.best_eval[k], eval_result[k])
                for k in self.best_eval
            }

    # SAVING TO DISK
    def store(self):

        savepath = self.best_trainer['network']['path']
        torch.save(self.best_trainer['network'], savepath)

        savepath1 = self.best_trainer['optim']['path']
        torch.save(self.best_trainer['optim'], savepath1)

        print("RIKI: STORED "+savepath+" "+savepath1)

    # SAVING TO MEMORY
    def save(self, epoch, lr, scale, make_save=False):

        to_save = {
            'network': self.save_network(self.net_G, 'G', str(epoch), str(scale), make_save),
            'optim': self.save_optimizer(self.optimizer_G, 'G', epoch, lr, str(scale), make_save),
        }

        print("RIKI: SAVED LATEST MODEL %d %d %d...DISK: %s"%(epoch, lr, scale, str(make_save)))
        return to_save

    def save_network(self, network, network_label, epoch_label, scale, make_save=False):
        network = network.module if isinstance(
            network, torch.nn.DataParallel) else network
        #save_filename = '%s_net_%s_x%s.pth' % (epoch_label, network_label, scale)
        save_filename = 'net_%s_x%s.pth' % (network_label, scale)
        save_path = os.path.join(self.save_dir, save_filename)
        to_save = {
            'state_dict': network.state_dict(),
            'path': save_path
        }

        if make_save:
            torch.save(to_save, save_path)

        return to_save

    def save_optimizer(self, optimizer, network_label, epoch, lr, scale, make_save=False):
        save_filename = 'optim_%s_x%s.pth' % (network_label, scale)
        save_path = os.path.join(self.save_dir, save_filename)

        to_save = {
            'state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'lr': lr,
            'path': save_path
        }

        if make_save:
            torch.save(to_save, save_path)

        return to_save

    def load(self, resume_from):
        self.load_network(self.net_G, resume_from[0])
        self.load_optimizer(self.optimizer_G, resume_from[1])

    def load_network(self, network, saved_path):
        network = network.module if isinstance(
            network, torch.nn.DataParallel) else network
        loaded_state = torch.load(saved_path)['state_dict']
        loaded_param_names = set(loaded_state.keys())

        # allow loaded states to contain keys that don't exist in current model
        # by trimming these keys;
        own_state = network.state_dict()
        extra = loaded_param_names - set(own_state.keys())
        if len(extra) > 0:
            print('Dropping ' + str(extra) + ' from loaded states')
        for k in extra:
            del loaded_state[k]

        try:
            network.load_state_dict(loaded_state)
        except KeyError as e:
            print(e)
        print('RIKI: loaded network state from ' + saved_path)

    def load_optimizer(self, optimizer, saved_path):

        data = torch.load(saved_path)
        loaded_state = data['state_dict']
        optimizer.load_state_dict(loaded_state)

        # Load more params
        self.start_epoch = data['epoch']
        self.lr = data['lr']

        print('RIKI: loaded optimizer state from ' + saved_path)