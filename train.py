import torch
from torch import nn
from opt import get_opts
import os
import glob
import imageio
import numpy as np
import cv2
from einops import rearrange
from math import sqrt
from collections import defaultdict
import random
# data
from torch.utils.data import DataLoader
from datasets import dataset_dict
from datasets.ray_utils import axisangle_to_R, get_rays

# models
from kornia.utils.grid import create_meshgrid3d
from models.networks import NGP,E_attr,implicit_mask
from models.rendering import render, MAX_SAMPLES

# optimizer, losses
#from apex.optimizers import FusedAdam
from torch.optim.lr_scheduler import CosineAnnealingLR


# metrics
from torchmetrics import (
    PeakSignalNoiseRatio, 
    StructuralSimilarityIndexMeasure
)
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from metrics import *
# pytorch-lightning
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.distributed import all_gather_ddp_if_available

from utils import slim_ckpt, load_ckpt,PosEmbedding

import warnings; warnings.filterwarnings("ignore")

torch.backends.cudnn.benchmark = True
###
from losses import loss_dict
# optimizer, scheduler, visualization
from utils_2 import *

def depth2img(depth):
    depth = (depth-depth.min())/(depth.max()-depth.min())
    depth_img = cv2.applyColorMap((depth*255).astype(np.uint8),
                                  cv2.COLORMAP_TURBO)

    return depth_img

from datasets import global_val

class NeRFSystem(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        #self.warmup_steps = 3052 # 256
        #self.update_interval = 256 # 763

        self.warmup_steps = 256  # 256
        self.update_interval = 256  # 763

        #self.loss = NeRFLoss(lambda_distortion=self.hparams.distortion_loss_w)
        """
        self.train_psnr = PeakSignalNoiseRatio(data_range=1)
        self.val_psnr = PeakSignalNoiseRatio(data_range=1)
        self.val_ssim = StructuralSimilarityIndexMeasure(data_range=1)
        if self.hparams.eval_lpips:
            self.val_lpips = LearnedPerceptualImagePatchSimilarity('vgg')
            for p in self.val_lpips.net.parameters():
                p.requires_grad = False
        """

        rgb_act = 'None' if self.hparams.use_exposure else 'Sigmoid'
        self.model = NGP(scale=self.hparams.scale, rgb_act=rgb_act)
        G = self.model.grid_size
        self.model.register_buffer('density_grid',
            torch.zeros(self.model.cascades, G**3))
        self.model.register_buffer('grid_coords',
            create_meshgrid3d(G, G, G, False, dtype=torch.int32).reshape(-1, 3))
        
        ###
        self.loss = loss_dict['hanerf'](hparams, coef=1)

        self.models_to_train = []
        self.models_to_train += [self.model]
        self.embedding_uv = PosEmbedding(10-1, 10)
        #self.embedding_uv = PosEmbedding(14-1, 15)

        if hparams.encode_a:
            self.enc_a = E_attr(3, hparams.N_a)
            self.models_to_train += [self.enc_a]
            self.embedding_a_list = [None] * hparams.N_vocab

        if hparams.use_mask:
            self.implicit_mask = implicit_mask()
            self.models_to_train += [self.implicit_mask]
            self.embedding_view = torch.nn.Embedding(hparams.N_vocab, 128) # 128
            self.models_to_train += [self.embedding_view]

     
        ###
    def forward(self, rays, ts, whole_img, W, H, rgb_idx, uv_sample, test_blender):
        results = defaultdict(list)
        kwargs ={}
        kwargs['exp_step_factor'] =1/256
        if self.hparams.encode_a:
            if test_blender:
                kwargs['a_embedded_from_img'] = self.embedding_a_list[0] if self.embedding_a_list[0] != None else self.enc_a(whole_img)
            else:
                kwargs['a_embedded_from_img'] = self.enc_a(whole_img) # (1,48)

            if self.hparams.encode_random:
                idexlist = [k for k,v in enumerate(self.embedding_a_list) if v != None]
                if len(idexlist) == 0:
                    kwargs['a_embedded_random'] = kwargs['a_embedded_from_img']
                else:
                    kwargs['a_embedded_random'] = self.embedding_a_list[random.choice(idexlist)]
        B = rays.shape[0]
        for i in range(0, B, self.hparams.chunk):
            rendered_ray_chunks = \
                    render(self.model,
                                rays[i:i+self.hparams.chunk],
                                ts[i:i+self.hparams.chunk],
                                self.hparams.use_disp,
                                self.hparams.chunk, # chunk size is effective in val mode
                                self.train_dataset.white_back,
                                **kwargs)
            torch.cuda.synchronize() ###   
            for k, v in rendered_ray_chunks.items():
                results[k] += [v]

        for k, v in results.items():
            results[k] = torch.cat(v, dim=0)
            #results[k] = torch.stack(v,dim=0)

        if self.hparams.use_mask:
            if test_blender:
                results['out_mask'] = torch.zeros(results['rgb_fine'].shape[0], 1).to(results['rgb_fine'])
            else:
                
                uv_embedded = self.embedding_uv(uv_sample)
                results['out_mask'] = self.implicit_mask(torch.cat((self.embedding_view(ts), uv_embedded), dim=-1))
                #results['out_mask'] = self.implicit_mask(torch.cat((self.embedding_view(ts), uv_sample), dim=-1))
                """
                ###也加入cnn
                uv_embedded = self.embedding_uv(uv_sample) # (1024,62)
                cnn_mask = kwargs['a_embedded_from_img'].repeat(1024,1) # (1024,64)
                results['out_mask'] = self.implicit_mask(torch.cat((cnn_mask,self.embedding_view(ts), uv_embedded), dim=-1)) # input(1024,254)
                """

        if self.hparams.encode_a:
            results['a_embedded'] = kwargs['a_embedded_from_img']
            if self.hparams.encode_random:
                results['a_embedded_random'] = kwargs['a_embedded_random']
                rec_img_random = results['rgb_random'].view(1, H, W, 3).permute(0, 3, 1, 2) * 2 - 1
                results['a_embedded_random_rec'] = self.enc_a(rec_img_random)
                self.embedding_a_list[ts[0]] = kwargs['a_embedded_from_img'].clone().detach()
        
        return results

    def setup(self, stage):
        dataset = dataset_dict[self.hparams.dataset_name]
        kwargs = {'root_dir': self.hparams.root_dir}
        if self.hparams.dataset_name == 'phototourism':
            kwargs['img_downscale'] = self.hparams.img_downscale
            #kwargs['val_num'] = self.hparams.num_gpus
            kwargs['use_cache'] = self.hparams.use_cache
            kwargs['batch_size'] = self.hparams.batch_size
            kwargs['scale_anneal'] = self.hparams.scale_anneal
            kwargs['min_scale'] = self.hparams.min_scale
            kwargs['masks_dir'] = self.hparams.masks_dir
            #kwargs['ndc_rays'] = self.hparams.ndc_rays
        self.train_dataset = dataset(split='train', **kwargs)
        self.val_dataset = dataset(split='val', **kwargs)

    def configure_optimizers(self):

        self.optimizer = get_optimizer(self.hparams, self.models_to_train)
        scheduler = get_scheduler(self.hparams, self.optimizer)
        return [self.optimizer], [scheduler]
      
    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          num_workers=16,
                          persistent_workers=True,
                          batch_size=None,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          num_workers=8,
                          batch_size=None,
                          pin_memory=True)


    def training_step(self, batch, batch_nb, *args):
        if self.global_step%self.update_interval == 0:
            self.model.update_density_grid(0.01*MAX_SAMPLES/3**0.5,
                                           warmup=True,
                                           erode=False)
                                           # warmup=True,

        rays, ts = batch['rays'].squeeze(), batch['ts'].squeeze()
        rgbs = batch['rgbs'].squeeze()
        uv_sample = batch['uv_sample'].squeeze()
        mask = batch['mask']
        if self.hparams.encode_a or self.hparams.use_mask:
            whole_img = rearrange(batch['whole_img'],'c h w -> 1 c h w ')
            rgb_idx = batch['rgb_idx']
        else:
            whole_img = None
            rgb_idx = None
        H = int(sqrt(rgbs.size(0)))
        W = int(sqrt(rgbs.size(0)))
        
        # random_mask = torch.full((1024,1), 0.3).to(mask)
        # mask_cat = random_mask + mask
        # mask_cat_clamp = torch.clamp(mask_cat, 0, 1)
        
        test_blender = False
        results = self(rays, ts, whole_img, W, H, rgb_idx, uv_sample, test_blender)
        loss_d, AnnealingWeight = self.loss(results, rgbs, self.hparams, self.global_step, mask)
        loss = sum(l for l in loss_d.values())

        with torch.no_grad():
            typ = 'fine' if 'rgb_fine' in results else 'coarse'
            psnr_ = psnr(results['rgb'], rgbs)

        self.log('lr', get_learning_rate(self.optimizer))
        self.log('train/loss', loss)
        self.log('train/AnnealingWeight', AnnealingWeight)
        self.log('train/min_scale_cur', batch['min_scale_cur'])
        for k, v in loss_d.items():
            self.log(f'train/{k}', v)
        self.log('train/psnr', psnr_)

        if (self.global_step + 1) % 100 == 0:
            img = results['rgb'].detach().view(H, W, 3).permute(2, 0, 1).cpu() # (3, H, W)
            img_gt = rgbs.detach().view(H, W, 3).permute(2, 0, 1).cpu() # (3, H, W)
            depth = visualize_depth(results['depth'].detach().view(H, W)) # (3, H, W)
            if self.hparams.use_mask:
                out_put_mask = results['out_mask'].detach().view(H, W, 1).permute(2, 0, 1).repeat(3, 1, 1).cpu() # (3, H, W)
                mlp_out_mask = results['out_mask']#.detach().view(H, W, 1).permute(2, 0, 1).repeat(3, 1, 1).cpu() # (3, H, W)
                if 'rgb_random' in results:
                    img_random = results['rgb_random'].detach().view(H, W, 3).permute(2, 0, 1).cpu() # (3, H, W)
                    my_mask = mask.detach().view(H, W, 1).permute(2, 0, 1).repeat(3, 1, 1).cpu() # (3, H, W)
                    or_mask = torch.max(mask, mlp_out_mask)
                    
                    or_mask = or_mask.detach().view(H, W, 1).permute(2, 0, 1).repeat(3, 1, 1).cpu() # (3, H, W)) 
                    # merged_mask = torch.max(mask, mlp_out_mask)
                    # mask_intersection = (mask == 1) & (mlp_out_mask > 0)
                    # merged_mask[mask_intersection] = mask[mask_intersection]
                    # mask = mask.detach().view(H, W, 1).permute(2, 0, 1).repeat(3, 1, 1).cpu() # (3, H, W)
                    # mlp_out_mask = mlp_out_mask.detach().view(H, W, 1).permute(2, 0, 1).repeat(3, 1, 1).cpu() # (3, H, W)
                    # merged_mask = merged_mask.detach().view(H, W, 1).permute(2, 0, 1).repeat(3, 1, 1).cpu() # (3, H, W)
                    stack = torch.stack([img_gt, img, img_random, depth, out_put_mask, my_mask, or_mask]) # (4, 3, H, W)
                    self.logger.experiment.add_images('train/GT_pred_depth_random_mask',
                                                      stack, self.global_step)
                else:
                    stack = torch.stack([img_gt, img, depth, mask]) # (3, 3, H, W)
                    self.logger.experiment.add_images('train/GT_pred_depth_mask',
                                                      stack, self.global_step)
            elif 'rgb_random' in results:
                img_random = results['rgb_random'].detach().view(H, W, 3).permute(2, 0, 1).cpu() # (3, H, W)
                mask = mask.detach().view(H, W, 1).permute(2, 0, 1).repeat(3, 1, 1).cpu() # (3, H, W)
                random_mask = random_mask.detach().view(H, W, 1).permute(2, 0, 1).repeat(3, 1, 1).cpu() # (3, H, W)
                mask_cat = mask_cat.detach().view(H, W, 1).permute(2, 0, 1).repeat(3, 1, 1).cpu() # (3, H, W)
                mask_cat_clamp = mask_cat_clamp.detach().view(H, W, 1).permute(2, 0, 1).repeat(3, 1, 1).cpu() # (3, H, W)
                
                stack = torch.stack([img_gt, img, img_random, depth, mask, random_mask, mask_cat, mask_cat_clamp]) # (4, 3, H, W)
                self.logger.experiment.add_images('train/GT_pred_depth_random',
                                                  stack, self.global_step)
            else:
                stack = torch.stack([img_gt, img, depth]) # (3, 3, H, W)
                self.logger.experiment.add_images('train/GT_pred_depth',
                                                  stack, self.global_step)
        
        return loss

    def validation_step(self, batch, batch_nb):
                                      
       
        #if self.global_step%self.update_interval == 0:
        #torch.cuda.empty_cache()
        
        self.model.update_density_grid(0.01*MAX_SAMPLES/3**0.5,
                                           warmup=self.global_step<self.warmup_steps,
                                           erode=True)
        
        rays, ts = batch['rays'].squeeze(), batch['ts'].squeeze()
        rgbs =  batch['rgbs'].squeeze()
        if self.hparams.dataset_name == 'phototourism':
            uv_sample = batch['uv_sample'].squeeze()
            WH = batch['img_wh']
            #W, H = WH[0, 0].item(), WH[0, 1].item()
            W, H = WH[0].item(), WH[1].item()
        else:
            W, H = self.hparams.img_wh
            uv_sample = None

        if self.hparams.encode_a or self.hparams.use_mask or self.hparams.deocclusion:
            if self.hparams.dataset_name == 'phototourism':
                whole_img = batch['whole_img']
            else:
                whole_img = rgbs.view(1, H, W, 3).permute(0, 3, 1, 2) * 2 - 1
            rgb_idx = batch['rgb_idx']
        else:
            whole_img = None
            rgb_idx = None

        test_blender = (self.hparams.dataset_name == 'blender')
        results = self(rays, ts, whole_img, W, H, rgb_idx, uv_sample, test_blender)
        loss_d, AnnealingWeight = self.loss(results, rgbs, self.hparams, self.global_step)
        loss = sum(l for l in loss_d.values())
        log = {'val_loss': loss}
        for k, v in loss_d.items():
            log[k] = v
        
        #typ = 'fine' if 'rgb_fine' in results else 'coarse'
        img = results['rgb'].view(H, W, 3).permute(2, 0, 1).cpu() # (3, H, W)
        img_gt = rgbs.view(H, W, 3).permute(2, 0, 1).cpu() # (3, H, W)
        if batch_nb == 0:
            depth = visualize_depth(results['depth'].view(H, W)) # (3, H, W)
            if self.hparams.use_mask:
                mask = results['out_mask'].detach().view(H, W, 1).permute(2, 0, 1).repeat(3, 1, 1).cpu() # (3, H, W)
                if 'rgb_fine_random' in results:
                    img_random = results['rgb_fine_random'].detach().view(H, W, 3).permute(2, 0, 1).cpu() # (3, H, W)
                    # stack = torch.stack([img_gt, img, depth, img_random, mask]) # (5, 3, H, W)
                    stack = torch.stack([img_gt, img, mask]) # (3, 3, H, W)
                    self.logger.experiment.add_images('val/GT_pred_depth_random_mask',
                                                      stack, self.global_step)
                else:
                    stack = torch.stack([img_gt, img, depth, mask]) # (4, 3, H, W)
                    self.logger.experiment.add_images('val/GT_pred_depth_mask',
                                                      stack, self.global_step)
            elif 'rgb_fine_random' in results:
                img_random = results[f'rgb_fine_random'].detach().view(H, W, 3).permute(2, 0, 1).cpu() # (3, H, W)
                stack = torch.stack([img_gt, img, depth, img_random]) # (4, 3, H, W)
                self.logger.experiment.add_images('val/GT_pred_depth_random',
                                                  stack, self.global_step)
            else:
                stack = torch.stack([img_gt, img, depth]) # (3, 3, H, W)
                self.logger.experiment.add_images('val/GT_pred_depth',
                                                  stack, self.global_step)
        #注释掉ssim可以Val  slove
        psnr_ = psnr(results['rgb'], rgbs)
        ssim_ = ssim(img[None,...], img_gt[None,...])
        log['val_psnr'] = psnr_
        log['val_ssim'] = ssim_

        return log

    def validation_epoch_end(self, outputs):
        if len(outputs) == 1:
            global_val.current_epoch = self.current_epoch
        else:
            global_val.current_epoch = self.current_epoch + 1
        mean_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        mean_psnr = torch.stack([x['val_psnr'] for x in outputs]).mean()
        mean_ssim = torch.stack([x['val_ssim'] for x in outputs]).mean()
        self.log('val/loss', mean_loss)
        self.log('val/psnr', mean_psnr, prog_bar=True)
        self.log('val/ssim', mean_ssim, prog_bar=True)

        if self.hparams.use_mask:
            #self.log('val/c_l', torch.stack([x['c_l'] for x in outputs]).mean())
            self.log('val/f_l', torch.stack([x['f_l'] for x in outputs]).mean())
            self.log('val/r_ms', torch.stack([x['r_ms'] for x in outputs]).mean())
            self.log('val/r_md', torch.stack([x['r_md'] for x in outputs]).mean())

# python train.py --root_dir ./brandenburg_gate --use_mask --use_cache --img_downscale 2 --maskrs_max 5e-2 --maskrs_min 6e-3 --maskrs_k 1e-3 --maskrd 0
# 
if __name__ == '__main__':
    hparams = get_opts()
    if hparams.val_only and (not hparams.ckpt_path):
        raise ValueError('You need to provide a @ckpt_path for validation!')
    system = NeRFSystem(hparams)

    ckpt_cb = ModelCheckpoint(dirpath=f'ckpts/{hparams.dataset_name}/{hparams.exp_name}',
                              filename='{epoch:d}',
                              save_weights_only=False, # True
                              every_n_epochs=1,
                              save_on_train_epoch_end=True,
                              save_top_k=-1)
                            # every_n_epochs=hparams.num_epochs
                            # monitor='val/psnr',
                            # mode='max',
    callbacks = [ckpt_cb, TQDMProgressBar(refresh_rate=1)]

    logger = TensorBoardLogger(save_dir=f"logs/{hparams.dataset_name}",
                               name=hparams.exp_name,
                               default_hp_metric=False)

    trainer = Trainer(enable_checkpointing=ckpt_cb,
                      max_epochs=hparams.num_epochs,
                      check_val_every_n_epoch=21,
                      callbacks=callbacks,
                      logger=logger,
                      enable_model_summary=False,
                      accelerator='gpu',
                     # devices=hparams.num_gpus,
                      devices= [1], # [1]
                      strategy=DDPPlugin(find_unused_parameters=False)
                               if hparams.num_gpus>1 else None,
                      num_sanity_val_steps=0 ,
                      precision=16,
                      benchmark=True)
                        # num_sanity_val_steps=-1 if hparams.val_only else 0
                        # check_val_every_n_epoch=hparams.num_epochs,

    trainer.fit(system)
    """
    if not hparams.val_only: # save slimmed ckpt for the last epoch
        ckpt_ = \
            slim_ckpt(f'ckpts/{hparams.dataset_name}/{hparams.exp_name}/epoch={hparams.num_epochs-1}.ckpt',
                      save_poses=hparams.optimize_ext)
   a     torch.save(ckpt_, f'ckpts/{hparams.dataset_name}/{hparams.exp_name}/epoch={hparams.num_epochs-1}_slim.ckpt')

    if (not hparams.no_save_test) and \
       hparams.dataset_name=='nsvf' and \
       'Synthetic' in hparams.root_dir: # save video
        imgs = sorted(glob.glob(os.path.join(system.val_dir, '*.png')))
        imageio.mimsave(os.path.join(system.val_dir, 'rgb.mp4'),
                        [imageio.imread(img) for img in imgs[::2]],
                        fps=30, macro_block_size=1)
        imageio.mimsave(os.path.join(system.val_dir, 'depth.mp4'),
                        [imageio.imread(img) for img in imgs[1::2]],
                        fps=30, macro_block_size=1)
    """                    
