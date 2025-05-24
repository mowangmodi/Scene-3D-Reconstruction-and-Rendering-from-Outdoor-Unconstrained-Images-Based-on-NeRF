import torch
from torch import nn
import tinycudann as tcnn
import vren
from einops import rearrange
from .custom_functions import TruncExp
import numpy as np

from .rendering import NEAR_DISTANCE


class NGP(nn.Module):
    def __init__(self, scale, rgb_act='Sigmoid'):
        super().__init__()

        self.rgb_act = rgb_act

        # scene bounding box
        self.scale = scale
        self.register_buffer('center', torch.zeros(1, 3))
        self.register_buffer('xyz_min', -torch.ones(1, 3)*scale)
        self.register_buffer('xyz_max', torch.ones(1, 3)*scale)
        self.register_buffer('half_size', (self.xyz_max-self.xyz_min)/2)

        # each density grid covers [-2^(k-1), 2^(k-1)]^3 for k in [0, C-1]
        self.cascades = max(1+int(np.ceil(np.log2(2*scale))), 1)
        self.grid_size = 128
        self.register_buffer('density_bitfield',
            torch.zeros(self.cascades*self.grid_size**3//8, dtype=torch.uint8))

        # constants
        L = 16; F = 4; log2_T = 19; N_min = 16
        b = np.exp(np.log(2048*scale/N_min)/(L-1)) # b = np.exp(np.log(2048*scale/N_min)/(L-1))
        print(f'GridEncoding: Nmin={N_min} b={b:.5f} F={F} T=2^{log2_T} L={L}')

        self.xyz_encoder = \
            tcnn.NetworkWithInputEncoding(
                n_input_dims=3, n_output_dims=16,
                encoding_config={
                    "otype": "Grid",
	                "type": "Hash",
                    "n_levels": L,
                    "n_features_per_level": F,
                    "log2_hashmap_size": log2_T,
                    "base_resolution": N_min,
                    "per_level_scale": b,
                    "interpolation": "Linear"
                },
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": 128,# 64
                    "n_hidden_layers": 1, # 1
                }
            )
        """
        # PE
        self.dir_encoder = \
            tcnn.Encoding(
                n_input_dims=3,#3
                encoding_config={
                    "otype": "Frequency",
                    "n_frequencies": 6,
                },
            )
        """
        self.dir_encoder = \
            tcnn.Encoding(
                n_input_dims=3, # 3
                encoding_config={
                    "otype": "SphericalHarmonics",
                    "degree": 4,
                },
            )
        

        self.rgb_net = \
            tcnn.Network(
                n_input_dims=80, n_output_dims=3, # n_input_dims=32  sh-->80    pe(f=4)-->88   96   degree=4-->16, degree=5->25  n**2
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": self.rgb_act,
                    "n_neurons": 64, # 64
                    "n_hidden_layers": 2,# 2
                }
            )

        if self.rgb_act == 'None': # rgb_net output is log-radiance
            for i in range(3): # independent tonemappers for r,g,b
                tonemapper_net = \
                    tcnn.Network(
                        n_input_dims=1, n_output_dims=1,
                        network_config={
                            "otype": "FullyFusedMLP",
                            "activation": "ReLU",
                            "output_activation": "Sigmoid",
                            "n_neurons": 64,
                            "n_hidden_layers": 1,
                        }
                    )
                setattr(self, f'tonemapper_net_{i}', tonemapper_net)

    def density(self, x, return_feat=False):
        """
        Inputs:
            x: (N, 3) xyz in [-scale, scale]
            return_feat: whether to return intermediate feature

        Outputs:
            sigmas: (N)
        """
        x = (x-self.xyz_min)/(self.xyz_max-self.xyz_min)
        h = self.xyz_encoder(x)
        sigmas = TruncExp.apply(h[:, 0])
        if return_feat: return sigmas, h
        return sigmas

    def log_radiance_to_rgb(self, log_radiances, **kwargs):
        """
        Convert log-radiance to rgb as the setting in HDR-NeRF.
        Called only when self.rgb_act == 'None' (with exposure)

        Inputs:
            log_radiances: (N, 3)

        Outputs:
            rgbs: (N, 3)
        """
        if 'exposure' in kwargs:
            log_exposure = torch.log(kwargs['exposure'])
        else: # unit exposure by default
            log_exposure = 0

        out = []
        for i in range(3):
            inp = log_radiances[:, i:i+1]+log_exposure
            out += [getattr(self, f'tonemapper_net_{i}')(inp)]
        rgbs = torch.cat(out, 1)
        return rgbs

    def forward(self, x, d, **kwargs):
        """
        Inputs:
            x: (N, 3) xyz in [-scale, scale]
            d: (N, 3) directions

        Outputs:
            sigmas: (N)
            rgbs: (N, 3)
        """
        sigmas, h = self.density(x, return_feat=True)
        d = d/torch.norm(d, dim=1, keepdim=True)
        d = self.dir_encoder((d+1)/2)

        ### fix dir
        dir = torch.cat([d,kwargs['a_embedded_from_img']],dim=-1)
        #rgbs = self.rgb_net(torch.cat([dir, h], 1))
        rgbs = self.rgb_net(torch.cat([h, dir], 1))
        
        if 'a_embedded_random' or 'a_embedded_from_img' in kwargs:
            #dir_random = torch.cat([d.detach(),kwargs['a_embedded_random']],dim=-1) ## detach() version 3
            #dir_random = torch.cat([d,kwargs['a_embedded_random']],dim=-1) 
            dir_random = torch.cat([d,kwargs['a_embedded_random']],dim=-1) 
            rgbs_random = self.rgb_net(torch.cat([h, dir_random], 1))

    
        """
        rgbs = self.rgb_net1(torch.cat([d, h], 1))
        ### 添加额外输入 
        if 'a_embedded_random' in kwargs:  # random
            d = torch.cat([d,kwargs['a_embedded_random']],dim=-1)
        else:
            d = torch.cat([d,kwargs['a_embedded_from_img']],dim=-1)
        rgbs_random = self.rgb_net(torch.cat([d, h], 1))


        
        ### 添加额外输入 
        if 'a_embedded_random' in kwargs:  # random
            d = torch.cat([d,kwargs['a_embedded_random']],dim=1)
        else:
            d = torch.cat([d,kwargs['a_embedded_from_img']],dim=1)
        d = self.dir_encoder((d+1)/2)
        rgbs = self.rgb_net(torch.cat([d, h], 1))
        
               
        if self.rgb_act == 'None': # rgbs is log-radiance
            if kwargs.get('output_radiance', False): # output HDR map
                rgbs = TruncExp.apply(rgbs)
            else: # convert to LDR using tonemapper networks
                rgbs = self.log_radiance_to_rgb(rgbs, **kwargs)
        """ 
        return sigmas, rgbs, rgbs_random

    @torch.no_grad()
    def get_all_cells(self):
        """
        Get all cells from the density grid.
        
        Outputs:
            cells: list (of length self.cascades) of indices and coords
                   selected at each cascade
        """
        indices = vren.morton3D(self.grid_coords).long()
        cells = [(indices, self.grid_coords)] * self.cascades

        return cells

    @torch.no_grad()
    def sample_uniform_and_occupied_cells(self, M, density_threshold):
        """
        Sample both M uniform and occupied cells (per cascade)
        occupied cells are sample from cells with density > @density_threshold
        
        Outputs:
            cells: list (of length self.cascades) of indices and coords
                   selected at each cascade
        """
        cells = []
        for c in range(self.cascades):
            # uniform cells
            coords1 = torch.randint(self.grid_size, (M, 3), dtype=torch.int32,
                                    device=self.density_grid.device)
            indices1 = vren.morton3D(coords1).long()
            # occupied cells
            indices2 = torch.nonzero(self.density_grid[c]>density_threshold)[:, 0]
            if len(indices2)>0:
                rand_idx = torch.randint(len(indices2), (M,),
                                         device=self.density_grid.device)
                indices2 = indices2[rand_idx]
            coords2 = vren.morton3D_invert(indices2.int())
            # concatenate
            cells += [(torch.cat([indices1, indices2]), torch.cat([coords1, coords2]))]

        return cells

    @torch.no_grad()
    def mark_invisible_cells(self, K, poses, img_wh, chunk=64**3):
        """
        mark the cells that aren't covered by the cameras with density -1
        only executed once before training starts

        Inputs:
            K: (3, 3) camera intrinsics
            poses: (N, 3, 4) camera to world poses
            img_wh: image width and height
            chunk: the chunk size to split the cells (to avoid OOM)
        """
        N_cams = poses.shape[0]
        self.count_grid = torch.zeros_like(self.density_grid)
        w2c_R = rearrange(poses[:, :3, :3], 'n a b -> n b a') # (N_cams, 3, 3)
        w2c_T = -w2c_R@poses[:, :3, 3:] # (N_cams, 3, 1)
        cells = self.get_all_cells()
        for c in range(self.cascades):
            indices, coords = cells[c]
            for i in range(0, len(indices), chunk):
                xyzs = coords[i:i+chunk]/(self.grid_size-1)*2-1
                s = min(2**(c-1), self.scale)
                half_grid_size = s/self.grid_size
                xyzs_w = (xyzs*(s-half_grid_size)).T # (3, chunk)
                xyzs_c = w2c_R @ xyzs_w + w2c_T # (N_cams, 3, chunk)
                uvd = K @ xyzs_c # (N_cams, 3, chunk)
                uv = uvd[:, :2]/uvd[:, 2:] # (N_cams, 2, chunk)
                in_image = (uvd[:, 2]>=0)& \
                           (uv[:, 0]>=0)&(uv[:, 0]<img_wh[0])& \
                           (uv[:, 1]>=0)&(uv[:, 1]<img_wh[1])
                covered_by_cam = (uvd[:, 2]>=NEAR_DISTANCE)&in_image # (N_cams, chunk)
                # if the cell is visible by at least one camera
                self.count_grid[c, indices[i:i+chunk]] = \
                    count = covered_by_cam.sum(0)/N_cams

                too_near_to_cam = (uvd[:, 2]<NEAR_DISTANCE)&in_image # (N, chunk)
                # if the cell is too close (in front) to any camera
                too_near_to_any_cam = too_near_to_cam.any(0)
                # a valid cell should be visible by at least one camera and not too close to any camera
                valid_mask = (count>0)&(~too_near_to_any_cam)
                self.density_grid[c, indices[i:i+chunk]] = \
                    torch.where(valid_mask, 0., -1.)

    @torch.no_grad()
    def update_density_grid(self, density_threshold, warmup=False, decay=0.95, erode=False):
        ###
        self.count_grid = torch.zeros_like(self.density_grid)
        ###
        density_grid_tmp = torch.zeros_like(self.density_grid)
        if warmup: # during the first steps
            cells = self.get_all_cells()
        else:
            cells = self.sample_uniform_and_occupied_cells(self.grid_size**3//4,
                                                           density_threshold)
        # infer sigmas
        for c in range(self.cascades):
            indices, coords = cells[c]
            s = min(2**(c-1), self.scale)
            half_grid_size = s/self.grid_size
            xyzs_w = (coords/(self.grid_size-1)*2-1)*(s-half_grid_size)
            # pick random position in the cell by adding noise in [-hgs, hgs]
            xyzs_w += (torch.rand_like(xyzs_w)*2-1) * half_grid_size
            density_grid_tmp[c, indices] = self.density(xyzs_w)

        if erode:
            # My own logic. decay more the cells that are visible to few cameras
            decay = torch.clamp(decay**(1/self.count_grid), 0.1, 0.95)
        self.density_grid = \
            torch.where(self.density_grid<0,
                        self.density_grid,
                        torch.maximum(self.density_grid*decay, density_grid_tmp))

        mean_density = self.density_grid[self.density_grid>0].mean().item()

        vren.packbits(self.density_grid, min(mean_density, density_threshold),
                      self.density_bitfield)

###hanerf
#my cnn

class E_attr(nn.Module):
  def __init__(self, input_dim_a, output_nc=8):
    super(E_attr, self).__init__()
    dim = 64
    
    ### ours cnn
    self.model = nn.Sequential(
        nn.ReflectionPad2d(2),
        nn.Conv2d(input_dim_a, 128, 3, 1), # nn.Conv2d(input_dim_a, 128, 3, 1),
        nn.ReLU(inplace=True),  ## size
        nn.ReflectionPad2d(2),
        nn.Conv2d(128, 256, 4, 1),  # nn.Conv2d(128, 256, 4, 1),
        nn.ReLU(inplace=True),  ## size/2
        nn.ReflectionPad2d(2),
        nn.Conv2d(256, 256, 4, 2),  # s = 2
        nn.ReLU(inplace=True),  ## size/4
        nn.ReflectionPad2d(2),
        nn.Conv2d(dim * 4, dim * 4, 3, 2),  # s = 2
        nn.ReLU(inplace=True),  ## size/8
        nn.ReflectionPad2d(1),
        nn.Conv2d(dim * 4, dim * 4, 3, 1),  # s = 2   nn.Conv2d(dim*4, dim*2, 3, 1),
        nn.ReLU(inplace=True),  ## size/16
        nn.AdaptiveAvgPool2d(1),
        nn.Conv2d(dim * 4, 48, 1, 1, 0))
    return
    
    
    
    ### HaNeRF cnn
    """
    self.model = nn.Sequential(
        nn.ReflectionPad2d(3),
        nn.Conv2d(3, dim, 7, 1),
        nn.ReLU(inplace=True),  ## size
        nn.ReflectionPad2d(1),
        nn.Conv2d(dim, dim*2, 4, 2),
        nn.ReLU(inplace=True),  ## size/2
        nn.ReflectionPad2d(1),
        nn.Conv2d(dim*2, dim*4, 4, 2),
        nn.ReLU(inplace=True),  ## size/4
        nn.ReflectionPad2d(1),
        nn.Conv2d(dim*4, dim*4, 4, 2),
        nn.ReLU(inplace=True),  ## size/8
        nn.ReflectionPad2d(1),
        nn.Conv2d(dim*4, dim*4, 4, 2),
        nn.ReLU(inplace=True),  ## size/16
        nn.AdaptiveAvgPool2d(1),
        nn.Conv2d(dim*4, 64, 1, 1, 0))  ## 1*1
    """
    
    """  
 #### raw hanerf cnn
class E_attr(nn.Module):
  def __init__(self, input_dim_a, output_nc=8): # 3,48
    super(E_attr, self).__init__()
    dim = 64
    self.model = nn.Sequential(
        nn.ReflectionPad2d(3),
        nn.Conv2d(input_dim_a, dim, 7, 1), # input_dim_a = 3
        nn.ReLU(inplace=True),  ## size
        nn.ReflectionPad2d(1),
        nn.Conv2d(dim, dim*2, 4, 2),
        nn.ReLU(inplace=True),  ## size/2
        nn.ReflectionPad2d(1),
        nn.Conv2d(dim*2, dim*4, 4, 2),
        nn.ReLU(inplace=True),  ## size/4
        nn.ReflectionPad2d(1),
        nn.Conv2d(dim*4, dim*4, 4, 2),
        nn.ReLU(inplace=True),  ## size/8
        nn.ReflectionPad2d(1),
        nn.Conv2d(dim*4, dim*4, 4, 2),
        nn.ReLU(inplace=True),  ## size/16
        nn.AdaptiveAvgPool2d(1),
        nn.Conv2d(dim*4, output_nc, 1, 1, 0))  ## 1*1
    return
    """  

  def forward(self, x):
    x = self.model(x)
    output = x.view(x.size(0), -1)   #val error (48,1) but train (1,48)
    #output = x.view(1,48)
    return output
  


class implicit_mask(nn.Module):
    def __init__(self, latent=128, W=256, in_channels_dir=42): # 42  62   cnn-->64 latent
        super().__init__()
        self.mask_mapping = nn.Sequential(
                            nn.Linear(latent + in_channels_dir, W), nn.ReLU(True),
                            nn.Linear(W, W), nn.ReLU(True),
                            nn.Linear(W, W), nn.ReLU(True),
                            nn.Linear(W, W), nn.ReLU(True),
                            nn.Linear(W, 1), nn.Sigmoid())

    def forward(self, x):
        mask = self.mask_mapping(x)
        return mask
