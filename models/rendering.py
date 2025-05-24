import torch
from .custom_functions import \
    RayAABBIntersector, RayMarcher, VolumeRenderer
from einops import rearrange
import vren

MAX_SAMPLES = 1024 # 1024

NEAR_DISTANCE = 0.01 # 0.01  -1.0
#FAR_DISTANCE = 64 #　４.00


@torch.cuda.amp.autocast()
def render(model,
            rays,
            ts,
            use_disp=False,
            chunk=1024*32,
            white_back=False,               
            **kwargs
                ):
    """
    Render rays by
    1. Compute the intersection of the rays with the scene bounding box
    2. Follow the process in @render_func (different for train/test)

    Inputs:
        model: NGP
        rays_o: (N_rays, 3) ray origins
        rays_d: (N_rays, 3) ray directions

    Outputs:
        result: dictionary containing final rgb and depth
    """
    
    rays_o,rays_d = rays[:, 0:3],rays[:, 3:6]
    rays_o = rays_o.contiguous()
    rays_d = rays_d.contiguous()
    
    ###
    near, far = rays[:,6],rays[:,7]
    far += 5.0
    hits_t = torch.stack((near,far),dim=1)
    hits_t = torch.unsqueeze(hits_t,dim=1)
    
    #hits_t[(hits_t[:, 0, 1]<FAR_DISTANCE), 0, 1] = FAR_DISTANCE
    #hits_t[(hits_t[:, 0, 0]<NEAR_DISTANCE), 0, 0] = NEAR_DISTANCE
    
    hits_t[(hits_t[:, 0, 0]>=0)&(hits_t[:, 0, 0]<NEAR_DISTANCE), 0, 0] = NEAR_DISTANCE 
    
    # _, hits_t, _ = \
    #     RayAABBIntersector.apply(rays_o, rays_d, model.center, model.half_size, 1)
    # hits_t[(hits_t[:, 0, 0]>=0)&(hits_t[:, 0, 0]<NEAR_DISTANCE), 0, 0] = NEAR_DISTANCE # (1024,1,2)
    
    
    if kwargs.get('test_time', False):
        render_func = __render_rays_test
    else:
        render_func = __render_rays_train

    results = render_func(model, rays_o, rays_d, hits_t, **kwargs)
    for k, v in results.items():
        if kwargs.get('to_cpu', False): #False
            v = v.cpu()
            if kwargs.get('to_numpy', False): # False
                v = v.numpy()
        results[k] = v
    return results


@torch.no_grad()
def __render_rays_test(model, rays_o, rays_d, hits_t, **kwargs):
    """
    Render rays by

    while (a ray hasn't converged)
        1. Move each ray to its next occupied @N_samples (initially 1) samples 
           and evaluate the properties (sigmas, rgbs) there
        2. Composite the result to output; if a ray has transmittance lower
           than a threshold, mark this ray as converged and stop marching it.
           When more rays are dead, we can increase the number of samples
           of each marching (the variable @N_samples)
    """
    exp_step_factor = kwargs.get('exp_step_factor', 1/256)# 0  1/256
    results = {}

    # output tensors to be filled in
    N_rays = len(rays_o)
    device = rays_o.device
    opacity = torch.zeros(N_rays, device=device)
    depth = torch.zeros(N_rays, device=device)
    rgb = torch.zeros(N_rays, 3, device=device)

    samples = total_samples = 0
    alive_indices = torch.arange(N_rays, device=device)
    # if it's synthetic data, bg is majority so min_samples=1 effectively covers the bg
    # otherwise, 4 is more efficient empirically
    min_samples = 1 if exp_step_factor==0 else 4

    while samples < kwargs.get('max_samples', MAX_SAMPLES):
        N_alive = len(alive_indices)
        if N_alive==0: break

        # the number of samples to add on each ray
        N_samples = max(min(N_rays//N_alive, 64), min_samples)
        samples += N_samples

        xyzs, dirs, deltas, ts, N_eff_samples = \
            vren.raymarching_test(rays_o, rays_d, hits_t[:, 0], alive_indices,
                                  model.density_bitfield, model.cascades,
                                  model.scale, exp_step_factor,
                                  model.grid_size, MAX_SAMPLES, N_samples)
        total_samples += N_eff_samples.sum()
        xyzs = rearrange(xyzs, 'n1 n2 c -> (n1 n2) c')
        dirs = rearrange(dirs, 'n1 n2 c -> (n1 n2) c')
        valid_mask = ~torch.all(dirs==0, dim=1)
        if valid_mask.sum()==0: break

        sigmas = torch.zeros(len(xyzs), device=device)
        rgbs = torch.zeros(len(xyzs), 3, device=device)
        sigmas[valid_mask], _rgbs = model(xyzs[valid_mask], dirs[valid_mask], **kwargs)
        rgbs[valid_mask] = _rgbs.float()
        sigmas = rearrange(sigmas, '(n1 n2) -> n1 n2', n2=N_samples)
        rgbs = rearrange(rgbs, '(n1 n2) c -> n1 n2 c', n2=N_samples)

        vren.composite_test_fw(
            sigmas, rgbs, deltas, ts,
            hits_t[:, 0], alive_indices, kwargs.get('T_threshold', 1e-4),
            N_eff_samples, opacity, depth, rgb)
        alive_indices = alive_indices[alive_indices>=0] # remove converged rays

    results['opacity'] = opacity
    results['depth'] = depth
    results['rgb'] = rgb
    results['total_samples'] = total_samples # total samples for all rays
    """
    if exp_step_factor==0: # synthetic
        rgb_bg = torch.ones(3, device=device)
    else: # real
        rgb_bg = torch.zeros(3, device=device)
    results['rgb'] += rgb_bg*rearrange(1-opacity, 'n -> n 1')
    """
    #return results
    ############last_results 
    last_results = {}
    if 'rgb' or 'depth'in results:
        last_results['rgb'] = results['rgb']
        last_results['depth'] = results['depth']
    
    return last_results

    


def __render_rays_train(model, rays_o, rays_d, hits_t, **kwargs):
    """
    Render rays by
    1. March the rays along their directions, querying @density_bitfield
       to skip empty space, and get the effective sample points (where
       there is object)
    2. Infer the NN at these positions and view directions to get properties
       (currently sigmas and rgbs)
    3. Use volume rendering to combine the result (front to back compositing
       and early stop the ray if its transmittance is below a threshold)
    """
    exp_step_factor = kwargs.get('exp_step_factor',1/256) # 0
    results = {}

    (rays_a, xyzs, dirs,
    results['deltas'], results['ts'], results['rm_samples']) = \
        RayMarcher.apply(
            rays_o, rays_d, hits_t[:, 0], model.density_bitfield,
            model.cascades, model.scale,
            exp_step_factor, model.grid_size, MAX_SAMPLES)
    
    ###添加额外输入 
    
    for k, v in kwargs.items():
        if isinstance(v, torch.Tensor):
            kwargs[k] = kwargs[k].repeat(dirs.size(0),1)  # (N, 48)

    
    """
    for k, v in kwargs.items(): # supply additional inputs, repeated per ray
        if isinstance(v, torch.Tensor):
            kwargs[k] = torch.repeat_interleave(v[rays_a[:, 0]], rays_a[:, 2], 0)

            apperance = v.repeat(dirs.size(0),1)
            kwargs[k] = torch.cat((dirs,apperance),dim=1)  #apperance--shape(1,48)
    """
    sigmas, rgbs, rgbs_random = model(xyzs, dirs, **kwargs)

    (results['vr_samples'], results['opacity'],
    results['depth'], results['rgb'], results['ws']) = \
        VolumeRenderer.apply(sigmas, rgbs.contiguous(), results['deltas'], results['ts'],
                             rays_a, kwargs.get('T_threshold', 1e-4))
    #results['rays_a'] = rays_a

    ####
    if 'a_embedded_random' in kwargs:
        (results['vr_samples'], results['opacity'],
    results['depth1'], results['rgb_random'], results['ws'])= \
        VolumeRenderer.apply(sigmas, rgbs_random.contiguous(), results['deltas'], results['ts'],
                             rays_a, kwargs.get('T_threshold', 1e-4))
    
    """
    if exp_step_factor==0: # synthetic
        rgb_bg = torch.ones(3, device=rays_o.device)
    else: # real
        if kwargs.get('random_bg', False):
            rgb_bg = torch.rand(3, device=rays_o.device)
        else:
            rgb_bg = torch.zeros(3, device=rays_o.device2
    results['rgb'] = results['rgb'] + \
                     rgb_bg*rearrange(1-results['opacity'], 'n -> n 1')
    """
    #return results
    ###  保证可以torch.cat(V,0),取出有用数据
    last_results = {}
    if 'rgb' or 'depth'or 'rgb_random'in results:
        last_results['rgb'] = results['rgb']
        last_results['depth'] = results['depth']
        last_results['rgb_random'] = results['rgb_random'] #  test delete
    
    return last_results

