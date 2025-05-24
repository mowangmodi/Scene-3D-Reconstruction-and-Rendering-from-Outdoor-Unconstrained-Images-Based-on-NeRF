import argparse

def get_opts():
    parser = argparse.ArgumentParser()

    # dataset parameters
    parser.add_argument('--root_dir', type=str, required=True,
                        help='root directory of dataset')

    parser.add_argument('--split', type=str, default='train',
                        choices=['train', 'trainval', 'trainvaltest'],
                        help='use which split to train')
    parser.add_argument('--downsample', type=float, default=1.0,
                        help='downsample factor (<=1.0) for the images')

    # model parameters
    parser.add_argument('--scale', type=float, default=4,
                        help='scene scale (whole scene must lie in [-scale, scale]^3')
    
    parser.add_argument('--use_exposure', action='store_true', default=False,
                        help='whether to train in HDR-NeRF setting')

    # loss parameters
    parser.add_argument('--distortion_loss_w', type=float, default=0,
                        help='''weight of distortion loss (see losses.py),
                        0 to disable (default), to enable,
                        a good value is 1e-3 for real scene and 1e-2 for synthetic scene
                        ''')

    # training options
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='number of rays in a batch')
    parser.add_argument('--ray_sampling_strategy', type=str, default='all_images',
                        choices=['all_images', 'same_image'],
                        help='''
                        all_images: uniformly from all pixels of ALL images
                        same_image: uniformly from all pixels of a SAME image
                        ''')
    
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='number of gpus')
    
    # experimental training options
    parser.add_argument('--optimize_ext', action='store_true', default=False,
                        help='whether to optimize extrinsics')
    parser.add_argument('--random_bg', action='store_true', default=False,
                        help='''whether to train with random bg color (real scene only)
                        to avoid objects with black color to be predicted as transparent
                        ''')

    # validation options
    parser.add_argument('--eval_lpips', action='store_true', default=False,
                        help='evaluate lpips metric (consumes more VRAM)')
    parser.add_argument('--val_only', action='store_true', default=False,
                        help='run only validation (need to provide ckpt_path)')
    parser.add_argument('--no_save_test', action='store_true', default=False,
                        help='whether to save test image and video')

    # misc
    parser.add_argument('--exp_name', type=str, default='gate',
                        help='experiment name')
    parser.add_argument('--ckpt_path', type=str, default=None,
                        help='pretrained checkpoint to load (including optimizers, etc)')
    parser.add_argument('--weight_path', type=str, default=None,
                        help='pretrained checkpoint to load (excluding optimizers, etc)')
    
    ### HaNeRF
    parser.add_argument('--dataset_name', type=str, default='phototourism',
                        choices=['nerf', 'nsvf', 'colmap', 'nerfpp', 'rtmv', 'phototourism'],
                        help='which dataset to train/test')
    parser.add_argument('--N_vocab', type=int, default=5000,
                        help='''number of vocabulary (number of images) 
                                in the dataset for nn.Embedding''')
    parser.add_argument('--encode_a', default=False, action="store_true",
                        help='whether to encode appearance')
    parser.add_argument('--encode_random', default=False, action="store_true",
                        help='whether to encode_random')
    parser.add_argument('--N_a', type=int, default=48, #48
                        help='number of embeddings for appearance')
    parser.add_argument('--use_mask', default=False, action="store_true",
                        help='whether to use mask')
    
    parser.add_argument('--img_downscale', type=int, default=1,
                        help='how much to downscale the images for phototourism dataset')
    parser.add_argument('--use_cache', default=False, action="store_true",
                        help='whether to use ray cache (make sure img_downscale is the same)')
    parser.add_argument('--scale_anneal', type=float, default=-1,
                        help='scale_anneal')
    parser.add_argument('--min_scale', type=float, default=0.5,
                        help='min_scale')
    #train optiong
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay')
    parser.add_argument('--optimizer', type=str, default='adam',
                        help='optimizer type',
                        choices=['sgd', 'adam', 'radam', 'ranger','FusedAdam'])
    parser.add_argument('--lr_scheduler', type=str, default='cosine',
                        help='scheduler type',
                        choices=['steplr', 'cosine', 'poly'])
    parser.add_argument('--num_epochs', type=int, default=20,
                        help='number of training epochs')
    parser.add_argument('--decay_step', nargs='+', type=int, default=[1,2,3,4,5,6,7,8,9,10],
                        help='scheduler decay step')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='learn rate momentum')                    
    parser.add_argument('--decay_gamma', type=float, default=0.5,
                        help='learning rate decay amount')
    parser.add_argument('--warmup_multiplier', type=float, default=1.0,
                        help='lr is multiplied by this factor after --warmup_epochs')
    parser.add_argument('--warmup_epochs', type=int, default=0,
                        help='Gradually warm-up(increasing) learning rate in optimizer')
    ###render
    parser.add_argument('--chunk', type=int, default=16*1024, 
                        help='chunk size to split the input to avoid OOM')
    parser.add_argument('--use_disp', default=False, action="store_true",
                        help='use disparity depth sampling')
    
    ###loss
    parser.add_argument('--maskrd', type=float, default=0.001,
                        help='regularize mask digit')
    parser.add_argument('--maskrs_max', type=float, default=0.0000015,
                        help='regularize mask size')
    parser.add_argument('--maskrs_min', type=float, default=0.0000015,
                        help='regularize mask size')
    parser.add_argument('--maskrs_k', type=float, default=0.9,
                        help='regularize mask size')
    parser.add_argument('--weightKL', type=float, default=1e-5,
                        help='regularize encA')
    parser.add_argument('--weightRecA', type=float, default=1e-3,
                        help='Rec A')
    parser.add_argument('--weightMS', type=float, default=1e-6,
                        help='mode seeking')
    
    ###masks 
    parser.add_argument('--masks_dir', type=str, default="./gate_masks",
                        help='root directory of masks')

    return parser.parse_args()
