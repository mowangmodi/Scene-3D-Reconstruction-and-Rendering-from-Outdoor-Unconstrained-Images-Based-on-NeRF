import argparse
from datasets import PhototourismDataset
import numpy as np
import os
import pickle

def get_opts():
    parser = argparse.ArgumentParser()

    parser.add_argument('--root_dir', type=str, required=True,
                        help='root directory of dataset')
    parser.add_argument('--img_downscale', type=int, default=1,
                        help='how much to downscale the images for phototourism dataset')
    parser.add_argument("--masks_dir", type=str, default="./gate_masks", help="Path to masks directory")

    return parser.parse_args()


if __name__ == '__main__':
    args = get_opts()

    print(f'Preparing cache for scale {args.img_downscale}...')
    cache_dir = os.path.join(args.root_dir, 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    dataset = PhototourismDataset(args.root_dir, 'train', args.img_downscale, masks_dir=args.masks_dir)
    # save img ids
    with open(os.path.join(args.root_dir, f'cache/img_ids.pkl'), 'wb') as f:
        pickle.dump(dataset.img_ids, f, pickle.HIGHEST_PROTOCOL)
    # save img paths
    with open(os.path.join(args.root_dir, f'cache/image_paths.pkl'), 'wb') as f:
        pickle.dump(dataset.image_paths, f, pickle.HIGHEST_PROTOCOL)
    # save Ks
    with open(os.path.join(args.root_dir, f'cache/Ks{args.img_downscale}.pkl'), 'wb') as f:
        pickle.dump(dataset.Ks, f, pickle.HIGHEST_PROTOCOL)

    # save all_imgs
    with open(os.path.join(args.root_dir, f'cache/all_imgs{8}.pkl'), 'wb') as f:
        pickle.dump(dataset.all_imgs, f, pickle.HIGHEST_PROTOCOL)

    # save scene points
    np.save(os.path.join(args.root_dir, 'cache/xyz_world.npy'),
            dataset.xyz_world)
    # save poses
    np.save(os.path.join(args.root_dir, 'cache/poses.npy'),
            dataset.poses)
    # save near and far bounds
    with open(os.path.join(args.root_dir, f'cache/nears.pkl'), 'wb') as f:
        pickle.dump(dataset.nears, f, pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(args.root_dir, f'cache/fars.pkl'), 'wb') as f:
        pickle.dump(dataset.fars, f, pickle.HIGHEST_PROTOCOL)
    # save rays and rgbs
    np.save(os.path.join(args.root_dir, f'cache/rays{args.img_downscale}.npy'),
            dataset.all_rays.numpy())
    np.save(os.path.join(args.root_dir, f'cache/rgbs{args.img_downscale}.npy'),
            dataset.all_rgbs.numpy())

    # save all_imgs_wh
    np.save(os.path.join(args.root_dir, f'cache/all_imgs_wh{args.img_downscale}.npy'),
            dataset.all_imgs_wh.numpy())

    print(f"Data cache saved to {os.path.join(args.root_dir, 'cache')} !")