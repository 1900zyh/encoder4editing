import os
from argparse import Namespace
import time
import os
import sys
import numpy as np
from PIL import Image
import torch
from glob import glob 
from tqdm import tqdm 
import torchvision.transforms as transforms
import torch.multiprocessing as mp 

sys.path.append(".")
sys.path.append("..")

from utils.common import tensor2im
from models.psp import pSp  # we use the pSp framework to load the e4e encoder.




model_path = "/home/t-yazen/pretrained/e4e_ffhq_encode.pt"
root_path = "/home/t-yazen/datasets/ravdess_vllp"
frame_dir = "ravdess_frames"
latent_dir = "ravdess_latents_e4e"
inverted_dir = "ravdess_frames_e4e"


def run_on_batch(inputs, net):
    with torch.no_grad(): 
        images, latents = net(
            inputs.cuda().float(), resize=False, randomize_noise=False, return_latents=True)
    return images, latents


def worker(gpu_id): 
    torch.cuda.set_device(gpu_id)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    resize_dims = (256, 256)

    ckpt = torch.load(model_path, map_location='cuda')
    opts = ckpt['opts']
    opts['checkpoint_path'] = model_path
    opts= Namespace(**opts)
    net = pSp(opts)
    net.eval()
    net.cuda()
    
    filelist = glob(os.path.join(root_path, frame_dir, '*/*.png'))
    filelist.sort()
    for i in tqdm(range(gpu_id, len(filelist), 4)):
        f = filelist[i]
        image = Image.open(f).convert("RGB")
        image.resize(resize_dims)
        image = transform(image)
        images, latents = run_on_batch(image.unsqueeze(0), net)
        result_image, latent = images[0], latents[0]
        latent_path = f.replace(frame_dir, latent_dir).replace('.png', '.pt')
        os.makedirs(os.path.dirname(latent_path), exist_ok=True)
        torch.save(latent, latent_path)

        inverted_path = f.replace(frame_dir, inverted_dir)
        os.makedirs(os.path.dirname(inverted_path), exist_ok=True)
        result_image = (result_image.permute(1,2,0) + 1.0)/2.0 * 255.0 
        result_image = result_image.detach().cpu()
        result_image = torch.clamp(result_image, 0, 255)
        result_image = result_image.numpy().astype(np.uint8)
        Image.fromarray(result_image).save(inverted_path)
        

    
if __name__ == '__main__': 
    mp.spawn(worker, nprocs=4)
    
    
    