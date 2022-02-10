import torch
from torchvision import transforms, utils
from util import *
from PIL import Image
import math
import random
import os
import argparse

import numpy as np
from torch import nn, autograd, optim
from torch.nn import functional as F
from tqdm import tqdm
from model import *
from e4e_projection import projection as e4e_projection

from copy import deepcopy

def parse_args():
    parser = argparse.ArgumentParser() 
    parser.add_argument("--device", default="cuda", help="device")
    parser.add_argument("--input", default="iu.jpeg", help="input image")
    parser.add_argument("--model_name", default="custom", help="model")
    parser.add_argument("--n_sample", default="5", help="n_sample")
    parser.add_argument("--seed", default="3000", help="seed")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    device = args.device
    filename = args.input
    filepath = f'test_input/{filename}'
    name = strip_path_extension(filepath)+'.pt'
    aligned_face = align_face(filepath)
    my_w = e4e_projection(aligned_face, name, device).unsqueeze(0)
    latent_dim = 512
    transform = transforms.Compose(
    [
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # reset generator
    original_generator = Generator(1024, latent_dim, 8, 2).to(device)
    ckpt = torch.load('models/stylegan2-ffhq-config-f.pt', map_location=lambda storage, loc: storage)
    original_generator.load_state_dict(ckpt["g_ema"], strict=False)
    generator = deepcopy(original_generator)


    #original_generator = Generator(1024, latent_dim, 8, 2).to(device)
    ckpt = torch.load(os.path.join('models', args.model_name+'.pt'), map_location=lambda storage, loc: storage)
    generator.load_state_dict(ckpt["g"], strict=False)


    # Generate results
    torch.manual_seed(int(args.seed))
    with torch.no_grad():
        generator.eval()

        my_sample = generator(my_w, input_is_latent=True)
        face = transform(aligned_face).unsqueeze(0).to(device)
        my_output = torch.cat([face, my_sample], 0)
        print(strip_path_extension(filepath))
        clear_name = "".join(strip_path_extension(filepath).split('/')[1])
        torchvision.utils.save_image(utils.make_grid(my_output, normalize=True, range=(-1, 1)), "results/result_"+args.model_name+"_"+clear_name+".png")