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
#import wandb
from model import *
from e4e_projection import projection as e4e_projection

from copy import deepcopy

def parse_args():
    parser = argparse.ArgumentParser() 
    parser.add_argument("--alpha", default="0.0", help="alpha value")
    parser.add_argument("--preserve_color", default="False", help="preserve_color")
    parser.add_argument("--num_iter", default="300", help="num_iter")
    parser.add_argument("--device", default="cuda", help="device")
    parser.add_argument("--model_name", default="custom", help="model name")
    parser.add_argument("--force_name", default="",help="image you forcing to train")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    device = args.device
    latent_dim = 512
    targets = []
    latents = []
    transform = transforms.Compose(
    [
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    styles = os.listdir("style_images")
    for name in styles:
        style_path = os.path.join('style_images', name)
        assert os.path.exists(style_path), f"{style_path} does not exist!"

        name = strip_path_extension(name)

        # # crop and align the face
        # style_aligned_path = os.path.join('style_images_aligned', f'{name}.png')
        # if not os.path.exists(style_aligned_path):
        #     style_aligned = align_face(style_path)
        #     style_aligned.save(style_aligned_path)
        # else:
        #     style_aligned = Image.open(style_aligned_path).convert('RGB')

        # # GAN invert
        # style_code_path = os.path.join('inversion_codes', f'{name}.pt')
        # if not os.path.exists(style_code_path):
        #     latent = e4e_projection(style_aligned, style_code_path, device)
        # else:
        #     latent = torch.load(style_code_path)['latent']

        # targets.append(transform(style_aligned).to(device))
        # latents.append(latent.to(device))

    # --------------------- manual codes
    style_aligned = Image.open(f"style_images_aligned/{args.force_name}.png").convert('RGB')
    style_code_path = os.path.join('inversion_codes', f'{args.force_name}.pt')
    latent = e4e_projection(style_aligned, style_code_path, device)
    targets.append(transform(style_aligned).to(device))
    latents.append(latent.to(device))
    # --------------------- manual codes

    targets = torch.stack(targets, 0)
    latents = torch.stack(latents, 0)


    # load discriminator for perceptual loss
    discriminator = Discriminator(1024, 2).eval().to(device)
    ckpt = torch.load('models/stylegan2-ffhq-config-f.pt', map_location=lambda storage, loc: storage)
    discriminator.load_state_dict(ckpt["d"], strict=False)

    # reset generator
    generator = Generator(1024, latent_dim, 8, 2).to(device)
    ckpt = torch.load('models/stylegan2-ffhq-config-f.pt', map_location=lambda storage, loc: storage)
    generator.load_state_dict(ckpt["g_ema"], strict=False)
    mean_latent = generator.mean_latent(10000)

    #generator = deepcopy(original_generator)

    g_optim = optim.Adam(generator.parameters(), lr=2e-3, betas=(0, 0.99))

    # Which layers to swap for generating a family of plausible real images -> fake image
    if "True" == args.preserve_color:
        id_swap = [9,11,15,16,17]
    else:
        id_swap = list(range(7, generator.n_latent))

    for idx in tqdm(range(int(args.num_iter))):
        mean_w = generator.get_latent(torch.randn([latents.size(0), latent_dim]).to(device)).unsqueeze(1).repeat(1, generator.n_latent, 1)
        in_latent = latents.clone()
        in_latent[:, id_swap] = float(args.alpha)*latents[:, id_swap] + (1-float(args.alpha))*mean_w[:, id_swap]

        img = generator(in_latent, input_is_latent=True)

        with torch.no_grad():
            real_feat = discriminator(targets)
        fake_feat = discriminator(img)

        loss = sum([F.l1_loss(a, b) for a, b in zip(fake_feat, real_feat)])/len(fake_feat)

        g_optim.zero_grad()
        loss.backward()
        g_optim.step()

    torch.save({"g": generator.state_dict()}, "models/"+args.model_name+".pt")