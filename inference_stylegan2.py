import jittor as jt
from models.psp_stylegan2 import pSp
import argparse

import jittor.transform as transform
from PIL import Image
import numpy as np

img_size = 256
transform_image = transform.Compose([
        transform.Resize(size = img_size),
        transform.ImageNormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

def read_img(path):
    img = Image.open(path).convert('RGB')
    img = transform_image(img)
    img = jt.array(img)
    img = img.unsqueeze(0)
    return img

def save_img(image, path):
    image = image.squeeze(0).detach().numpy()
    image = (np.transpose(image, (1, 2, 0)) + 1) / 2.0 * 255.0
    image = np.clip(image, 0, 255).astype(np.uint8)
    image = Image.fromarray(image)
    image.save(path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference")
    parser.add_argument("--images_path", type=str, default="./imgs/input/69918.png",
                        help="The path of the images to be inverted")
    parser.add_argument("--save_path", type=str, default="./imgs/output/69918_stylegan2.png",
                        help="The path to save the inversion images. (default: images_dir)")
    parser.add_argument("--ckpt",  type=str, default="./weights/e4e_ffhq_encode_stylegan2.pt", help="path to generator checkpoint")

    args = parser.parse_args()

    checkpoint_path = args.ckpt
    ckpt = jt.load(checkpoint_path)
    opts = ckpt['opts']
    opts['checkpoint_path'] = checkpoint_path
    opts = argparse.Namespace(**opts)

    net = pSp(opts)
    net.eval()

    img = read_img(args.images_path)
    e4e_img = net(img, resize = False)
    save_img(e4e_img, args.save_path)
    
#python inference_stylegan2.py --images_path ./imgs/input/69918.png --save_path ./imgs/output/69918_stylegan2.png --ckpt ./weights/e4e_ffhq_encode_stylegan2.pt

