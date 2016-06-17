from image_processing import load_image, save_image
import argparse

DIM = 140
EXPAND = False
IMG_DIR = None

parser = argparse.ArgumentParser()

parser.add_argument('--img_dir', dest='img_dir')
parser.add_argument('--expand', dest='expand')
parser.add_argument('--dim', dest='dim', action="store", type=int)

args = parser.parse_args()

if args.img_dir:
    IMG_DIR = args.img_dir
if args.expand:
    EXPAND = True
if args.dim:
    DIM = args.dim

data = load_image(IMG_DIR, dim=DIM, expand_train=EXPAND, mode="RGB")
save_image(data, "bird_full_no_cropped_no_empty_{0}_rgb.pkl.gz".format(DIM))
