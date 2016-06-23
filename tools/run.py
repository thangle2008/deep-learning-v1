from image_processing import load_image, save_image
import argparse

DIM = 140
EXPAND = False
IMG_DIR = None

parser = argparse.ArgumentParser()

parser.add_argument('--img_dir', dest='img_dir')
parser.add_argument('--expand', dest='expand')
parser.add_argument('--dim', dest='dim', action="store", type=int)
parser.add_argument('--mode', dest='mode')
parser.add_argument('--add_gray', dest='add_gray', action="store_true")
parser.add_argument('--compounded', dest='compounded', action="store_true")

args = parser.parse_args()

if args.img_dir:
    IMG_DIR = args.img_dir
if args.expand:
    EXPAND = True
if args.dim:
    DIM = args.dim

data = load_image(IMG_DIR, dim=DIM, expand_train=EXPAND, mode=args.mode, add_gray=args.add_gray, 
                  compounded=args.compounded)
save_image(data, "bird_full_no_cropped_no_empty_{0}_{1}.pkl.gz".format(DIM, args.mode))
