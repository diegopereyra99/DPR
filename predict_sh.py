import os
from utils.inference import image_generator, load_model
import argparse
import numpy as np
from tqdm import tqdm
import csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("src_dir", type=str)
    parser.add_argument("--dst", default="predicted_sh.csv", type=str)
    parser.add_argument("--img_list", "-l", default=None, type=str)
    parser.add_argument("--batch_size", "-b", default=32, type=int)
    # parser.add_argument()
    args = parser.parse_args()

    if args.img_list:
        with open(args.img_list, "r") as fp:
            img_fns = fp.read().splitlines()
    else:
        img_fns = os.listdir(args.src_dir)

    img_fns = [fn for fn in img_fns if os.path.exists(os.path.join(args.src_dir, fn))]    

    batches = image_generator(img_fns, args.src_dir, batch_size=args.batch_size)
    print("Loading Model...")
    model = load_model()
    print("Model loaded")

    predicted_sh = []
    for batch in tqdm(batches, total=np.ceil(len(img_fns) / args.batch_size).astype(int)):
        _, est_sh = model(batch, None, 0, only_embbeding=True)
        sh = est_sh[..., 0, 0].tolist()
        predicted_sh.extend(sh)

    with open(args.dst, "w") as file:
        csv_writer = csv.writer(file)
        for img_fn, sh in zip(img_fns, predicted_sh):
            csv_writer.writerow([img_fn] + sh)

    # txt_fn = os.path.splitext(img_fn)[0] + ".txt"
    # np.savetxt(os.path.join(args.dst_dir, txt_fn), np.array(sh), fmt="%.7f")


if __name__ == "__main__":
    main()
