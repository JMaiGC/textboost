#!/usr/bin/env python3
import argparse
import glob
import os
import pickle

import open_clip
import torch
from PIL import Image
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Retrieve images from a dataset")
parser.add_argument(
    "--compute",
    action="store_true",
    help="Whether to compute image features",
)
parser.add_argument(
    "--dataset",
    type=str,
    default="/raid/backups/kunkim/ILSVRC/Data/CLS-LOC/train",
    help="Dataset to retrieve images from",
)
parser.add_argument(
    "--batch-size",
    type=int,
    default=256,
    help="Batch size for retrieving images",
)
parser.add_argument(
    "--gpu",
    type=str,
    default="3",
    help="Device to use for inference",
)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, root, transform):
        self.root = root
        self.transform = transform
        self.images = []
        if os.path.exists("filelist.pkl"):
            self.images = pickle.load(open("filelist.pkl", "rb"))
        else:
            paths = glob.glob(os.path.join(root, "**/*.JPEG"), recursive=True)
            paths = sorted(paths)
            print(len(paths))
            for path in paths:
                self.images.append(os.path.relpath(path, start=root))
            pickle.dump(self.images, open("filelist.pkl", "wb"))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        path = self.images[idx]
        full_path = os.path.join(self.root, path)
        image = Image.open(full_path).convert("RGB")
        return self.transform(image), image, path


def collate_fn(batch):
    images, raw_images, paths = zip(*batch)
    images = torch.stack(images, dim=0)
    return images, raw_images, paths


@torch.inference_mode()
def compute(args):
    device = torch.device(f"cuda:{args.gpu}")
    model, _, preprocess = open_clip.create_model_and_transforms(
        # 'ViT-H-14', pretrained='laion2b_s32b_b79k',
        'ViT-B-16', pretrained='datacomp_xl_s13b_b90k',
    )
    model.eval().requires_grad_(False)
    model = model.to(device)

    dataset = Dataset(args.dataset, transform=preprocess)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, num_workers=8,
        collate_fn=collate_fn, pin_memory=True, shuffle=False, drop_last=False,
    )

    emb_list = []
    path_list = []
    for images, raw_images, path in tqdm(loader):
        images = images.to(device)
        image_features = model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        emb_list.append(image_features.cpu())
        path_list += path
    emb_list = torch.cat(emb_list, dim=0)
    torch.save(emb_list, "imagenet-vit_b_16-datacomp_xl_s13b_b90k.pt")
    pickle.dump(path_list, open("imagenet_paths.pkl", "wb"))


def retrieve(args):
    pass


def main():
    args = parser.parse_args()
    if args.compute:
        compute(args)
    else:
        retrieve(args)


if __name__ == "__main__":
    main()
