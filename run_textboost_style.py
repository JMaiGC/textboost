#!/usr/bin/env python
import argparse
import subprocess
import os

# Classes - ("subject_name", "class"),
INSTANCES = [
    ("00", "A canyon in {} style", "watercolor+painting"),
    ("01", "A house in {} style", "watercolor+painting"),
    ("02", "A cat in {} style", "watercolor+painting"),
    ("03", "Flowers in {} style", "watercolor+painting"),
    ("04", "A village in {} style", "oil+painting"),
    ("05", "A village in {} style", "line+drawing"),
    ("06", "A portrait of a man in {} style", "line+drawing"),
    ("07", "A portrait of a person wearing a hat in {} style", "oil+painting"),
    ("08", "A woman walking a dog in {} style", "flat+cartoon+illustration"),
    ("09", "A woman working on a laptop in {} style", "flat+cartoon+illustration"),
    ("10", "A Christmas tree in {} style", "sticker"),
    ("11", "A wave in {} style", "abstract+rainbow+colored+flowing+smoke+wave+design"),
    ("12", "A mushroom in {} style", "glowing"),
    ("13", "a cat sits in front of a window in {} style", "drawing"),
    ("14", "a path through the woods with trees and fog in {} style", "artistic"),
    ("15", "Slices of watermelon and clouds in the background in {} style", "3+d+rendering"),
    ("16", "A house in {} style", "3+d+rendering"),
    ("17", "A thumbs up in {} style", "glowing"),
    ("18", "A woman in {} style", "3+d+rendering"),
    ("19", "A bear in {} style animal", "kid+crayon+drawing"),
    ("20", "a statue of a man's head in {} style", "silver+sculpture"),
    ("21", "A flower in {} style", "melting+golden+3+d+rendering"),
    ("22", "A Viking face with beard in {} style", "wooden+sculpture"),
]

parser = argparse.ArgumentParser(description='Run TEPA experiment')
parser.add_argument("-g", "--gpu", type=str, default="7")
parser.add_argument("-m", "--model", type=str, default="sd2.1")
parser.add_argument("--instances", type=str, nargs="+", default=None)
parser.add_argument("--augment", type=str, default="pda")
parser.add_argument("--lora-rank", type=int, default=4)
parser.add_argument("--null-prob", type=float, default=0.1)
parser.add_argument("--kp-weight", type=float, default=0.1)
parser.add_argument("--no-weighted-sample", action="store_true", default=False)
parser.add_argument("--no-inversion", action="store_true", default=False)
parser.add_argument("--desc", type=str, default=None)


def main(args):
    if args.instances is not None:
        instances = []
        for name, cls in INSTANCES:
            if name in args.instances:
                instances.append((name, cls))
    else:
        instances = INSTANCES

    outdir = f"output/tb_style-{args.model}"
    if args.desc is not None:
        outdir += f"-{args.desc}"

    os.makedirs(outdir, exist_ok=True)

    model = args.model.lower().replace("-", "")
    if model == "sd1.4":
        args.model = "CompVis/stable-diffusion-v1-4"
    elif model == "sd1.5":
        # args.model = "runwayml/stable-diffusion-v1-5"
        args.model = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    elif model == "sd2.1":
        args.model = "stabilityai/stable-diffusion-2-1"
    elif model == "sdxl":
        args.model = "stabilityai/stable-diffusion-xl-base-1.0"

    num_gpu = len(args.gpu.split(","))
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torchrun_cmd = [
        "torchrun",
        "--rdzv-backend=c10d",
        "--rdzv-endpoint=localhost:0",
        f"--nproc-per-node={num_gpu}",
    ]
    for name, template, init_token in instances:
        token = init_token.split("+")
        token = "+".join([f"<{t}>" for t in token])
        instance_token = token.replace("<3> <d>", "<3><d>")
        val_token = token.replace("+", " ").replace("<3> <d>", "<3><d>")
        cmd = [
            "train_textboost.py",
            f"--pretrained_model_name_or_path={args.model}",
            f"--instance_data_dir=./data/styledrop/{name}",
            f"--class_data_dir=./data/gen_prior/dog",
            f"--output_dir=./{outdir}/{name}",
            "--class_token=dog",  # NOTE: not used
            f"--instance_token={instance_token}",
            f"--validation_prompt=a dog in {val_token} style",
            "--validation_steps=40",
            "--placeholder_token", f"{token}",
            "--initializer_token", f"{init_token}",
            f"--lora_rank={args.lora_rank}",
            "--learning_rate=1e-4",
            "--emb_learning_rate=1e-3",
            "--train_batch_size=4", 
            "--max_train_steps=200",
            "--checkpointing_steps=40",
            "--gradient_accumulation_steps=1",
            f"--augment={args.augment}",
            f"--text_ppl_weight={args.kp_weight}",
            f"--null_prob={args.null_prob}",
            f"--template={template}",
            "--augment_ops=style",
        ]
        if not args.no_inversion:
            cmd.append("--augment_inversion")
        if args.no_weighted_sample:
            cmd.append("--disable_weighted_sample")
        subprocess.run(torchrun_cmd + cmd)

        # save cmd as text file
        cmd_txt = "\n".join(cmd)
        with open(f"{outdir}/{name}/cmd.txt", "w") as file:
            file.write(cmd_txt)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
