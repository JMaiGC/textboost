#!/usr/bin/env python
import argparse
import subprocess
import os

# subject_name, class, init_token
INSTANCES = [
    ("backpack", "backpack", None),
    ("backpack_dog", "backpack", "dog"),
    ("bear_plushie", "stuffed animal", None),
    ("berry_bowl", "bowl", None),
    ("can", "can", None),
    ("candle", "candle", None),
    ("cat", "cat", None),
    ("cat2", "cat", None),
    ("clock", "clock", None),
    ("colorful_sneaker", "sneaker", None),
    ("dog", "dog", None),
    ("dog2", "dog", None),
    ("dog3", "dog", None),
    ("dog5", "dog", None),
    ("dog6", "dog", None),
    ("dog7", "dog", None),
    ("dog8", "dog", None),
    ("duck_toy", "toy", None),
    ("fancy_boot", "boot", None),
    ("grey_sloth_plushie", "stuffed animal", "sloth"),
    ("monster_toy", "toy", None),
    ("pink_sunglasses", "glasses", None),
    ("poop_emoji", "toy", None),
    ("rc_car", "toy", None),
    ("red_cartoon", "cartoon", None),
    ("robot_toy", "toy", None),
    ("shiny_sneaker", "sneaker", None),
    ("teapot", "teapot", None),
    ("vase", "vase", None),
    ("wolf_plushie", "stuffed animal", None),
]

parser = argparse.ArgumentParser(description='Run TextBoost experiment')
parser.add_argument("-g", "--gpu", type=str, default="0")
parser.add_argument("-n", "--num-samples", type=int, default=None)
parser.add_argument("-m", "--model", type=str, default="sd1.5")
parser.add_argument("--instances", type=str, nargs="+", default=None)

parser.add_argument("--lora-rank", type=int, default=4)
parser.add_argument("--total-steps", type=int, default=250)
parser.add_argument("--lr", type=float, default=5e-5)

parser.add_argument("--train-params", type=str, default="none")
parser.add_argument("--augment", type=str, default="pda")
parser.add_argument("--augment-p", type=float, default=0.8)
parser.add_argument("--null-prob", type=float, default=0.1)
parser.add_argument("--kp-weight", type=float, default=0.1)

parser.add_argument("--no-weighted-sample", action="store_true", default=False)
parser.add_argument("--no-inversion", action="store_true", default=False)

parser.add_argument("--desc", type=str, default=None)


def main(args):
    if args.instances is not None:
        instances = []
        for name, cls, init_token in INSTANCES:
            if name in args.instances:
                instances.append((name, cls, init_token))
    else:
        instances = INSTANCES

    num_str = "all" if args.num_samples is None else f"n{args.num_samples}"
    outdir = f"output/tb-{args.model}-{num_str}"
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

    resolution = {
        "sd1.4": 512,
        "sd1.5": 512,
        "sd2.1": 768,
        "sdxl": 1024,
    }[model]

    data_dir = "data/dreambooth_n1_train"

    num_gpu = len(args.gpu.split(","))
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torchrun_cmd = [
        "torchrun",
        "--rdzv-backend=c10d",
        "--rdzv-endpoint=localhost:0",
        f"--nproc-per-node={num_gpu}",
    ]
    for name, cls, init_token in instances:
        if init_token is None:
            init_token = cls
        if len(init_token.split(" ")) > 1:
            init_token = init_token.split(" ")[-1]
        cmd = [
            "train_textboost.py",
            f"--pretrained_model_name_or_path={args.model}",
            f"--instance_data_dir={os.path.join(data_dir, name)}",
            f"--output_dir=./{outdir}/{name}",
            f"--class_token={cls}",
            f"--instance_token=<{name}> {cls}",
            f"--validation_prompt=a <{name}> {cls} in the jungle",
            f"--validation_steps={args.total_steps // 5}",
            f"--placeholder_token=<{name}>",
            f"--initializer_token={init_token}",
            f"--resolution={resolution}",
            f"--lora_rank={args.lora_rank}",
            "--learning_rate=5e-5",
            "--emb_learning_rate=1e-3", 
            "--train_batch_size=8",
            f"--max_train_steps={args.total_steps}",
            f"--checkpointing_steps={args.total_steps // 5}",
            "--gradient_accumulation_steps=1",
            f"--unet_params_to_train={args.train_params}",
            f"--augment={args.augment}",
            f"--augment_p={args.augment_p}",
            f"--null_prob={args.null_prob}",
            f"--text_ppl_weight={args.kp_weight}",
        ]
        if args.num_samples is not None:
            cmd.append(f"--num_samples={args.num_samples}")
        if not args.no_inversion:
            cmd.append("--augment_inversion")
        if args.no_weighted_sample:
            cmd.append("--disable_weighted_sample")
        if args.augment == "none":
            cmd.append("--center_crop")
        subprocess.run(torchrun_cmd + cmd)

        # save cmd as text file
        cmd_txt = "\n".join(cmd)
        with open(f"{outdir}/{name}/cmd.txt", "w") as file:
            file.write(cmd_txt)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
