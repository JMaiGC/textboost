import os
import pickle

import torch
from diffusers import DiffusionPipeline
from huggingface_hub.utils import insecure_hashlib
from PIL import Image
from tqdm import tqdm
from transformers import (BlipForConditionalGeneration, BlipProcessor,
                          PretrainedConfig)

from .dataset import PromptDataset


def encode_prompt(text_encoder, input_ids, attention_mask, text_encoder_use_attention_mask=None):
    text_input_ids = input_ids.to(text_encoder.device)

    if text_encoder_use_attention_mask:
        attention_mask = attention_mask.to(text_encoder.device)
    else:
        attention_mask = None

    prompt_embeds = text_encoder(
        text_input_ids,
        attention_mask=attention_mask,
        return_dict=False,
    )
    prompt_embeds = prompt_embeds[0]

    return prompt_embeds


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel

    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel

        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")


def generate_prior_images(
        class_images_dir,
        class_token,
        num_prior_images,
        args,
        logger,
        accelerator,
        use_blip_caption=False,
):
    if use_blip_caption:
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        blip = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        blip = blip.to(accelerator.device)

    if not class_images_dir.exists():
        class_images_dir.mkdir(parents=True)
    cur_class_images = len(list(class_images_dir.iterdir()))

    if cur_class_images < num_prior_images:
        torch_dtype = torch.float16 if accelerator.device.type == "cuda" else torch.float32
        if args.prior_generation_precision == "fp32":
            torch_dtype = torch.float32
        elif args.prior_generation_precision == "fp16":
            torch_dtype = torch.float16
        elif args.prior_generation_precision == "bf16":
            torch_dtype = torch.bfloat16
        pipeline = DiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            torch_dtype=torch_dtype,
            safety_checker=None,
            revision=args.revision,
            variant=args.variant,
        )
        pipeline.set_progress_bar_config(disable=True)

        num_new_images = num_prior_images - cur_class_images
        logger.info(f"Number of class images to sample: {num_new_images}.")

        sample_dataset = PromptDataset(class_token, num_samples=num_new_images)
        sample_dataloader = torch.utils.data.DataLoader(sample_dataset, batch_size=args.sample_batch_size)

        sample_dataloader = accelerator.prepare(sample_dataloader)
        pipeline.to(accelerator.device)

        for example in tqdm(
            sample_dataloader, desc="Generating class images", disable=not accelerator.is_local_main_process
        ):
            images = pipeline(example["prompt"]).images

            for i, image in enumerate(images):
                hash_image = insecure_hashlib.sha1(image.tobytes()).hexdigest()
                caption = example["prompt"][i].replace(' ', '_')
                image_filename = class_images_dir / f"{example['index'][i] + cur_class_images}-{hash_image}-{caption}.jpg"

                if use_blip_caption:
                    inputs = processor(image, return_tensors="pt").to(accelerator.device)
                    out = blip.generate(**inputs)
                    caption = processor.decode(out[0], skip_special_tokens=True)
                    image_filename = class_images_dir / f"{example['index'][i] + cur_class_images}-{caption.replace(' ', '_')}.jpg"

                image.save(image_filename)

        del pipeline
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def retrieve_prior_images(
        class_images_dir,
        class_token,
        num_prior_images,
        logger,
        accelerator,
):
    import open_clip

    if not class_images_dir.exists():
        class_images_dir.mkdir(parents=True)
    cur_class_images = len(list(class_images_dir.iterdir()))
    if cur_class_images >= num_prior_images:
        logger.info(f"Number of class images: {cur_class_images}.")
        return

    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    blip = blip.to(accelerator.device)

    embedding = torch.load("assets/imagenet-vit_b_16-datacomp_s13b_b90k.pt")
    embedding = embedding.to(accelerator.device)
    with open("assets/imagenet_paths.pkl", "rb") as f:
        paths = pickle.load(f)

    model, _, preprocess = open_clip.create_model_and_transforms(
        # 'ViT-H-14', pretrained='laion2b_s32b_b79k',
        'ViT-B-16', pretrained='datacomp_xl_s13b_b90k',
    )
    tokenizer = open_clip.get_tokenizer('ViT-B-16')
    model = model.to(accelerator.device)

    ROOT = "/raid/backups/kunkim/ILSVRC/Data/CLS-LOC/train"
    prompt = f"photo of a {class_token}"
    text_emb = model.encode_text(
        tokenizer([prompt]).to(accelerator.device)
    )

    sim = (embedding @ text_emb.T).squeeze(1)
    topk, indices = sim.topk(num_prior_images)
    for i, idx in enumerate(indices):
        full_path = os.path.join(ROOT, paths[idx])
        ext = full_path.split(".")[-1]
        pil_image = Image.open(full_path).convert("RGB")

        inputs = processor(pil_image, return_tensors="pt").to(accelerator.device)
        out = blip.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)
        clean_caption = caption.replace('/', '_').replace(' ', '_').replace('\'', '').replace('\"', '')

        pil_image.save(
            os.path.join(class_images_dir, f"{i}-{clean_caption}.{ext}")
        )

    del model, tokenizer, embedding, text_emb
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def add_token(
    text_encoder,
    tokenizer,
    placeholder_token,
    initializer_token,
    num_vectors=1,
):
    # Add the placeholder token in tokenizer
    placeholder_tokens = [placeholder_token]

    if num_vectors < 1:
        raise ValueError(f"--num_vectors has to be larger or equal to 1, but is {num_vectors}")
    # add dummy tokens for multi-vector
    additional_tokens = []
    if num_vectors > 1:
        if placeholder_token.endswith(">"):
            placeholder_tokens[0] = placeholder_token[:-1] + "_0>"
            for i in range(1, num_vectors):
                additional_tokens.append(f"{placeholder_token[:-1]}_{i}>")
        else:
            for i in range(1, num_vectors):
                additional_tokens.append(f"{placeholder_token}_{i}")
    placeholder_tokens += additional_tokens
    num_added_tokens = tokenizer.add_tokens(placeholder_tokens)
    if num_added_tokens != num_vectors:
        raise ValueError(
            f"The tokenizer already contains the token {placeholder_token}. Please pass a different"
            " `placeholder_token` that is not already in the tokenizer."
        )

    # Convert the initializer_token, placeholder_token to ids
    token_ids = tokenizer.encode(initializer_token, add_special_tokens=False)
    # Check if initializer_token is a single token or a sequence of tokens
    if len(token_ids) > len(placeholder_tokens):
        raise ValueError("The initializer token must be a single token.")

    initializer_token_ids = token_ids
    placeholder_token_ids = tokenizer.convert_tokens_to_ids(placeholder_tokens)

    # Resize the token embeddings as we are adding new special tokens to the tokenizer
    text_encoder.resize_token_embeddings(len(tokenizer))

    # Initialise the newly added placeholder token with the embeddings of the initializer token
    token_embeds = text_encoder.get_input_embeddings().weight.data
    with torch.no_grad():
        for token_id in placeholder_token_ids:
            for initializer_token_id in initializer_token_ids:
                token_embed = token_embeds[initializer_token_id].clone()
                token_embeds[token_id] = token_embed
    return placeholder_token_ids


def add_augmentation_tokens(
    text_encoder,
    tokenizer,
    aug_type="object",
):
    assert aug_type in ("object", "style"), \
        f"aug_type must be either 'object' or 'style', but is {aug_type}"

    if aug_type == "object":
        augmentations = {
            "<grayscale>": "gray",  # 1
            "<zoom-in>": "zoom in",  # 2
            "<zoom-out>": "far away",  # 2
            "<bright>": "bright",
            "<dimmed>": "dark",
            "<hflip>": "flip",
            "<vflip>": "flip",
            "<crop>": "crop",
            "<cutout>": "hole",
            "<left>": "on the left",  # 3
            "<right>": "on the right",  # 3
            "<grid>": "grid of",  # 2
        }
    else:
        augmentations = {
            "<hflip>": "flip",
            "<vflip>": "flip",
            "<rot90>": "rotate",
            "<rot270>": "rotate",
            "<jigsaw>": "jigsaw",
        }

    aug_token_ids = []
    aug_token_dict = dict()
    for placeholder_token, initializer_token in augmentations.items():
        token_ids = tokenizer.encode(initializer_token, add_special_tokens=False)
        num_vectors = len(token_ids)
        new_token = add_token(
            text_encoder,
            tokenizer,
            placeholder_token,
            initializer_token,
            num_vectors=num_vectors,
        )
        # assert len(new_token) == 1, f"{new_token} is not a single token."
        aug_token_ids += new_token
        if num_vectors > 1:
            for i, token_id in enumerate(new_token):
                placeholder = placeholder_token.replace(">", f"_{i}>")
                aug_token_dict[placeholder] = token_id
        else:
            aug_token_dict[placeholder_token] = new_token[0]
    return aug_token_ids, aug_token_dict
