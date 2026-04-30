# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.teacher_blur_dataset import TeacherBlurDataset
from models.teacher_sd_turbo import MVFSTeacherSDTurbo
from losses.diffusion_losses import noise_prediction_loss, reconstruction_l1_loss


def save_checkpoint(model: MVFSTeacherSDTurbo, out_dir: Path, step: int):
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "pose_guider": model.pose_guider.state_dict(),
        "id_adapter": model.id_adapter.state_dict(),
        "step": step,
    }
    torch.save(ckpt, ckpt_dir / f"teacher_adapters_step_{step:07d}.pt")


def tensor_to_bgr_uint8(x: torch.Tensor) -> np.ndarray:
    """
    x: 3xHxW in [-1,1]
    return: HxWx3 BGR uint8
    """
    x = x.detach().float().cpu().clamp(-1, 1)
    x = (x + 1.0) * 0.5
    x = x.permute(1, 2, 0).numpy()
    x = np.clip(x * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(x, cv2.COLOR_RGB2BGR)


def add_caption(img_bgr: np.ndarray, text: str, bar_h: int = 32) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    bar = np.full((bar_h, w, 3), 245, dtype=np.uint8)
    cv2.putText(bar, text, (10, int(bar_h * 0.7)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2, cv2.LINE_AA)
    return np.vstack([bar, img_bgr])


def make_debug_panel(identity: torch.Tensor, condition: torch.Tensor, recon: torch.Tensor, clean: torch.Tensor) -> np.ndarray:
    id_img = add_caption(tensor_to_bgr_uint8(identity), "ID")
    cond_img = add_caption(tensor_to_bgr_uint8(condition), "BLUR CONDITION")
    recon_img = add_caption(tensor_to_bgr_uint8(recon), "RECON")
    clean_img = add_caption(tensor_to_bgr_uint8(clean), "GT")
    return np.hstack([id_img, cond_img, recon_img, clean_img])


def save_debug_images(
    out_dir: Path,
    step: int,
    epoch: int,
    batch,
    recon_images: torch.Tensor,
    max_samples: int = 2,
):
    debug_dir = out_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    n = min(max_samples, batch["clean"].shape[0], recon_images.shape[0])
    rows = []
    for i in range(n):
        panel = make_debug_panel(
            identity=batch["identity"][i],
            condition=batch["condition"][i],
            recon=recon_images[i],
            clean=batch["clean"][i],
        )
        meta_h = 38
        meta = np.full((meta_h, panel.shape[1], 3), 255, dtype=np.uint8)
        person_id = batch.get("person_id", [""] * n)[i] if isinstance(batch.get("person_id"), list) else ""
        id_path = batch.get("identity_path", [""] * n)[i] if isinstance(batch.get("identity_path"), list) else ""
        txt = f"step={step} epoch={epoch} sample={i} person_id={person_id}"
        cv2.putText(meta, txt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 2, cv2.LINE_AA)
        rows.append(np.vstack([meta, panel]))

    canvas = rows[0] if len(rows) == 1 else np.vstack(rows)
    save_path = debug_dir / f"step_{step:07d}.jpg"
    cv2.imwrite(str(save_path), canvas)


def main(args):
    device = args.device
    dtype = torch.float16 if args.fp16 else torch.float32

    ds = TeacherBlurDataset(
        args.index,
        image_size=args.image_size,
        random_identity_same_dir=args.random_identity_same_dir,
    )
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    model = MVFSTeacherSDTurbo(
        pretrained_model_name_or_path=args.pretrained,
        id_dim=args.id_dim,
        num_id_tokens=args.num_id_tokens,
        train_unet=args.train_unet,
        device=device,
        dtype=dtype,
    )
    model.train()
    model.vae.eval()
    model.text_encoder.eval()
    if not args.train_unet:
        model.unet.eval()

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "train_config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    step = 0
    num_debug_saved = 0

    for epoch in range(args.epochs):
        pbar = tqdm(dl, desc=f"epoch {epoch}")
        for batch in pbar:
            clean = batch["clean"].to(device, non_blocking=True)
            cond = batch["condition"].to(device, non_blocking=True)
            id_embed = batch.get("id_embed")
            if id_embed is not None:
                id_embed = id_embed.to(device, non_blocking=True)
            else:
                id_embed = torch.zeros((clean.shape[0], args.id_dim), device=device, dtype=torch.float32)

            need_debug = (
                args.debug_image_every > 0
                and step % args.debug_image_every == 0
                and num_debug_saved < args.debug_max_save
            )
            decode_recon = (
                (args.lambda_recon > 0 and (step % args.recon_every == 0))
                or need_debug
            )

            out = model(
                clean=clean,
                condition=cond,
                id_embed=id_embed,
                prompt=args.prompt,
                decode_recon=decode_recon,
                fixed_high_timestep=args.fixed_high_timestep,
            )

            loss_noise = noise_prediction_loss(out.model_pred, out.target, args.noise_loss)
            loss_recon = reconstruction_l1_loss(out.recon_image, clean) if out.recon_image is not None else clean.new_tensor(0.0)
            loss = args.lambda_noise * loss_noise + args.lambda_recon * loss_recon

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
            opt.step()

            if step % args.log_every == 0:
                pbar.set_postfix({
                    "loss": float(loss.item()),
                    "noise": float(loss_noise.item()),
                    "recon": float(loss_recon.item()),
                })

            if need_debug and out.recon_image is not None:
                save_debug_images(
                    out_dir=out_dir,
                    step=step,
                    epoch=epoch,
                    batch=batch,
                    recon_images=out.recon_image,
                    max_samples=args.debug_num_samples,
                )
                num_debug_saved += 1

            if args.save_every > 0 and step > 0 and step % args.save_every == 0:
                save_checkpoint(model, out_dir, step)

            step += 1
            if args.max_steps > 0 and step >= args.max_steps:
                save_checkpoint(model, out_dir, step)
                return

    save_checkpoint(model, out_dir, step)


def build_parser():
    p = argparse.ArgumentParser("Train MVFS blur-condition teacher")
    p.add_argument("--index", required=True, help="teacher_index.jsonl")
    p.add_argument("--output", required=True)
    p.add_argument("--pretrained", default="stabilityai/sd-turbo")
    p.add_argument("--device", default="cuda")
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--train-unet", action="store_true", help="Not recommended for 12GB. Prefer adapter-only first.")
    p.add_argument("--id-dim", type=int, default=512)
    p.add_argument("--num-id-tokens", type=int, default=4)
    p.add_argument("--random-identity-same-dir", action="store_true")
    p.add_argument("--prompt", default="")
    p.add_argument("--fixed-high-timestep", type=int, default=999)
    p.add_argument("--noise-loss", default="mse", choices=["mse", "l1"])
    p.add_argument("--lambda-noise", type=float, default=1.0)
    p.add_argument("--lambda-recon", type=float, default=0.1)
    p.add_argument("--recon-every", type=int, default=4, help="Decode recon every N steps to save VRAM/time.")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--save-every", type=int, default=1000)

    p.add_argument("--debug-image-every", type=int, default=100, help="Save debug panel every N steps. 0 disables.")
    p.add_argument("--debug-max-save", type=int, default=200, help="Maximum number of debug files to save.")
    p.add_argument("--debug-num-samples", type=int, default=1, help="How many samples to show per debug panel.")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    if args.fixed_high_timestep < 0:
        args.fixed_high_timestep = None
    main(args)
