# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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
from losses.diffusion_losses import noise_prediction_loss
from losses.id_loss import build_id_loss
from losses.lpips_loss import LPIPSLoss, masked_blend_pair, masked_l1


def save_checkpoint(model: MVFSTeacherSDTurbo, out_dir: Path, opt_step: int, micro_step: int | None = None):
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_unet = any(p.requires_grad for p in model.unet.parameters())
    ckpt = {
        "pose_guider": model.pose_guider.state_dict(),
        "id_adapter": model.id_adapter.state_dict(),
        "unet": model.unet.state_dict() if train_unet else None,
        "opt_step": opt_step,
        "micro_step": micro_step,
        "step": opt_step,
        "train_unet": train_unet,
        "condition_channels": getattr(model, "condition_channels", None),
    }

    if micro_step is None:
        name = f"teacher_opt_{opt_step:07d}.pt"
    else:
        name = f"teacher_micro_{micro_step:07d}_opt_{opt_step:07d}.pt"

    torch.save(ckpt, ckpt_dir / name)


def load_checkpoint(model: MVFSTeacherSDTurbo, ckpt_path: str | Path):
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    if ckpt.get("pose_guider") is not None:
        model.pose_guider.load_state_dict(ckpt["pose_guider"], strict=True)
    if ckpt.get("id_adapter") is not None:
        model.id_adapter.load_state_dict(ckpt["id_adapter"], strict=True)
    if ckpt.get("unet") is not None:
        model.unet.load_state_dict(ckpt["unet"], strict=True)
    return int(ckpt.get("opt_step", ckpt.get("step", 0)))


def tensor_to_bgr_uint8(x: torch.Tensor) -> np.ndarray:
    x = torch.nan_to_num(x.detach().float().cpu(), nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1, 1)
    x = ((x + 1.0) * 0.5).permute(1, 2, 0).numpy()
    return cv2.cvtColor(np.clip(x * 255, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def mask_lm_overlay_bgr(face_mask: torch.Tensor, landmark_map: torch.Tensor) -> np.ndarray:
    """
    Debug view: face segmentation as grayscale, landmark as green only.
    """
    mask = face_mask.detach().float().cpu()[0].clamp(0, 1).numpy()
    lm = landmark_map.detach().float().cpu()
    if lm.ndim == 3:
        lm = lm.max(dim=0).values
    lm = lm.clamp(0, 1).numpy()

    base = np.clip(mask * 180, 0, 255).astype(np.uint8)
    out = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)

    green = np.zeros_like(out)
    green[..., 1] = np.clip(lm * 255, 0, 255).astype(np.uint8)
    out = np.maximum(out, green)
    return out


def add_caption(img_bgr: np.ndarray, text: str, bar_h: int = 32) -> np.ndarray:
    bar = np.full((bar_h, img_bgr.shape[1], 3), 245, dtype=np.uint8)
    cv2.putText(bar, text, (10, int(bar_h * 0.7)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2, cv2.LINE_AA)
    return np.vstack([bar, img_bgr])


def make_debug_panel(identity, condition_rgb, face_mask, landmark_map, recon, clean) -> np.ndarray:
    return np.hstack([
        add_caption(tensor_to_bgr_uint8(identity), "ID"),
        add_caption(tensor_to_bgr_uint8(condition_rgb), "BLUR CONDITION"),
        add_caption(mask_lm_overlay_bgr(face_mask, landmark_map), "FACE SEG + LM"),
        add_caption(tensor_to_bgr_uint8(recon), "RECON"),
        add_caption(tensor_to_bgr_uint8(clean), "GT"),
    ])


def save_debug_images(out_dir: Path, step: int, epoch: int, batch, recon_images: torch.Tensor, max_samples: int = 1):
    debug_dir = out_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    n = min(max_samples, batch["clean"].shape[0], recon_images.shape[0])
    rows = []
    for i in range(n):
        panel = make_debug_panel(
            batch["identity"][i],
            batch["condition_rgb"][i],
            batch["face_mask"][i],
            batch["landmark_map"][i],
            recon_images[i],
            batch["clean"][i],
        )
        meta = np.full((38, panel.shape[1], 3), 255, dtype=np.uint8)
        person_id = batch.get("person_id", [""] * n)[i] if isinstance(batch.get("person_id"), list) else ""
        cv2.putText(
            meta,
            f"micro={step} epoch={epoch} sample={i} person_id={person_id}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (30, 30, 30),
            2,
            cv2.LINE_AA,
        )
        rows.append(np.vstack([meta, panel]))

    cv2.imwrite(str(debug_dir / f"micro_{step:07d}.jpg"), rows[0] if len(rows) == 1 else np.vstack(rows))


def build_parser():
    p = argparse.ArgumentParser("Train MVFS teacher with face-seg blur + 3DDFA landmark condition + LPIPS")
    p.add_argument("--index", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--pretrained", default="stabilityai/sd-turbo")
    p.add_argument("--device", default="cuda")
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum-steps", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--max-steps", type=int, default=65000, help="optimizer steps, not micro-batches")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--train-unet", action="store_true")
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--resume", type=str, default="")

    p.add_argument("--condition-channels", type=int, default=4)
    p.add_argument("--landmark-sigma", type=float, default=2.0)
    p.add_argument("--no-landmark-lines", action="store_true")
    p.add_argument("--blur-downsample-size", type=int, default=8)
    p.add_argument("--blur-gaussian-radius", type=float, default=8.0)
    p.add_argument("--blur-feather-sigma", type=float, default=0.0)

    p.add_argument("--id-dim", type=int, default=512)
    p.add_argument("--num-id-tokens", type=int, default=4)
    p.add_argument("--random-identity-same-dir", action="store_true")
    p.add_argument("--prompt", default="")
    p.add_argument("--fixed-high-timestep", type=int, default=999)
    p.add_argument("--noise-loss", default="mse", choices=["mse", "l1"])

    p.add_argument("--lambda-noise", type=float, default=1.0)
    p.add_argument("--lambda-l1", type=float, default=1.0)
    p.add_argument("--lambda-lpips", type=float, default=10.0)
    p.add_argument("--lambda-id", type=float, default=1.0)

    p.add_argument("--id-loss-start-step", type=int, default=50000)
    p.add_argument("--id-loss-every", type=int, default=1)
    p.add_argument("--id-loss-target", type=str, default="identity", choices=["identity", "clean"])
    p.add_argument("--id-loss-backend", type=str, default="facenet", choices=["facenet", "torchscript"])
    p.add_argument("--facenet-pretrained", type=str, default="vggface2")
    p.add_argument("--id-loss-model", type=str, default="")
    p.add_argument("--id-loss-input-range", type=str, default="minus1_1", choices=["minus1_1", "0_1", "imagenet"])

    p.add_argument("--lpips-net", type=str, default="alex", choices=["alex", "vgg", "squeeze"])

    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--save-every", type=int, default=1000, help="optimizer-step checkpoint interval")
    p.add_argument("--save-every-micro", type=int, default=0, help="optional micro-step checkpoint interval")
    p.add_argument("--debug-image-every", type=int, default=100)
    p.add_argument("--debug-max-save", type=int, default=200)
    p.add_argument("--debug-num-samples", type=int, default=1)
    return p


def main(args):
    device = args.device
    dtype = torch.float16 if args.fp16 else torch.float32

    ds = TeacherBlurDataset(
        args.index,
        image_size=args.image_size,
        random_identity_same_dir=args.random_identity_same_dir,
        landmark_sigma=args.landmark_sigma,
        landmark_draw_lines=not args.no_landmark_lines,
        blur_downsample_size=args.blur_downsample_size,
        blur_gaussian_radius=args.blur_gaussian_radius,
        blur_feather_sigma=args.blur_feather_sigma,
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
        condition_channels=args.condition_channels,
        train_unet=args.train_unet,
        device=device,
        dtype=dtype,
    )

    resume_opt_step = 0
    if args.resume:
        resume_opt_step = load_checkpoint(model, args.resume)
        print(f"[RESUME] loaded {args.resume}, checkpoint opt_step={resume_opt_step}")

    model.train()
    model.vae.eval()
    model.text_encoder.eval()
    model.unet.train(args.train_unet)

    if args.gradient_checkpointing and hasattr(model.unet, "enable_gradient_checkpointing"):
        model.unet.enable_gradient_checkpointing()
        print("[INFO] UNet gradient checkpointing enabled")

    id_loss_fn = None
    if args.lambda_id > 0:
        id_loss_fn = build_id_loss(
            backend=args.id_loss_backend,
            device=device,
            model_path=args.id_loss_model,
            facenet_pretrained=args.facenet_pretrained,
            input_range=args.id_loss_input_range,
        ).eval()

    lpips_loss_fn = LPIPSLoss(net=args.lpips_net, device=device).eval() if args.lambda_lpips > 0 else None

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.fp16)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "train_config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    grad_accum = max(1, int(args.grad_accum_steps))
    opt.zero_grad(set_to_none=True)

    opt_step = resume_opt_step
    micro_step = 0
    accum_count = 0
    num_debug_saved = 0
    skipped_nonfinite = 0

    for epoch in range(args.epochs):
        pbar = tqdm(dl, desc=f"ep{epoch}", dynamic_ncols=True, leave=True)
        for batch in pbar:
            clean = batch["clean"].to(device, non_blocking=True)
            cond = batch["condition"].to(device, non_blocking=True)
            identity = batch["identity"].to(device, non_blocking=True)
            face_mask = batch["face_mask"].to(device, non_blocking=True)

            id_embed = batch.get("id_embed")
            if id_embed is not None:
                id_embed = id_embed.to(device, non_blocking=True)
            else:
                id_embed = torch.zeros((clean.shape[0], args.id_dim), device=device, dtype=torch.float32)

            id_active = args.lambda_id > 0 and opt_step >= args.id_loss_start_step
            id_this_step = id_active and (micro_step % args.id_loss_every == 0)
            need_debug = args.debug_image_every > 0 and micro_step % args.debug_image_every == 0 and num_debug_saved < args.debug_max_save
            decode_recon = True  # L1/LPIPS/debug all need recon.

            with torch.amp.autocast("cuda", enabled=args.fp16):
                out = model(
                    clean=clean,
                    condition=cond,
                    id_embed=id_embed,
                    prompt=args.prompt,
                    decode_recon=decode_recon,
                    fixed_high_timestep=args.fixed_high_timestep,
                )
                loss_noise = noise_prediction_loss(out.model_pred, out.target, args.noise_loss)

            if out.recon_image is not None:
                recon_face, gt_face = masked_blend_pair(out.recon_image.float(), clean.float(), face_mask.float())
                loss_l1 = masked_l1(recon_face, gt_face, face_mask.float()) if args.lambda_l1 > 0 else clean.new_tensor(0.0)
                loss_lpips = lpips_loss_fn(recon_face, gt_face) if lpips_loss_fn is not None else clean.new_tensor(0.0)
            else:
                loss_l1 = clean.new_tensor(0.0)
                loss_lpips = clean.new_tensor(0.0)

            if id_loss_fn is not None and id_this_step and out.recon_image is not None:
                with torch.amp.autocast("cuda", enabled=False):
                    id_target = clean if args.id_loss_target == "clean" else identity
                    loss_id = id_loss_fn(out.recon_image.float(), id_target.float())
            else:
                loss_id = clean.new_tensor(0.0)

            loss = (
                args.lambda_noise * loss_noise.float()
                + args.lambda_l1 * loss_l1.float()
                + args.lambda_lpips * loss_lpips.float()
                + args.lambda_id * loss_id.float()
            )

            if not torch.isfinite(loss):
                skipped_nonfinite += 1
                opt.zero_grad(set_to_none=True)
                accum_count = 0
                if need_debug and out.recon_image is not None:
                    save_debug_images(out_dir, micro_step, epoch, batch, out.recon_image, args.debug_num_samples)
                    num_debug_saved += 1
                micro_step += 1
                continue

            scaler.scale(loss / grad_accum).backward()
            accum_count += 1

            if need_debug and out.recon_image is not None:
                save_debug_images(out_dir, micro_step, epoch, batch, out.recon_image, args.debug_num_samples)
                num_debug_saved += 1

            if args.save_every_micro > 0 and micro_step > 0 and micro_step % args.save_every_micro == 0:
                save_checkpoint(model, out_dir, opt_step, micro_step)
                print(f"[SAVE] micro={micro_step}, opt={opt_step}")

            if accum_count >= grad_accum:
                if args.grad_clip > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)

                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)

                opt_step += 1
                accum_count = 0

                if args.save_every > 0 and opt_step > 0 and opt_step % args.save_every == 0:
                    save_checkpoint(model, out_dir, opt_step, micro_step)
                    print(f"[SAVE] opt={opt_step}, micro={micro_step}")

            if micro_step % args.log_every == 0:
                pbar.set_postfix_str(
                    f"L={loss.item():.3f} N={loss_noise.item():.3f} "
                    f"L1={loss_l1.item():.3f} P={loss_lpips.item():.3f} "
                    f"I={loss_id.item():.3f}"
                    # f"id={int(id_active)} s={opt_step} a={accum_count} skip={skipped_nonfinite}"
                )

            micro_step += 1

            if args.max_steps > 0 and opt_step >= args.max_steps:
                save_checkpoint(model, out_dir, opt_step, micro_step)
                return

    save_checkpoint(model, out_dir, opt_step, micro_step)


if __name__ == "__main__":
    args = build_parser().parse_args()
    if args.fixed_high_timestep < 0:
        args.fixed_high_timestep = None
    main(args)
