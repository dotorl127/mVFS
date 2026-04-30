# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse, json, sys
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
from losses.diffusion_losses import noise_prediction_loss, reconstruction_l1_loss
from losses.id_loss import build_id_loss

def save_checkpoint(model: MVFSTeacherSDTurbo, out_dir: Path, step: int):
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    train_unet = any(p.requires_grad for p in model.unet.parameters())
    torch.save({
        "pose_guider": model.pose_guider.state_dict(),
        "id_adapter": model.id_adapter.state_dict(),
        "unet": model.unet.state_dict() if train_unet else None,
        "step": step,
        "train_unet": train_unet,
        "condition_channels": getattr(model, "condition_channels", None),
    }, ckpt_dir / f"teacher_step_{step:07d}.pt")

def load_checkpoint(model: MVFSTeacherSDTurbo, ckpt_path: str | Path):
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    if ckpt.get("pose_guider") is not None:
        model.pose_guider.load_state_dict(ckpt["pose_guider"], strict=True)
    if ckpt.get("id_adapter") is not None:
        model.id_adapter.load_state_dict(ckpt["id_adapter"], strict=True)
    if ckpt.get("unet") is not None:
        model.unet.load_state_dict(ckpt["unet"], strict=True)
    return int(ckpt.get("step", 0))

def tensor_to_bgr_uint8(x: torch.Tensor) -> np.ndarray:
    x = torch.nan_to_num(x.detach().float().cpu(), nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1, 1)
    x = ((x + 1.0) * 0.5).permute(1, 2, 0).numpy()
    return cv2.cvtColor(np.clip(x * 255, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

def landmark_map_to_bgr_uint8(m: torch.Tensor) -> np.ndarray:
    m = m.detach().float().cpu()
    x = m[0] if m.ndim == 3 and m.shape[0] == 1 else (m.max(dim=0).values if m.ndim == 3 else m)
    img = np.clip(x.clamp(0, 1).numpy() * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

def add_caption(img_bgr: np.ndarray, text: str, bar_h: int = 32) -> np.ndarray:
    bar = np.full((bar_h, img_bgr.shape[1], 3), 245, dtype=np.uint8)
    cv2.putText(bar, text, (10, int(bar_h * 0.7)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20,20,20), 2, cv2.LINE_AA)
    return np.vstack([bar, img_bgr])

def make_debug_panel(identity, condition_rgb, landmark_map, recon, clean) -> np.ndarray:
    return np.hstack([
        add_caption(tensor_to_bgr_uint8(identity), "ID"),
        add_caption(tensor_to_bgr_uint8(condition_rgb), "BLUR CONDITION"),
        add_caption(landmark_map_to_bgr_uint8(landmark_map), "PROJECTED 3DDFA LM"),
        add_caption(tensor_to_bgr_uint8(recon), "RECON"),
        add_caption(tensor_to_bgr_uint8(clean), "GT"),
    ])

def save_debug_images(out_dir: Path, step: int, epoch: int, batch, recon_images: torch.Tensor, max_samples: int = 1):
    debug_dir = out_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    n = min(max_samples, batch["clean"].shape[0], recon_images.shape[0])
    rows = []
    for i in range(n):
        panel = make_debug_panel(batch["identity"][i], batch["condition_rgb"][i], batch["landmark_map"][i], recon_images[i], batch["clean"][i])
        meta = np.full((38, panel.shape[1], 3), 255, dtype=np.uint8)
        person_id = batch.get("person_id", [""] * n)[i] if isinstance(batch.get("person_id"), list) else ""
        cv2.putText(meta, f"step={step} epoch={epoch} sample={i} person_id={person_id}", (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30,30,30), 2, cv2.LINE_AA)
        rows.append(np.vstack([meta, panel]))
    cv2.imwrite(str(debug_dir / f"step_{step:07d}.jpg"), rows[0] if len(rows)==1 else np.vstack(rows))

def main(args):
    device = args.device
    dtype = torch.float16 if args.fp16 else torch.float32

    ds = TeacherBlurDataset(
        args.index,
        image_size=args.image_size,
        random_identity_same_dir=args.random_identity_same_dir,
        use_landmark_condition=not args.no_landmark_condition,
        landmark_map_mode=args.landmark_map_mode,
        landmark_sigma=args.landmark_sigma,
        landmark_draw_lines=not args.no_landmark_lines,
    )
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    expected_ch = 3 if args.no_landmark_condition else (7 if args.landmark_map_mode == "parts" else 4)
    if args.condition_channels != expected_ch:
        print(f"[WARN] overriding condition_channels {args.condition_channels} -> {expected_ch}")
        args.condition_channels = expected_ch

    model = MVFSTeacherSDTurbo(
        pretrained_model_name_or_path=args.pretrained,
        id_dim=args.id_dim,
        num_id_tokens=args.num_id_tokens,
        condition_channels=args.condition_channels,
        train_unet=args.train_unet,
        device=device,
        dtype=dtype,
    )
    if args.resume:
        print(f"[RESUME] loaded {args.resume}, checkpoint step={load_checkpoint(model, args.resume)}")

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

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.fp16)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "train_config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    grad_accum = max(1, int(args.grad_accum_steps))
    opt.zero_grad(set_to_none=True)
    global_step = micro_step = accum_count = num_debug_saved = skipped_nonfinite = 0

    for epoch in range(args.epochs):
        pbar = tqdm(dl, desc=f"epoch {epoch}")
        for batch in pbar:
            clean = batch["clean"].to(device, non_blocking=True)
            cond = batch["condition"].to(device, non_blocking=True)
            identity = batch["identity"].to(device, non_blocking=True)
            id_embed = batch.get("id_embed")
            id_embed = id_embed.to(device, non_blocking=True) if id_embed is not None else torch.zeros((clean.shape[0], args.id_dim), device=device)

            id_active = args.lambda_id > 0 and global_step >= args.id_loss_start_step
            id_this_step = id_active and (micro_step % args.id_loss_every == 0)
            need_debug = args.debug_image_every > 0 and micro_step % args.debug_image_every == 0 and num_debug_saved < args.debug_max_save
            decode_recon = (args.lambda_recon > 0 and micro_step % args.recon_every == 0) or id_this_step or need_debug

            with torch.amp.autocast("cuda", enabled=args.fp16):
                out = model(clean=clean, condition=cond, id_embed=id_embed, prompt=args.prompt, decode_recon=decode_recon, fixed_high_timestep=args.fixed_high_timestep)
                loss_noise = noise_prediction_loss(out.model_pred, out.target, args.noise_loss)
                loss_recon = reconstruction_l1_loss(out.recon_image, clean) if (out.recon_image is not None and args.lambda_recon > 0) else clean.new_tensor(0.0)

            if id_loss_fn is not None and id_this_step and out.recon_image is not None:
                with torch.amp.autocast("cuda", enabled=False):
                    id_target = clean if args.id_loss_target == "clean" else identity
                    loss_id = id_loss_fn(out.recon_image.float(), id_target.float())
            else:
                loss_id = clean.new_tensor(0.0)

            loss = args.lambda_noise * loss_noise.float() + args.lambda_recon * loss_recon.float() + args.lambda_id * loss_id.float()

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

            if accum_count >= grad_accum:
                if args.grad_clip > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                global_step += 1
                accum_count = 0
                if args.save_every > 0 and global_step > 0 and global_step % args.save_every == 0:
                    save_checkpoint(model, out_dir, global_step)

            if micro_step % args.log_every == 0:
                pbar.set_postfix({
                    "loss": float(loss.item()), "noise": float(loss_noise.item()),
                    "recon": float(loss_recon.item()), "id": float(loss_id.item()),
                    "id_on": int(id_active), "opt_step": global_step, "accum": accum_count,
                    "skip": skipped_nonfinite,
                })

            micro_step += 1
            if args.max_steps > 0 and global_step >= args.max_steps:
                save_checkpoint(model, out_dir, global_step)
                return

    save_checkpoint(model, out_dir, global_step)

def build_parser():
    p = argparse.ArgumentParser("Train MVFS teacher with projected 3DDFA landmark condition")
    p.add_argument("--index", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--pretrained", default="stabilityai/sd-turbo")
    p.add_argument("--device", default="cuda")
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum-steps", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--max-steps", type=int, default=65000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--train-unet", action="store_true")
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--condition-channels", type=int, default=4)
    p.add_argument("--no-landmark-condition", action="store_true")
    p.add_argument("--landmark-map-mode", type=str, default="single", choices=["single", "parts"])
    p.add_argument("--landmark-sigma", type=float, default=2.0)
    p.add_argument("--no-landmark-lines", action="store_true")
    p.add_argument("--id-dim", type=int, default=512)
    p.add_argument("--num-id-tokens", type=int, default=4)
    p.add_argument("--random-identity-same-dir", action="store_true")
    p.add_argument("--prompt", default="")
    p.add_argument("--fixed-high-timestep", type=int, default=999)
    p.add_argument("--noise-loss", default="mse", choices=["mse", "l1"])
    p.add_argument("--lambda-noise", type=float, default=1.0)
    p.add_argument("--lambda-recon", type=float, default=10.0)
    p.add_argument("--lambda-id", type=float, default=1.0)
    p.add_argument("--id-loss-start-step", type=int, default=50000)
    p.add_argument("--id-loss-every", type=int, default=1)
    p.add_argument("--id-loss-target", type=str, default="identity", choices=["identity", "clean"])
    p.add_argument("--id-loss-backend", type=str, default="facenet", choices=["facenet", "torchscript"])
    p.add_argument("--facenet-pretrained", type=str, default="vggface2")
    p.add_argument("--id-loss-model", type=str, default="")
    p.add_argument("--id-loss-input-range", type=str, default="minus1_1", choices=["minus1_1", "0_1", "imagenet"])
    p.add_argument("--recon-every", type=int, default=1)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--debug-image-every", type=int, default=100)
    p.add_argument("--debug-max-save", type=int, default=200)
    p.add_argument("--debug-num-samples", type=int, default=1)
    return p

if __name__ == "__main__":
    args = build_parser().parse_args()
    if args.fixed_high_timestep < 0:
        args.fixed_high_timestep = None
    main(args)
