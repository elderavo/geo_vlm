"""Train I-JEPA on geospatial GeoTIFF imagery."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from geo_vlm.data.geotiff_jepa import (
    GeoTiffJEPADataset,
    inspect_rasters,
    write_manifest,
)
from geo_vlm.models.jepa import apply_masks, build_jepa_models, repeat_interleave_batch
from geo_vlm.training.masks import MultiBlockMaskCollator
from geo_vlm.training.schedulers import CosineWDSchedule, WarmupCosineSchedule

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def _make_optimizer(encoder, predictor, weight_decay: float):
    encoder_decay, encoder_no_decay = [], []
    predictor_decay, predictor_no_decay = [], []
    for name, parameter in encoder.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim == 1 or name.endswith("bias"):
            encoder_no_decay.append(parameter)
        else:
            encoder_decay.append(parameter)
    for name, parameter in predictor.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim == 1 or name.endswith("bias"):
            predictor_no_decay.append(parameter)
        else:
            predictor_decay.append(parameter)
    return torch.optim.AdamW(
        [
            {"params": encoder_decay, "weight_decay": weight_decay},
            {"params": predictor_decay, "weight_decay": weight_decay},
            {"params": encoder_no_decay, "weight_decay": 0.0, "WD_exclude": True},
            {"params": predictor_no_decay, "weight_decay": 0.0, "WD_exclude": True},
        ]
    )


def _save_checkpoint(path: Path, *, epoch: int, encoder, target_encoder, predictor, optimizer) -> None:
    torch.save(
        {
            "epoch": epoch,
            "encoder": encoder.state_dict(),
            "target_encoder": target_encoder.state_dict(),
            "predictor": predictor.state_dict(),
            "opt": optimizer.state_dict(),
        },
        path,
    )


def train(config: dict, output_dir: Path) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config_resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )

    dataset = GeoTiffJEPADataset(
        root=Path(config["data"]["root_path"]),
        image_product=config["data"]["image_product"],
        crop_size=config["data"]["crop_size"],
        crop_scale=tuple(config["data"]["crop_scale"]),
        expected_channels=config["model"]["in_channels"],
    )
    metadata = inspect_rasters(dataset.files, max_files=min(5, len(dataset.files)))
    write_manifest(
        output_path=output_dir / "dataset_manifest.json",
        root=Path(config["data"]["root_path"]),
        image_product=config["data"]["image_product"],
        files=dataset.files,
        metadata=metadata,
    )

    collator = MultiBlockMaskCollator(
        input_size=config["data"]["crop_size"],
        patch_size=config["model"]["patch_size"],
        enc_mask_scale=tuple(config["mask"]["enc_mask_scale"]),
        pred_mask_scale=tuple(config["mask"]["pred_mask_scale"]),
        aspect_ratio=tuple(config["mask"]["aspect_ratio"]),
        num_enc_masks=config["mask"]["num_enc_masks"],
        num_pred_masks=config["mask"]["num_pred_masks"],
        min_keep=config["mask"]["min_keep"],
        allow_overlap=config["mask"]["allow_overlap"],
    )
    loader = DataLoader(
        dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=True,
        num_workers=config["data"]["num_workers"],
        pin_memory=config["data"]["pin_memory"],
        collate_fn=collator,
        drop_last=True,
    )

    encoder, target_encoder, predictor = build_jepa_models(
        model_name=config["model"]["model_name"],
        img_size=config["data"]["crop_size"],
        patch_size=config["model"]["patch_size"],
        in_channels=config["model"]["in_channels"],
        predictor_depth=config["model"]["predictor_depth"],
        predictor_embed_dim=config["model"]["predictor_embed_dim"],
    )
    encoder.to(device)
    target_encoder.to(device)
    predictor.to(device)

    optimizer = _make_optimizer(
        encoder,
        predictor,
        weight_decay=float(config["optimization"]["weight_decay"]),
    )
    iterations_per_epoch = len(loader)
    total_steps = max(1, iterations_per_epoch * config["optimization"]["epochs"])
    lr_schedule = WarmupCosineSchedule(
        optimizer,
        warmup_steps=config["optimization"]["warmup"] * iterations_per_epoch,
        start_lr=float(config["optimization"]["start_lr"]),
        ref_lr=float(config["optimization"]["lr"]),
        t_max=int(total_steps * config["optimization"]["ipe_scale"]),
        final_lr=float(config["optimization"]["final_lr"]),
    )
    wd_schedule = CosineWDSchedule(
        optimizer,
        ref_wd=float(config["optimization"]["weight_decay"]),
        final_wd=float(config["optimization"]["final_weight_decay"]),
        t_max=int(total_steps * config["optimization"]["ipe_scale"]),
    )
    ema_start, ema_end = config["optimization"]["ema"]

    log_path = output_dir / f'{config["logging"]["write_tag"]}.jsonl'
    global_step = 0
    for epoch in range(config["optimization"]["epochs"]):
        for images, context_masks, target_masks in loader:
            images = images.to(device)
            context_masks = [mask.to(device) for mask in context_masks]
            target_masks = [mask.to(device) for mask in target_masks]

            lr = lr_schedule.step()
            wd = wd_schedule.step()

            with torch.no_grad():
                target_latents = target_encoder(images)
                target_latents = F.layer_norm(target_latents, (target_latents.size(-1),))
                target_latents = apply_masks(target_latents, target_masks)
                target_latents = repeat_interleave_batch(
                    target_latents,
                    batch_size=len(images),
                    repeat=len(context_masks),
                )

            context_latents = encoder(images, context_masks)
            predictions = predictor(context_latents, context_masks, target_masks)
            loss = F.smooth_l1_loss(predictions, target_latents)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            progress = global_step / max(1, total_steps - 1)
            momentum = ema_start + progress * (ema_end - ema_start)
            with torch.no_grad():
                for source, target in zip(encoder.parameters(), target_encoder.parameters()):
                    target.data.mul_(momentum).add_((1.0 - momentum) * source.detach().data)

            record = {
                "epoch": epoch + 1,
                "step": global_step,
                "loss": float(loss.detach().cpu()),
                "lr": lr,
                "weight_decay": wd,
                "momentum": momentum,
            }
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            global_step += 1

        _save_checkpoint(
            output_dir / "latest.pt",
            epoch=epoch + 1,
            encoder=encoder,
            target_encoder=target_encoder,
            predictor=predictor,
            optimizer=optimizer,
        )
        _save_checkpoint(
            output_dir / "best.ckpt",
            epoch=epoch + 1,
            encoder=encoder,
            target_encoder=target_encoder,
            predictor=predictor,
            optimizer=optimizer,
        )


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output_dir = args.output_dir or Path(config["logging"]["folder"])
    train(config, output_dir)


if __name__ == "__main__":
    main()
