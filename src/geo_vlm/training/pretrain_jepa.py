"""Train I-JEPA on geospatial imagery."""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from geo_vlm.data.copernicus_jepa import (
    COPERNICUS_CHANNELS,
    CopernicusJEPADataset,
    write_copernicus_manifest,
)
from geo_vlm.data.geotiff_jepa import (
    GeoTiffJEPADataset,
    inspect_rasters,
    write_manifest,
)
from geo_vlm.models.jepa import apply_masks, build_jepa_models, repeat_interleave_batch
from geo_vlm.training.masks import MultiBlockMaskCollator
from geo_vlm.training.schedulers import CosineWDSchedule, WarmupCosineSchedule

LOGGER = logging.getLogger(__name__)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        return torch.device("cuda:0")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _load_checkpoint(path: Path, *, encoder, target_encoder, predictor, optimizer, device: torch.device) -> dict:
    checkpoint = torch.load(path, map_location=device)
    encoder.load_state_dict(checkpoint["encoder"])
    target_encoder.load_state_dict(checkpoint["target_encoder"])
    predictor.load_state_dict(checkpoint["predictor"])
    optimizer.load_state_dict(checkpoint["opt"])
    return checkpoint


def _stream_state_dir(config: dict) -> Path | None:
    state_dir = config.get("checkpoint", {}).get("state_dir")
    if state_dir is None:
        return None
    return Path(state_dir).expanduser()


def _best_metric_path(state_dir: Path) -> Path:
    return state_dir / "best_metric.json"


def _read_best_metric(state_dir: Path) -> float:
    path = _best_metric_path(state_dir)
    if not path.exists():
        return math.inf
    with path.open("r", encoding="utf-8") as handle:
        return float(json.load(handle)["loss"])


def _write_best_metric(state_dir: Path, *, loss: float, epoch: int) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    payload = {"loss": loss, "epoch": epoch}
    _best_metric_path(state_dir).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _make_collator(config: dict) -> MultiBlockMaskCollator:
    return MultiBlockMaskCollator(
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


def _make_loader(config: dict, output_dir: Path) -> tuple[DataLoader, int]:
    source = config["data"].get("source", "geotiff")
    collator = _make_collator(config)

    if source == "geotiff":
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
        loader = DataLoader(
            dataset,
            batch_size=config["data"]["batch_size"],
            shuffle=True,
            num_workers=config["data"]["num_workers"],
            pin_memory=config["data"]["pin_memory"],
            collate_fn=collator,
            drop_last=True,
        )
        return loader, len(loader)

    if source == "copernicus_pretrain":
        if config["model"]["in_channels"] != COPERNICUS_CHANNELS:
            raise ValueError(
                "Copernicus JEPA input requires "
                f"model.in_channels={COPERNICUS_CHANNELS}"
            )
        steps_per_epoch = int(config["optimization"].get("steps_per_epoch", 0))
        if steps_per_epoch <= 0:
            raise ValueError("optimization.steps_per_epoch is required for copernicus_pretrain")

        url_key = config["data"].get("url_key", "100_example")
        dataset = CopernicusJEPADataset(
            crop_size=config["data"]["crop_size"],
            url_key=url_key,
            urls=config["data"].get("urls"),
            shardshuffle=config["data"].get("shardshuffle", False),
            resampled=config["data"].get("resampled", False),
            require_all_modalities=config["data"].get("require_all_modalities", False),
        )
        write_copernicus_manifest(
            output_path=output_dir / "dataset_manifest.json",
            url_key=url_key,
            urls=dataset.urls,
            crop_size=config["data"]["crop_size"],
            require_all_modalities=dataset.require_all_modalities,
        )
        loader = DataLoader(
            dataset,
            batch_size=config["data"]["batch_size"],
            shuffle=False,
            num_workers=config["data"]["num_workers"],
            pin_memory=config["data"]["pin_memory"],
            collate_fn=collator,
            drop_last=True,
        )
        return loader, steps_per_epoch

    raise ValueError(f"Unsupported data.source {source!r}")


def train(config: dict, output_dir: Path) -> None:
    device = get_device()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config_resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )

    loader, iterations_per_epoch = _make_loader(config, output_dir)

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
    state_dir = _stream_state_dir(config)
    resume_path = state_dir / "latest.pt" if state_dir is not None else None
    if resume_path is not None and resume_path.exists():
        checkpoint = _load_checkpoint(
            resume_path,
            encoder=encoder,
            target_encoder=target_encoder,
            predictor=predictor,
            optimizer=optimizer,
            device=device,
        )
        LOGGER.info(
            "resumed checkpoint=%s epoch=%s",
            resume_path,
            checkpoint.get("epoch", "unknown"),
        )
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
    log_every = int(config.get("logging", {}).get("log_every", 0))
    global_step = 0
    final_loss = math.inf
    for epoch in range(config["optimization"]["epochs"]):
        for batch_idx, (images, context_masks, target_masks) in enumerate(loader):
            if batch_idx >= iterations_per_epoch:
                break
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

            final_loss = float(loss.detach().cpu())
            record = {
                "epoch": epoch + 1,
                "step": global_step,
                "loss": final_loss,
                "lr": lr,
                "weight_decay": wd,
                "momentum": momentum,
            }
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            if log_every > 0 and (global_step % log_every == 0):
                LOGGER.info(
                    "epoch=%s step=%s/%s loss=%.6f lr=%.6g",
                    epoch + 1,
                    batch_idx + 1,
                    iterations_per_epoch,
                    record["loss"],
                    lr,
                )
            global_step += 1

        _save_checkpoint(
            output_dir / "latest.pt",
            epoch=epoch + 1,
            encoder=encoder,
            target_encoder=target_encoder,
            predictor=predictor,
            optimizer=optimizer,
        )
        if state_dir is not None:
            _save_checkpoint(
                state_dir / "latest.pt",
                epoch=epoch + 1,
                encoder=encoder,
                target_encoder=target_encoder,
                predictor=predictor,
                optimizer=optimizer,
            )
            best_loss = _read_best_metric(state_dir)
            if final_loss < best_loss:
                _save_checkpoint(
                    state_dir / "best.pt",
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
                _write_best_metric(state_dir, loss=final_loss, epoch=epoch + 1)
        else:
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
