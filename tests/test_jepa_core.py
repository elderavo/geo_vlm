import torch

from geo_vlm.models.jepa import apply_masks, build_jepa_models, repeat_interleave_batch
from geo_vlm.training.masks import MultiBlockMaskCollator


def test_multiblock_masks_have_expected_counts() -> None:
    collator = MultiBlockMaskCollator(
        input_size=224,
        patch_size=16,
        enc_mask_scale=(0.85, 1.0),
        pred_mask_scale=(0.15, 0.2),
        aspect_ratio=(0.75, 1.5),
        num_enc_masks=1,
        num_pred_masks=4,
        min_keep=10,
        allow_overlap=False,
    )
    batch = [torch.zeros(3, 224, 224) for _ in range(2)]

    images, context_masks, target_masks = collator(batch)

    assert images.shape == (2, 3, 224, 224)
    assert len(context_masks) == 1
    assert len(target_masks) == 4
    assert context_masks[0].shape[0] == 2
    assert target_masks[0].shape[0] == 2


def test_jepa_forward_and_ema_update_are_finite() -> None:
    encoder, target_encoder, predictor = build_jepa_models(
        model_name="vit_tiny",
        img_size=224,
        patch_size=16,
        in_channels=3,
        predictor_depth=2,
        predictor_embed_dim=96,
    )
    collator = MultiBlockMaskCollator(
        input_size=224,
        patch_size=16,
        enc_mask_scale=(0.85, 1.0),
        pred_mask_scale=(0.15, 0.2),
        aspect_ratio=(0.75, 1.5),
        num_enc_masks=1,
        num_pred_masks=4,
        min_keep=10,
        allow_overlap=False,
    )
    images, context_masks, target_masks = collator([torch.rand(3, 224, 224) for _ in range(2)])

    with torch.no_grad():
        targets = target_encoder(images)
        targets = torch.nn.functional.layer_norm(targets, (targets.size(-1),))
        targets = apply_masks(targets, target_masks)
        targets = repeat_interleave_batch(targets, batch_size=2, repeat=len(context_masks))

    preds = predictor(encoder(images, context_masks), context_masks, target_masks)
    loss = torch.nn.functional.smooth_l1_loss(preds, targets)
    assert torch.isfinite(loss)
    assert preds.shape == targets.shape

    before = [param.detach().clone() for param in target_encoder.parameters()]
    with torch.no_grad():
        for source, target in zip(encoder.parameters(), target_encoder.parameters()):
            target.data.mul_(0.996).add_(0.004 * source.detach().data)
    after = list(target_encoder.parameters())
    assert any(not torch.equal(old, new) for old, new in zip(before, after))
