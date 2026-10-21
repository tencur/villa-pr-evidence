# Test cases removed from the PR on 2026-10-21 (maintainer asked for small tests on #1998).
# They need the helpers and imports of the PR's test file; kept here as evidence only.

def test_native_model_pads_input_depth_and_resizes_like_training():
    from vesuvius.ink_detection.config import InkConfig
    from vesuvius.ink_detection.models.model import make_model
    from vesuvius.ink_detection.training.train import (
        prepare_loss_inputs,
        prepare_model_input,
    )

    kernel = [3, 3, 3]
    mapping = {
        "model_type": "vesuvius_unet",
        "mode": "full_3d",
        "patch_size": [3, 8, 8],
        "patch_overlap": 0.25,
        "patch_min_labeled_coverage": 0.0,
        "batch_size": 1,
        "in_channels": 1,
        "model_config": {
            "autoconfigure": True,
            "basic_encoder_block": "BasicBlockD",
            "basic_decoder_block": "ConvBlock",
            "features_per_stage": [2, 4],
            "n_blocks_per_stage": [1, 1],
            "n_conv_per_stage_decoder": [1],
            "kernel_sizes": [kernel, kernel],
            "strides": [[1, 1, 1], [2, 2, 2]],
            "pool_op_kernel_sizes": [[1, 1, 1], [2, 2, 2]],
            "spacing": [1, 1, 1],
            "input_pad_depth_to": 4,
        },
        "targets": {
            "ink": {
                "out_channels": 1,
                "activation": "none",
                "z_projection_mode": "none",
            }
        },
        "datasets": [{"segments_path": "/tmp/ink", "volume_scale": 0}],
    }
    config = InkConfig.from_mapping(mapping)
    torch.manual_seed(0)
    reference = make_model(config).eval()
    payload = {"config": mapping, "model": reference.state_dict()}
    args = native.parse_args(["tifxyz", "memory.pth", "out.zarr", "--no-compile"])
    bundle = native._load_native_model(payload, config, args)
    depths = []
    bundle.model.model.register_forward_pre_hook(
        lambda _module, inputs: depths.append(int(inputs[0].shape[2]))
    )
    image = torch.randn(1, 1, 3, 8, 8, generator=torch.Generator().manual_seed(1))

    with torch.no_grad():
        actual = native.predict_batch(
            bundle.model,
            image.to(bundle.device),
            variants=[()],
            tta_batch_size=None,
            patch_size_zyx=(3, 8, 8),
        ).cpu()
        training_config = SimpleNamespace(
            surface_mask_channel=False, ink=SimpleNamespace(model=config.model)
        )
        model_input = prepare_model_input({"image": image}, training_config)
        logits = reference(model_input)["ink"]
        batch = {
            "image": image,
            "inklabels": torch.zeros_like(image),
            "supervision_mask": torch.ones_like(image),
        }
        expected = prepare_loss_inputs(logits, batch, mode="full_3d")[0].sigmoid()

    # The network sees the padded depth it was trained on, and its output is
    # resized to the patch exactly as training resizes it.
    assert depths == [4]
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=1e-6)
