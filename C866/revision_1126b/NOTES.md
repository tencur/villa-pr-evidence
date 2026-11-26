# #2721 second revision 2026-11-26 (villa-review re-review of 9ee250e0d, 05:32Z): NOT pushed, NOT commented

## Branch
- Worktree (scratch host) `<work>/fixwork/sw5k/K5B/wt-C866`, new local branch `r1126c/napari-trainer-revision2`, created
  from `9ee250e0d` (the live PR head). Four new commits on top; nothing rebased or amended (`logs/COMMITS.txt`):
  1. `524f0ed1491c323d1b11059c3ff1d220f75dbd4a` napari_trainer: stop before saving when the split leaves no training patch (point 1)
  2. `3c209902f37a0bc416c1fe4528f7df659b458359` napari_trainer: select patches without ignored voxels and train on exactly those (point 2)
  3. `6198b3dc5a857a83242398b97d1f55f7dc308bd5` napari_trainer: keep configured losses and refuse losses that cannot take the labels (point 3)
  4. `56b28fcb8efb9f31fce9805ed2a6b48c84a7effd` napari_trainer: infer with the checkpoint's preprocessing and on the source layer's grid (point 4; new head)
  Author Ben <60955077+tencur@users.noreply.github.com>, Co-Authored-By trailer.
- Patch: `revision.patch` (`git format-patch --stdout 9ee250e0d..HEAD`).
- Files vs 9ee250e0d: source "3 files changed, 288 insertions(+), 38 deletions(-)" (`main_window.py`, `inference.py`,
  and `models/preprocessing/patches/generate.py` restored to main), tests "2 files changed, 337 insertions(+),
  3 deletions(-)". Per commit: 90/1, 231/35, 115/7, 197/6 lines (+/-).
- Whole PR vs main after this revision: source + pyproject "4 files changed, 525 insertions(+), 134 deletions(-)":
  `napari_trainer/main_window.py`, `napari_trainer/inference.py`, `models/evaluation/voi.py` (2D VOI, first revision),
  `pyproject.toml` (package data, first revision). No new shared change: all four repairs are in `napari_trainer`.
  `generate.py` is back to main's version (the first revision's `generate_patch_caches_for_config` split is no longer
  used, because the widget now selects patches itself, see point 2), so the PR's shared footprint shrank by one file.
- To publish (lead): push `r1126c/napari-trainer-revision2` onto `tencur:fix/napari-trainer-launch-and-train`
  (fast-forward from 9ee250e0d, no force), then post `REPLY_DRAFT.md`.

## Test setup
- As in the first revision: CPU, Python 3.13, the napari venv with torch via `site.addsitedir` (not PYTHONPATH),
  vc_reader built locally, timm 1.0.22 (as in `vesuvius/uv.lock`) first on the path. Runners `test/run_tests.sh`,
  `test/env.sh`, `test/run_units.sh` (per-commit + regression), `test/session/run_rev2.sh` (real app).
- Source trees by `git archive` of 659e2fadb (main), 9ee250e0d, each new commit; the final test files (42 tests:
  the 24 of 9ee250e0d + 18 new) copied into every tree. Synthetic data only.
- First error of every failure: `logs/first_errors.txt`; counts: `logs/unit_summary.txt`.

## Per review point (tests in `vesuvius/tests/napari_trainer/`)

1. **Empty training split.** `run_training` now uses `_viewer_layer_trainer(BaseTrainer, selected)`, a subclass
   whose `_configure_dataloaders` calls the trainer's own split and raises `ValueError` ("... puts all N selected
   patches into validation, so there is nothing to train on; nothing was saved ...") when it returns no training
   index. That runs before `_initialize_training` creates the model/checkpoint directories, so nothing is written.
   I put the guard on the trainer's actual split rather than re-deriving `floor(tr_val_split * n)` in the widget.
   Test `test_a_split_without_training_patches_stops_before_saving` (real BaseTrainer, a tiny 2D U-Net config; 16
   labeled pixels in one 32x32 patch at 1% -> 1 patch; a file `checkpoints/tiny2d/tiny2d_final.pth` with known bytes
   exists beforehand): new head raises, the file is byte-identical and the `checkpoints` tree unchanged.
   9ee250e0d: FAILED `DID NOT RAISE ValueError`; its log shows "1 patches have at least 1.0% labeled voxels",
   "Checkpoint saved to: checkpoints/tiny2d_.../tiny2d_epoch1.pth" and "Final model saved to
   checkpoints/tiny2d/tiny2d_final.pth" (the earlier file overwritten).
2. **Patch selection** (all checks before any export unless stated).
   - (a) Ignored voxels: the widget now selects patches itself with villa's `find_valid_patches`, passing the
     selecting target's ignore label (`ignore_labels`, which zeroes it before counting) and the run's
     `valid_patch_value`. Test `test_ignored_voxels_do_not_count_toward_the_labeled_percentage` (the guided preset
     `ps128_guided_dinovol_ink.yaml` itself: ignore_label 2, no valid_patch_value): a label of only 2 -> ValueError
     "No 32x32 patch ... (ignore label 2 not counted)"; with 16 pixels of 1 in patch (0, 0) and 2 elsewhere ->
     exactly [("image0", (0, 0))]. 9ee250e0d: FAILED `DID NOT RAISE ValueError` ("4 patches have at least 1.0%
     labeled voxels" from labels made only of 2).
   - (b) Exact consumption: the selection is written as the patch cache under the parameters `ZarrDataset`
     looks it up with (same `build_cache_params` fields and defaults as `_try_load_cache`), and the trainer
     subclass checks in `_build_dataset_for_mgr` that the training and validation datasets list exactly the
     selected (volume, position) pairs, else RuntimeError before training. Targets with different
     `valid_patch_value` (incl. null on one, 2 on another) are rejected before export, so writer and reader
     cannot key the cache differently.
     Tests: `test_training_reads_exactly_the_selected_patches` (ink + surface targets on two images: dataset list ==
     cache list == the 5 expected patches; a control, it also passes on 9ee250e0d, whose generator agrees in this
     case); `test_targets_with_different_valid_patch_values_are_rejected` (the reviewer's null/2 case; 9ee250e0d:
     FAILED `DID NOT RAISE ValueError`, log "No patch cache found. Enumerating all patches without validation");
     `test_the_trainer_stops_if_its_dataset_is_not_the_selection` (real BaseTrainer, `_try_load_cache` forced to miss;
     new: RuntimeError "... has 4 patches, not the 2 selected ...", checkpoints unchanged; 9ee250e0d: FAILED
     `DID NOT RAISE RuntimeError`, it trained on all patches and saved checkpoints).
   - (c) Missing labels: every paired image must have a label layer for every target of the run, else ValueError
     naming the image and the missing target(s); `unlabeled_foreground_enabled` configs are rejected (they add
     image-only patches nobody selected). Chosen: reject, not partial supervision. Tests
     `test_images_without_a_label_for_every_target_are_rejected[True|False]` (allow_unlabeled_data true/false;
     vol01_ink + vol02_surface) and `test_unlabeled_foreground_configs_are_rejected`. 9ee250e0d: [True] FAILED
     `DID NOT RAISE ValueError` (it trained with a zero-filled surface target); [False] FAILED `FileNotFoundError:
     Label not found: .../labels/image0_surface.zarr` raised mid-run after export; unlabeled_foreground: FAILED
     `DID NOT RAISE ValueError`.
3. **Losses.** `_run_targets` copies each target's config settings; a target with configured `losses` keeps them,
   the widget's loss fills only targets without losses (printed per target). `_check_losses_fit_the_labels`
   rejects, before export: `nnUNet_DC_and_CE_loss` with out_channels < 2 or label values outside 0..out_channels-1;
   `BCEWithLogitsLoss` with out_channels != 1 or labels other than 0/1 (the ignore label excepted in both). Other
   configured losses are kept without a check. Tests: `test_configured_losses_are_kept_and_train` (real BaseTrainer,
   tiny 2D U-Net, the documented ink DC+CE 2-channel + damage BCEWithLogits 1-channel targets, damage label with
   foreground 1): trains with batches whose damage max is 1.0, final checkpoint has ink DC+CE, damage
   BCEWithLogits, damage out_channels 1. 9ee250e0d: FAILED `IndexError: Target 1 is out of bounds.`
   `test_losses_that_cannot_take_the_labels_are_rejected[widget-loss-on-one-channel|class-out-of-range|bce-non-binary]`:
   9ee250e0d: 3 FAILED `DID NOT RAISE ValueError`.
4. **Inference** (`inference.py`).
   - (a) Preprocessing: `ModelLoader` records the checkpoint's top-level `normalization_scheme` and
     `intensity_properties` (what `save_checkpoint` stores from the training dataset). When present, the normalizer
     is built by ZarrDataset's own `_initialize_normalization` (on an instance made with `ZarrDataset.__new__`, only
     `mgr.skip_intensity_sampling`, the scheme and the properties set) and applied to each window's part inside
     the image, which is then zero-padded, as `ZarrDataset.__getitem__` normalizes a patch and pads it. Without a
     saved scheme the old whole-image `A.Normalize("minmax", eps=1e-5)` path runs unchanged. Tests:
     `test_zscore_checkpoint_sees_zscored_windows` (20x48 image, 32x32 patch: window mean < 1e-4, std within 1e-3 of
     1, equal to (x - mean)/std of the crop, padded rows 0; 9ee250e0d: FAILED, mean 0.5513);
     `test_ct_checkpoint_sees_its_saved_intensity_properties` (window == (clip(x, p0.5, p99.5) - mean)/std;
     9ee250e0d: FAILED); control `test_checkpoint_without_saved_preprocessing_keeps_whole_image_minmax` (first
     window bit-identical to the old whole-image min-max; passes on 9ee250e0d and new).
   - (b) Placement: each prediction layer gets `scale`, `translate`, `rotate`, `shear`, `affine` from the source
     layer, expanded with identity on the leading batch/channel axes (napari `CompositeAffine(...).expand_dims`,
     `layer.affine.expand_dims`). Non-napari layer objects (plain objects in API use) get no transform, as before.
     Test `test_prediction_layers_keep_the_source_layers_transform` (napari Image 40x48, scale (2, 0.5), translate
     (10, -4), rotate 30, shear 0.2, affine with shear/offset): `data_to_world((0, c, y, x))` of the prediction
     equals `(0, c) + source.data_to_world((y, x))` at three points for both channels. 9ee250e0d: FAILED (world
     coordinates 0/0 vs 12.6/-6).
   - (c) Contiguity: the batch is `.contiguous()` before the model (per-window batches are built from fresh
     arrays). Test `test_the_network_gets_contiguous_windows[no-saved-preprocessing|zscore]` (forward pre-hook,
     strided source array, batch 3). These pass on 9ee250e0d as well: its `torch.cat` already copied the windows; the
     test guards the new per-window path, not a reproduced failure.

## Totals (`logs/unit_napari_trainer_*.log`, 42 tests: 24 of 9ee250e0d + 18 new)
| tree | result |
| --- | --- |
| new 56b28fcb8 | 42 passed |
| 6198b3dc5 (commits 1-3) | 3 failed (4a zscore, 4a ct, 4b), 39 passed |
| 3c209902f (commits 1-2) | 7 failed (+ 4 loss tests), 35 passed |
| 524f0ed14 (commit 1) | 13 failed (+ 6 patch-selection tests), 29 passed |
| 9ee250e0d | 14 failed, 28 passed (the 24 existing + 4 controls: exact-selection multi-target, min-max, 2 contiguity) |
| main 659e2fadb | 37 failed, 5 passed |

Regression (`logs/regression_*.log`): `tests/models/configuration`, `tests/models/run/test_inference_legacy_checkpoint.py`,
`tests/models/datasets/test_find_valid_patches_plain_array.py`, `tests/models/build/test_primus_shapes.py`: new head
62 passed, main 62 passed (rerun now; 9ee250e0d 62 passed in the first revision, not rerun).

Changes to existing test code: the `recorded` fixture also reads the patch cache the run wrote (`selected`), and
`_default_manager()` takes a config name (default unchanged). The 24 existing tests are otherwise unchanged and pass.

## Real-app session (`test/session/run_rev2.sh`, same `napari_session.py` as the first revision; `logs/session_*`)
- new, train (tiny 3D U-Net, 32^3, 4 epochs, 714 s): one volume `image0` equal to the viewer layers, "32 patches have
  at least 10.0% voxels labeled for target 'ink'", "Split: 28 FG + 0 BG patches for training, 4 FG patches for
  validation", ink keeps its configured DC+CE, checkpoints written, export removed.
- new, infer in a fresh session with that checkpoint ("Normalizing each window with the checkpoint's 'zscore'
  scheme"): default config and `tiny_unet_wide_ink.yaml` both 0 of 90 tensors not loaded, one layer
  (1, 2, 64, 128, 128), **ink IoU 0.9233** (both).
- Same checkpoint inferred with 9ee250e0d (whole-image min-max input): 0 of 90 not loaded, IoU 0.9184.
- First revision 0.9522 and villa-review 0.9766 came from other training runs; training is not deterministic, so the
  IoU of this run is not a measure of the preprocessing change alone (the same-checkpoint pair above is).
- new, train2d (tiny 2D U-Net, 64x64, 2 epochs, 14 s): 4 patches, split 3/1, 3 checkpoints written.
- No export directory left behind by these runs (three older `napari_trainer_*` dirs from 01:55-02:21Z, the old code's
  sessions before the first revision, remain in the scratch host's /tmp as noted then).

## Trade-offs / out of scope
- Selection follows `vesuvius.find_patches`: the first target's labels decide which patches qualify; other targets'
  labels do not. One `valid_patch_value` for all targets (mixed values are rejected, not supported).
- Partially labeled data (an image without some target's label, `allow_unlabeled_data`, `unlabeled_foreground_enabled`)
  is rejected rather than trained with that target excluded from the loss.
- Loss check covers `nnUNet_DC_and_CE_loss` and `BCEWithLogitsLoss` only (the documented ones and the widget's);
  other configured losses train as configured. The check reads every label layer once more (`np.unique`).
- The widget's loss menu still has one entry; it now only fills targets without configured losses.
- Normalization follows the trainer's ZarrDataset transform (per-patch z-score for `zscore`). `vesuvius.predict`
  maps `zscore` with saved mean/std to a global z-score; napari-trained checkpoints save empty intensity properties
  by default (sampling skipped), so both give per-patch z-score for them. The normalizer is built through
  ZarrDataset's private `_initialize_normalization`; the zscore/ct tests pin the result. Legacy nested
  `config.image_normalization` is not read (such checkpoints keep min-max, as before).
- The transform test is 2D; the 3D session uses identity transforms. `units`/`axis_labels` are not copied.
- Not run: GPU/CUDA, a visible display, real scroll data, Zarr 2 vs 3 variants.

## Scratch
- `<work>/fixwork/c866rev` (166 MB after cleanup: source trees of main, 2333252cb, 1d3b563ed, 9ee250e0d and the new
  head, timm 1.0.22, logs); intermediate-commit trees and session work dirs removed. The `tree_main` and `tree_new`
  test directories now hold the 42-test files. Free disk at the end: 4.3 GB.
- The default config sets no `skip_intensity_sampling`; ZarrDataset then skips sampling, so its
  `intensity_properties` (and the checkpoint's) are the config's, `{}` for the shipped napari configs.
