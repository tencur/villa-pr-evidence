# #2721 revision 2026-11-26 (villa-review review of 1d3b563ed): NOT pushed, NOT commented

## Branch
- Worktree (scratch host) `<work>/fixwork/sw5k/K5B/wt-C866`, new local branch `r1126b/napari-trainer-revision`, created
  from `1d3b563ed` (the PR head; base upstream main `659e2fadb`). Four new commits on top; nothing rebased or amended
  (`logs/COMMITS.txt`):
  1. `82c7f72ae10dede81615363e6551f59a62cfb78a` napari_trainer: train on a per-run copy of the config and the paired layers (points 1, 2, 3, 5)
  2. `41775063dbf86f4ba01cd4a9384243cd17319826` models: compute VOI on 2D validation patches (point 4)
  3. `cbea85f479e8373913ef80768500f7a47cdba441` napari_trainer: restore the checkpoint's network and stop on missing weights (point 6)
  4. `9ee250e0d2b8c5ba9dadfe1a2915472b0bfb348c` vesuvius: ship the napari trainer's default config in the wheel (point 7; new head)
  Author Ben <60955077+tencur@users.noreply.github.com>, Co-Authored-By trailer.
- Patch: `revision.patch` (`git format-patch --stdout 1d3b563ed..HEAD`).
- Files vs 1d3b563ed: source + pyproject "5 files changed, 246 insertions(+), 144 deletions(-)" (`main_window.py`,
  `inference.py`, `models/evaluation/voi.py`, `models/preprocessing/patches/generate.py`, `pyproject.toml`); tests
  "3 files changed, 369 insertions(+), 5 deletions(-)". Whole PR vs main: source "5 files changed, 269 insertions(+),
  128 deletions(-)". Two files outside napari_trainer: `voi.py` (point 4) and `generate.py` (point 3, a split, no
  behaviour change for `vesuvius.find_patches`). The sparse worktree set was widened to those two directories.
- To publish (lead): push `r1126b/napari-trainer-revision` onto `tencur:fix/napari-trainer-launch-and-train`
  (fast-forward from 1d3b563ed, no force), then post `REPLY_DRAFT.md`. PR_BODY.md is now out of date (see end).

## Test setup
- CPU, Python 3.13, torch 2.14.1+cpu, napari from the napari venv, vc_reader built locally; source trees by
  `git archive` of 659e2fadb (main), 1d3b563ed (old), each new commit, and 9ee250e0d (new); the new test files were
  copied into the main/old/intermediate trees. Runner `test/run_tests.sh` + `test/env.sh`.
- timm: the venv has timm 1.0.30, with which no Primus network builds (villa's own
  `tests/models/build/test_primus_shapes.py` fails 4 of 6 on main with "The size of tensor a (88) must match the size
  of tensor b (132)"). All runs here put timm 1.0.22 (the version in `vesuvius/uv.lock`) first on the path; with it
  `test_primus_shapes.py` passes on main. This also changes an earlier claim, see "PR_BODY.md" below.
- Synthetic data only (the tests' arrays and the session harness's generated volume).

## Per review point (tests in `vesuvius/tests/napari_trainer/`; first error per failure in `logs/unit_summary.txt`)

1. **Run-local settings.** `run_training` deep-copies the selected config (`mgr = deepcopy(_config_manager)`) and
   every later step (2D/3D ops, `update_config`, export, targets, losses, trainer) uses the copy; the GUI config is
   never written. Test `test_training_uses_a_copy_of_the_selected_config`: default config, train on `vol01_surface`,
   then on `vol01_ink`; the selected config's `convert_to_dict()`, `data_path` and `train_patch_size` are unchanged,
   neither run got the selected object, the second run's ink target still has `ignore_label: 2`. Because the selected
   config is unchanged, inference's weights-only fallback builds from the YAML as selected (not separately tested
   with a checkpoint). Old: FAILED (`assert ({'tr_setup':...3'), (32, 32)) == ({'tr_setup':...28, 128, 128))`);
   new: passed.
2. **Export.**
   - Pairing: a label belongs to the image with the longest matching name (`_pair_layers`). Test
     `test_label_layer_pairs_with_the_longest_matching_image_name` (images `vol`, `vol_crop`, label `vol_crop_ink`):
     one volume, equal to `vol_crop` + label, target `ink`. Old: FAILED (`FileNotFoundError: Label not found:
     .../labels/vol_ink.zarr`).
   - Grids: the label must have the image's shape and the same `scale`, `translate`, `rotate`, `shear` and `affine`
     (ValueError naming both layers). `test_label_layer_on_another_grid_is_rejected[translate|scale|shape]`. Old: 3
     FAILED (`DID NOT RAISE ValueError`).
   - Bypassing configs: `dataset_config.volumes` or a `dataset_type` other than `zarr` (cross_frame) is refused before
     any export. `test_configs_that_read_other_data_are_rejected[volumes|cross_frame]` (the cross_frame case uses a
     config with dummy `file:///nowhere` URLs; the bundled presets were not loaded). Old: FAILED
     (`FileNotFoundError: Explicit image volume not found ...` / `DID NOT RAISE ValueError`).
   - Writes and lifetime: files are named `images/image<i>.zarr` and `labels/image<i>_<target>.zarr` (indexes, not
     layer names), under `<tempdir>/napari_trainer_XXXX/data`, created per run and removed in `finally` (after
     success and after any error, including a failed export or patch selection). Target names must match
     `[A-Za-z0-9][A-Za-z0-9_-]*`. Test `test_exports_stay_in_a_run_directory_removed_after_the_run`: layers named
     `../../escaped` produce nothing outside the run directory, the directory is gone after a run and after a trainer
     that raises; a label `vol01_../../escaped` is refused. Old: FAILED (`ValueError: No .zarr directories found in
     .../images`: the image went outside `images/`).
   - `<image>_mask` is still not used (out of scope, as the review notes).
3. **Minimum-label threshold.** After the export the run writes the patch cache with villa's generator
   (`generate_patch_caches` split into `generate_patch_caches_for_config(mgr, ...)` so it takes the in-memory run
   config); ZarrDataset loads that cache with the same parameters. The widget's percentage is the only criterion
   (see trade-offs); no qualifying patch is a ValueError. 3D layers with Patch Size Z 0 raise ValueError before
   anything is written. Tests `test_min_labeled_percentage_selects_the_patches` (16 labeled pixels in one of four
   32x32 patches: 0% -> 4 patches, 1% -> 1 patch at (0, 0), 10% -> error) and
   `test_3d_layers_with_patch_size_z_0_are_rejected`. Old: FAILED (`assert [4, 4] == [4, 1]`; `DID NOT RAISE`).
4. **2D validation.** The README's napari section has no 2D/3D statement, but its screenshot
   (`docs/images/napari_trainer.png`) shows a 2D image `32` with label `32_ink` and Patch Size Z 0, and the widget
   defaults to Z 0, so a 3D-only scope would contradict the documented example. Chosen fix: dimension-appropriate
   VOI. `_bbox3d` in `voi.py` now handles 2D and 3D masks (the 3D path is unchanged); `label_components` already
   reads connectivity 26 as 8 on 2D masks. No trainer change. Tests: `test_validation_metrics_accept_2d_patches`
   (every metric the trainer installs, with ignore_label 2, updated with 2D logits/labels) and
   `test_2d_voi_equals_voi_of_the_same_patch_as_a_one_slice_volume`. Old: 2 FAILED (`AssertionError` in
   `_bbox3d`). Real app (`train2d`, tiny 2D U-Net, 2 epochs, 128x128 slice, 64x64 patch): new head trained, validated
   and wrote 3 checkpoints; 1d3b563ed raised `AssertionError` at `assert mask.ndim == 3`, no checkpoint
   (`logs/session_new_train2d.log`, `logs/session_old_train2d.log`).
5. **Losses.** The menu offers only `nnUNet_DC_and_CE_loss` (the default). Test
   `test_offered_losses_fit_the_default_target_and_keep_its_ignore_label`: each offered loss, built by
   `BaseTrainer._build_loss` for a target with ignore_label 2, runs on a 2-channel prediction / 1-channel target,
   scores matching logits lower than swapped ones, and does not change when only the ignored region's prediction
   changes. Old: FAILED (`ValueError: Target size (torch.Size([2, 0, 16, 16])) must be the same as input size`).
   Per old choice (`logs/loss_choices.log`): DC_and_CE ok; DC_and_BCE and BCEWithLogits raise; MSE and L1 give the
   same loss for matching and swapped logits; standalone Dice changes when the ignored region changes.
6. **Inference checkpoint restore** (`inference.py`).
   - Per-target decoders: for each target, `task_decoders.<t>.*` keys in the weights -> `separate_decoder: true`,
     `task_heads.<t>.*` -> `false`, set on a copy of the saved config before building.
   - Nested legacy `config`: unwrapped like `vesuvius.predict` (`model_config` + `patch_size`, `batch_size`,
     `in_channels`, `targets`); written inline because importing `vesuvius.models.run.inference` calls
     `multiprocessing.set_start_method('spawn', force=True)` for the whole process.
   - Primus: when the saved config has `input_shape` and no patch size, it is returned as `patch_size`, which the
     network build and the sliding window both use.
   - Weights: `load_state_dict(strict=False)`; any missing network tensor or a shape mismatch raises before
     `run_inference` adds a layer; extra checkpoint tensors are printed and skipped. `module.`/`_orig_mod.` prefixes
     are stripped (this replaces the old compile-before-load branch on CUDA).
   - Tests: `test_per_target_decoder_layout_is_restored` (2D U-Net, ink shared + surface separate; all tensors equal,
     both outputs equal), `test_nested_legacy_config_is_unwrapped`, `test_primus_checkpoint_infers_with_its_input_shape`
     (Primus-S 3D 16^3, 20x24x16 image, one output layer), `test_missing_weights_stop_before_a_prediction`
     (one head tensor removed: RuntimeError, no layer). Old: 4 FAILED (differing tensors; differing tensors; `ValueError:
     No patch_size or train_patch_size found`; `DID NOT RAISE RuntimeError`). Control
     `test_extra_checkpoint_tensors_are_not_rejected` (extra `shared_decoder.seg_layers.0.*`) passes on old and new.
7. **Packaging.** `models/configuration/single_task/ps128_dicece.yaml` added to `[tool.setuptools.package-data]`.
   Test `test_default_config_is_package_data` (old: FAILED `assert False`). Wheel: `pip wheel --no-deps
   --no-build-isolation --ignore-requires-python` (the venv is 3.13, the project asks >=3.14) of the old and new
   trees: old wheel has 1 YAML (`install/configs/cubes.yaml`), new has also `ps128_dicece.yaml`
   (`logs/wheel_contents_*.txt`). `test/startup_check.py` runs `main_window.main()` from the unpacked wheel with
   napari offscreen: old `FileNotFoundError: ... unpacked/vesuvius/models/configuration/single_task/ps128_dicece.yaml`;
   new starts with docks `config file`, `inference`, `training` and the config from the unpacked package
   (`logs/wheel_startup_*.log`).

## Totals (`logs/unit_napari_trainer_*.log`, 24 tests: the 5 existing + 19 new)
| tree | result |
| --- | --- |
| new 9ee250e0d | 24 passed |
| cbea85f47 (commits 1-3) | 1 failed (package data), 23 passed |
| 41775063d (commits 1-2) | 5 failed (4 inference + package data), 19 passed |
| 82c7f72ae (commit 1) | 7 failed (+ 2 VOI), 17 passed |
| old 1d3b563ed | 18 failed, 6 passed (the 5 existing tests + the extra-tensor control) |
| main 659e2fadb | 22 failed, 2 passed (existing: 4 failed, 1 passed, as before; + the extra-tensor control) |

Regression (`logs/regression_*.log`): `tests/models/configuration`, `tests/models/run/test_inference_legacy_checkpoint.py`,
`tests/models/datasets/test_find_valid_patches_plain_array.py`, `tests/models/build/test_primus_shapes.py`: main 62
passed, new 62 passed. No existing tests cover `voi.py` or `generate.py`.

Changes to the existing tests (the 5 still pass on old and new): `test_run_training_trains_on_the_viewer_layers` now
reads the dataset's arrays inside the recording trainer's `train()` (the export is removed when the run ends; read
afterwards the lazy arrays return zeros), and its `volume_id == "vol01"` check became "one volume" (ids are now
indexes, `image0`). The other four are unchanged.

## Real-app sessions (`test/session/`, `logs/session_*.log`, `logs/session_summary.txt`)
Same harness as the 26 Nov rerun plus a `train2d` mode and an optional `SEED` before "Run Inference".
- new, train (tiny 3D U-Net, 32^3, 4 epochs, 706 s): dataset = 1 volume (`image0`), image and ink equal to the
  viewer layers, 32 patches (train and val), "32 patches have at least 10.0% labeled voxels", checkpoints written,
  "Removed the exported layers (/tmp/napari_trainer_29nfe0vc)"; no SIGSEGV this time.
- new, infer in a fresh session with that checkpoint: default config and `tiny_unet_wide_ink.yaml` both 0 of 90
  tensors not loaded, one output layer, **ink IoU 0.9522** (26 Nov rerun: 0.9378, another training run).
- First-commit-only (2333252cb) baseline, alternate config: 89 of 90 tensors not loaded, so the output is from a
  freshly initialised network and its IoU depends on the random draw: earlier 0.0041 (12 Nov), 0.025 and 0.3734
  (26 Nov), 0.0 (villa-review). With `SEED=0` twice: 0.0005 and 0.0005; `SEED=1`: 0.0008. The logged 0.3734 was an
  unrepresentative draw.
- First-commit-only, default config, timm 1.0.22, SEED=0: builds the Primus network, 89 of 90 not loaded, then
  "inference error: PrimusEncoder received a runtime spatial shape that does not match the configured input_shape.
  Expected (128, 128, 128), got (32, 32, 32)", no output layer (as on 12 Nov and as villa-review saw).

## Trade-offs / out of scope
- Patch selection uses only the widget's percentage: `min_bbox_percent` is set to 0.0 (main and 1d3b563ed set 0.90,
  which never took effect because no cache existed). Keeping 0.90 would silently drop most patches of a sparsely
  painted label (the README example is a painted stroke) with no widget to change it. At 0% every patch is used, as
  before. The generator counts voxels equal to the config's `valid_patch_value` (1 in the default config) and does
  not apply the ignore label (inherited from `vesuvius.find_patches`).
- One loss in the menu. Fixing the others would mean shared-loss changes (multi-channel targets for BCE, masking for
  standalone Dice) for no workflow need; the combo box stays so the loss in use is visible.
- Volume ids in the trainer's logs/debug output are `image<i>`; the run prints which layers each id came from.
- The export is removed in `finally`; a process that dies hard (e.g. the SIGSEGV seen after the final checkpoint in 2
  of 3 earlier training sessions) leaves its run directory in the temp dir. Three such directories (11 MB) from the
  earlier 26 Nov sessions of the old code (01:55-02:21Z) are still in the scratch host's /tmp; I did not remove
  them. The old head's `train2d` session here also left its export (`/tmp/napari_trainer_zpds2h13`, the old code never
  removes it); I removed that one.
- Shape check is exact equality, so a multichannel image layer (C, Y, X) with a (Y, X) label is refused with a clear
  message (it was not supported before either: in_channels is 1).
- Inference: legacy checkpoints with an EMA copy still load the `model` weights (`vesuvius.predict` prefers
  `ema_model` when `config.ema.validate`); not changed. A checkpoint whose weights lack a module that is filled
  elsewhere (e.g. a frozen guide backbone, if not saved) would now stop with the missing-weights error; not seen in
  the napari configs, not tested. CUDA path not run.
- The weights-only fallback still needs patch metadata in the GUI model config (inherited, as the review says); zero
  X/Y patch sizes remain an inherited invalid input.
- Not run: GPU, a visible display, Zarr 2 vs 3 variants, real scroll data, the old `_mask` layer.

## For the lead
- PR_BODY.md is out of date after this revision: it says the temporary copy "stays in the system temp directory",
  lists the old loss behaviour and IoU 0.938, and says the first-commit default config "fails while building the
  Primus network (96 vs 144)". That build failure came from the venv's timm 1.0.30; with the lockfile's timm 1.0.22
  it builds, loads 1 of 90 tensors and stops at the first patch (runtime spatial shape error), no output layer.
  Suggested numbers for the body: IoU 0.952 (this run), baseline "a freshly initialised network, IoU about 0".
- Scratch: `<work>/fixwork/c866rev` (138 MB: four source trees, timm 1.0.22, the 3D/2D training work dirs,
  logs); free disk at the end 4.3 GB.
