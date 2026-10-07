# villa PR evidence

Screenshots and reproduction scripts for my pull requests to [ScrollPrize/villa](https://github.com/ScrollPrize/villa).
Each folder is named after one PR and holds:

- `screenshot_before.png` and `screenshot_after.png`: terminal output of `run_proof.sh <ID> main` (upstream main `e0bbb8b`) and `run_proof.sh <ID> fix` (the PR branch), on the same data and settings.
- `scripts/`: the reproduction scripts that `run_proof.sh` calls.
- `rebased_rerun.txt`: the tests and proof rerun after rebasing onto `e0bbb8b`.

`FINAL_PROOFS.txt` is the complete output of the final run for every PR, on main and on the fix.

## Data

All inputs are public Vesuvius Challenge open data (PHerc0500P2 and PHerc0139 segments and labels, and a Scroll 5 / PHerc172 segment layer),
read from `s3://vesuvius-challenge-open-data/` or https://dl.ash2txt.org/. `run_proof.sh` shows the exact paths used. Its
directory layout is the machine I ran it on (Raspberry Pi 5, CPU only, Python 3.14, torch 2.12 CPU); change the variables at the top to run it elsewhere.

| Folder | PR |
| --- | --- |
| [C47](C47/) | optimized_inference: do not re-apply the layer window to the zarr the prepare step cropped |
| [C03](C03/) | optimized_inference: prepare keeps the signal of 16-bit layer TIFFs |
| [C04](C04/) | optimized_inference: keep each S3 surface volume's disk cache separate |
| [C12](C12/) | ink_detection: labels made with create_label_zarrs are now seen by flat training on 21- and 28-slice volumes |
| [C64](C64/) | patch grids: add an end-aligned patch so the tail of every axis is trained |
| [C63](C63/) | vesuvius.train: make --seed reproduce a run |
| [C38](C38/) | vesuvius.train: reject binary-loss labels outside [0, 1] (0/255 masks); fix the data docs |
| [C31](C31/) | find_patches/train: honour ignore_label and bg_sampling_enabled |
| [C49](C49/) | vesuvius.predict: store logits for train.py checkpoints with an output activation |
| [C52](C52/) | vesuvius.train: validate on logits for targets with an output activation |
| [C44](C44/) | vesuvius.compute_st: integrate the structure tensor by default |
