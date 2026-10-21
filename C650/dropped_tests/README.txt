Cases dropped from the PR's test file to keep one small test (maintainer feedback on #1998, 2026-10-21).
Run 2026-10-21 on rackpi1 (Python 3.13.5, torch 2.14.1+cpu), the pre-trim kit test (2f12dbfd2: a real tiny vesuvius_unet through _load_native_model and predict_batch vs the training path) against upstream main efddd7770: main 1 failed, 13 passed; main + this PR 14 passed.
