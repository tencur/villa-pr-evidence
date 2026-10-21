Cases dropped from the PR's test file to keep one small test (maintainer feedback on #1998, 2026-10-21).
Run 2026-10-21 on rackpi1 (Python 3.13.5, torch 2.14.1+cpu), the pre-trim kit test file (b043f0f68) against upstream main efddd7770: main 3 failed (these two plus the case kept in the PR), 44 passed; main + this PR 47 passed.
