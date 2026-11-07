Dropped by P3 (2026-10-21, bruniss test-size rule): the kit's Qt test (186 lines + 23 lines of test CMake + making AVAHI's imported target GLOBAL), which drives the overlay's Apply in a real viewer and must link all of VC3D's objects. Kept here with its CMake diff. On upstream 15db7cd57 (rackpi6) it fails on main and passes with the PR's source change.

2026-10-29 rework on upstream main 1c905c522 (after #2358 moved segments to the object store): the old Qt test builds a
VolumePkg and cannot run there. Replacement: SurfaceRotationApplyTest_1c905c522.inc, a VC3D self-test in upstream's own
style (VC3D --test-surface-rotation, ctest surface_rotation_apply), wired by selftest_wiring_1c905c522.diff (the .inc,
its #include under VC_TESTING, VC_TESTING for that source file, the VCAppMain flag, the ctest entry). It opens a stored
surface through the surface panel, saves through SegmentationModule::saveSurface with CWindow's store-pin resolver and
reopen-on-commit rule, and clicks Flip + Apply in the overlay of a real viewer twice without selecting the segment again.
Main fails "a second flip starts from the saved geometry: two flips give the original back"; the PR passes
(logs: ../rebase_1c905c522/).
