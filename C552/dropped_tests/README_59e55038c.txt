2026-11-01 (H5 lane E, upstream main 59e55038c): SurfaceRotationApplyTest_59e55038c.inc extends the 1c905c522 self-test.
It also wires CWindow::onSurfaceWillBeDeleted (replacing the shown surface clears CState's active surface when it was the
active one): this is what happens in the real app, and why on main the second Rotate shows no tool at all. It checks that
after Apply the segment is still the active surface (pointer and id, MenuActionController::surfaceFromSelection's test),
and that an Apply with no change (Flip ticked and unticked) keeps it selected and Rotate opens again.
Wiring: selftest_wiring_cmake_main.diff (CMake VC_TESTING flag for the overlay source, VCAppMain --test-surface-rotation,
the ctest entry) + append to SurfaceRotationOverlayController.cpp:
  #ifdef VC_TESTING
  #include "SurfaceRotationApplyTest.inc"
  #endif
and copy the .inc to apps/VC3D/overlays/SurfaceRotationApplyTest.inc. Logs: ../selftest_59e55038c/.
