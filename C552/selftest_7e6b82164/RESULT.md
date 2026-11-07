# C552P: C552 self-test ported to upstream main 7e6b82164 (2026-11-07)

Kit: release/228-C552-vc3d-rotate-apply-updates-segment (VC3D Rotate/Flip Apply: the segment stays selected and Rotate opens again).
Proof dir: rackpi6 /home/pi/scrollprize_work/vcbuild/c552n (src = 7e6b82164 + self-test wiring; fix = SurfaceRotationOverlayController.{cpp,hpp}, variants/h5e.*; main = variants/main.*).

## Result
The defect still reproduces on 7e6b82164, and the fix still passes. Run `./run.sh main:test_ext h5e:test_ext main:test_orig` (03:54:08Z to 03:58:11Z; VC3D built under the flock lock; ctest `surface_rotation_apply`):

| run | build | ctest | failing check |
|---|---|---|---|
| main:test_ext | ninja exit 0 | **Failed** (exit 8) | `Actions > Transforms > Rotate opens the tool (Flip and Apply shown)` |
| h5e:test_ext | ninja exit 0 | **Passed** 1/1 | none |
| main:test_orig | ninja exit 0 | **Failed** (exit 8) | `a second flip starts from the saved geometry: two flips give the original back` |

These match the 59e55038c results in the kit's tests_current.txt: main extended fails on the Rotate tool check, -h5e passes, and main with the kit's original test fails on the second-flip check.

Which Rotate fails in main:test_ext: `flipAndApply` uses the same message for both Rotate calls. The failure is on the second one, for three reasons:
- In main:test_orig, the first Apply passes and the failure comes at the second flip's stored-geometry check.
- Before the first Rotate, test_ext adds only the onSurfaceWillBeDeleted wiring to test_orig's steps.
- Qt's offscreen `propagateSizeHints()` notice appears once in the main:test_ext log, compared with twice in the main:test_orig log (where the tool opens both times). This last point is inferred.

ctest excerpts (logs/ctest_*.log):
```
main:test_ext   1/1 Test #166: surface_rotation_apply ...***Failed 0.39 sec
                surface rotation apply test: Actions > Transforms > Rotate opens the tool (Flip and Apply shown)
h5e:test_ext    1/1 Test #166: surface_rotation_apply ...   Passed 0.35 sec
                100% tests passed, 0 tests failed out of 1
main:test_orig  1/1 Test #166: surface_rotation_apply ...***Failed 0.38 sec
                surface rotation apply test: a second flip starts from the saved geometry: two flips give the original back
```

## API mapping used (59e55038c -> 7e6b82164, #2462)
1. **Creating the stored quad surface.** This follows SurfacePanelStoreTest.inc's viewer fixture and `add()` on 7e6b82164.
   - Old: `state.objectStoreSession()->act(attachment, {{uuid, nullopt}}, [&](db, txn){ create_quad_in_store(db, txn, uuid, grid, "segment", {}, volumeCoordinates(volume->id(), 1), {"surface-rotation-test"}); })`
   - New: `auto request = session->newRequest(attachment); request.expectedObjects = {{uuid, nullopt}}; request.operations = createSurfaceOperations(uuid, "segment", {}, volumeCoordinates(volume->id(), 1), {"surface-rotation-test"}, grid.meta, quad_surface_geometry(grid)); session->submit(attachment, request);`
   - `create_quad_in_store` no longer exists. `createSurfaceOperations` is in vc_object_store/objects/SurfaceCreate.hpp and `quad_surface_geometry` is in vc/core/util/QuadSurface.hpp.
2. **Reading the head and observing commits.** The test's copy of CWindow's `SurfaceCatalog::surfacesCommitted` handler now matches CWindow.cpp:2730 on 7e6b82164.
   - Old: `catalog.session()->read(row->source, ...db.currentObjectRevision(txn, id)->uuid...)`, compared with `known->objectRevision`
   - New: `const auto head = catalog.session()->currentObject(row->source, active->id); if (!head || (known && known->version == head->version && known->generation == head->generation)) return;`
   - `PinnedObject::objectRevision` no longer exists. A pin is now `{attachment, object, generation, version, geometryRevision, writeGeneration}`.
   - I also mirrored CWindow's new `if (editingEnabled) return;`. CWindow shows a status-bar message there; the test just returns. It has no effect in this test, because the SegmentationModule is constructed with editing disabled.
3. **Unchanged.** These were already ported or still match CWindow: TestVolume.hpp `VolumeSpec`/`createVolume`, project creation (`ProjectDocument::create` + `VolumePkg::open`, the same as upstream's `openStoreAsProject`), the onSurfaceActivated and onSurfaceWillBeDeleted wiring, `setStorePinResolver` (`StoreOrigin{opened->pin, opened->view}`, the same as CWindow.cpp:2973) and `setEditingDestinationResolver`.

Every check and its message text is unchanged. A script compared all string literals before and after the port, excluding the unchanged "segment" and "surface-rotation-test" arguments, and found them identical. Diffs: variants/port_test_ext.diff and variants/port_test_orig.diff, 2 hunks each, the same in both files.

## Files
- rackpi6 c552n/variants/test_ext.inc and test_orig.inc: ported (sha256 3bfb125d… and e1e15e41…). Backups: test_ext.inc.bak-p3 and test_orig.inc.bak-p3.
- c552n/run.log: this run. The previous run.log was moved to run.log.3.
- Copies here: variants/ (ported files, backups, diffs) and logs/ (run.log, run.log.3, ctest_*, build_*).

## Run-log hazard (run.log.3, for the main session)
Two run.sh processes ran in c552n at the same time:
- an orphan `./run.sh main:test_ext h5e:test_ext main:test_orig`, PID 3402495, started 02:35Z
- the main session's `./run.sh h5e:stub`, started 03:43Z

run.sh swaps the source files before it waits on the lock, so each run built the other's files. As a result, run.log.3's lines are mislabelled:
- "h5e:test_ext ctest exit 0" was h5e built with stub.inc, so it proves nothing.
- "h5e:stub … failed: a second flip…" was main built with my ported test_orig.inc.

Both processes had exited before my run, and I did not kill anything. The results table above comes from a single run.sh run alone in the dir.

Cost: about 35 minutes of wall time, including about 10 minutes waiting on the lock and other runs and about 4 minutes of build and test (ccache warm).
