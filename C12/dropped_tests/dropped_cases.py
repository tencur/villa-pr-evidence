# Test cases removed from the PR on 2026-10-21 (maintainer asked for small tests on #1998).
# They need the helpers and imports of the PR's test file; kept here as evidence only.

def test_patch_discovery_reads_each_label_store_at_its_own_mid_plane(tmp_path):
    # Only the ink labels were re-converted to create_label_zarrs' 65 planes; the supervision mask
    # still has the surface volume's 28. Coverage must be read from the ink labels' own mid-plane.
    config = _config(tmp_path, patch_size=[3, 2, 2], patch_overlap=1.0, patch_min_labeled_coverage=0.5)
    image = np.zeros((28, 2, 2), dtype=np.uint8)
    labels = np.zeros((label_zarr.VOLUME_DEPTH, 2, 2), dtype=np.uint8)
    labels[label_zarr.LABEL_SLICE] = 255
    supervision = np.zeros((28, 2, 2), dtype=np.uint8)
    supervision[14] = 255
    paths = {}
    for name, value in (("image", image), ("labels", labels), ("supervision", supervision)):
        paths[name] = tmp_path / f"{name}.zarr"
        _write_pyramid(paths[name], value)
    segment = replace(
        _segment(config, tmp_path, image_volume=paths["image"]),
        inklabels=paths["labels"],
        supervision_mask=paths["supervision"],
    )
    training, _ = find_segment_patches(segment, lambda path, scale: open_volume(path, resolution=scale))
    assert len(training) == 1


def test_subtiling_patches_take_their_depth_window_from_the_image(tmp_path):
    config = _config(
        tmp_path,
        patch_size=[3, 2, 2],
        patch_overlap=1.0,
        patch_finding_type="subtiling",
        patch_finding_filter_empty_tile=True,
    )
    segment = replace(
        _segment(config, tmp_path, image_volume="image"),
        inklabels=Path("labels"),
        supervision_mask=Path("supervision"),
    )
    labels = np.zeros((label_zarr.VOLUME_DEPTH, 2, 2), dtype=np.uint8)
    labels[label_zarr.LABEL_SLICE] = 255
    volumes = {
        "image": np.zeros((7, 2, 2), dtype=np.uint8),
        "labels": labels,
        "supervision": labels,
    }

    training, _ = find_subtiling_segment_patches(
        segment, lambda path, resolution: volumes[str(path)]
    )

    assert [patch.bbox for patch in training] == [(2, 0, 0, 5, 2, 2)]
