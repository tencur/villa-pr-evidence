// Drives the Rotate/Flip tool the way a user does: a segment of a project is selected (CState entry, active
// surface and "segmentation" slot, as CWindow::onSurfaceActivated sets them), the tool is started, "Flip
// horizontally" is ticked and "Apply" is clicked in the overlay widget of a real viewer. After Apply every
// holder of the segment must show the saved (flipped) geometry, and a second flip must compose with the first.
#include "CState.hpp"
#include "ViewerManager.hpp"
#include "overlays/SurfaceRotationOverlayController.hpp"
#include "overlays/SurfaceRotationTransform.hpp"
#include "volume_viewers/CVolumeViewerView.hpp"
#include "volume_viewers/VolumeViewerBase.hpp"

#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/util/QuadSurface.hpp"

#include <QApplication>
#include <QCheckBox>
#include <QGraphicsProxyWidget>
#include <QGraphicsScene>
#include <QGraphicsView>
#include <QPushButton>
#include <QTemporaryDir>
#include <QTest>
#include <QWidget>

#include <opencv2/core.hpp>

#include <filesystem>
#include <memory>
#include <string>

namespace fs = std::filesystem;

namespace
{
constexpr const char* kSegmentId = "seg_a";

cv::Mat_<cv::Vec3f> makeGrid()
{
    // Asymmetric along both axes, so a flip is visible in every vertex.
    cv::Mat_<cv::Vec3f> points(4, 6);
    for (int r = 0; r < points.rows; ++r) {
        for (int c = 0; c < points.cols; ++c) {
            points(r, c) = cv::Vec3f(100.0f + 10.0f * c, 200.0f + 7.0f * r, 300.0f + 3.0f * c + r);
        }
    }
    return points;
}

bool samePoints(const cv::Mat_<cv::Vec3f>& a, const cv::Mat_<cv::Vec3f>& b)
{
    if (a.size() != b.size()) {
        return false;
    }
    return cv::norm(a.reshape(1), b.reshape(1), cv::NORM_INF) < 1e-4;
}

cv::Mat_<cv::Vec3f> pointsOf(const std::shared_ptr<Surface>& surface)
{
    auto quad = std::dynamic_pointer_cast<QuadSurface>(surface);
    return quad ? quad->rawPoints().clone() : cv::Mat_<cv::Vec3f>();
}

cv::Mat_<cv::Vec3f> pointsOnDisk(const fs::path& path)
{
    try {
        QuadSurface surface(path);
        return surface.rawPoints().clone();
    } catch (...) {
        return {};
    }
}

// CWindow::onSurfaceActivated: the CState entry for the id first, then VolumePkg.
void selectSegment(CState& state, const std::string& id)
{
    auto surface = std::dynamic_pointer_cast<QuadSurface>(state.surface(id));
    if (!surface && state.vpkg()) {
        surface = state.vpkg()->getSurface(id);
    }
    state.setActiveSurface(id, surface);
    state.setSurface("segmentation", state.activeSurface().lock(), false, false);
}

template <typename T>
T* findInOverlay(VolumeViewerBase* viewer, const char* objectName)
{
    auto* scene = viewer->graphicsView() ? viewer->graphicsView()->scene() : nullptr;
    if (!scene) {
        return nullptr;
    }
    for (auto* item : scene->items()) {
        if (auto* proxy = dynamic_cast<QGraphicsProxyWidget*>(item); proxy && proxy->widget()) {
            if (auto* child = proxy->widget()->findChild<T*>(QString::fromLatin1(objectName))) {
                return child;
            }
        }
    }
    return nullptr;
}
} // namespace

class SurfaceRotationApplyTest : public QObject
{
    Q_OBJECT

private slots:
    void applyUpdatesEveryHolderAndComposes();
};

void SurfaceRotationApplyTest::applyUpdatesEveryHolderAndComposes()
{
    QTemporaryDir tmp;
    QVERIFY(tmp.isValid());
    const fs::path root(tmp.path().toStdString());
    const fs::path segPath = root / "paths" / kSegmentId;
    fs::create_directories(segPath.parent_path());

    const cv::Mat_<cv::Vec3f> original = makeGrid();
    {
        QuadSurface seed(original, cv::Vec2f(1.0f, 1.0f));
        seed.save(segPath);
    }
    cv::Mat_<cv::Vec3f> flipped;
    {
        QuadSurface copy(original.clone(), cv::Vec2f(1.0f, 1.0f));
        vc3d::surface_rotation::Transform{0.0f, true}.applyInMemory(copy);
        flipped = copy.rawPoints().clone();
    }
    QVERIFY(!samePoints(original, flipped));

    auto pkg = VolumePkg::newEmpty();
    pkg->save(root / "project.volpkg.json");
    pkg->addSegmentsEntry("paths");
    pkg->setSegmentationDirectory("paths");

    CState state;
    state.setVpkg(pkg);
    auto loaded = pkg->loadSurface(kSegmentId);
    QVERIFY(loaded);
    state.setSurface(kSegmentId, loaded, true, false);
    selectSegment(state, kSegmentId);

    QWidget host;
    host.resize(640, 480);
    ViewerManager viewers(&state, state.pointCollection());
    auto* viewer = viewers.createViewerInWidget("segmentation", &host);
    QVERIFY(viewer);
    host.show();

    SurfaceRotationOverlayController tool(&state);
    tool.setViewerManager(&viewers);

    auto flipAndApply = [&](const cv::Mat_<cv::Vec3f>& expectedOnDisk) {
        tool.beginRotate();
        auto* flip = findInOverlay<QCheckBox>(viewer, "surfaceHorizontalFlipCheckBox");
        auto* apply = findInOverlay<QPushButton>(viewer, "surfaceTransformApplyButton");
        QVERIFY(flip);
        QVERIFY(apply);
        flip->setChecked(true);
        QVERIFY(apply->isEnabled());
        apply->click();
        QTRY_VERIFY_WITH_TIMEOUT(samePoints(pointsOnDisk(segPath), expectedOnDisk), 20000);
        // The save worker reports back on the event loop.
        QTRY_VERIFY_WITH_TIMEOUT(findInOverlay<QPushButton>(viewer, "surfaceTransformApplyButton") == nullptr, 20000);
    };

    // First Apply: the flipped segment is on disk and in the viewer.
    flipAndApply(flipped);
    QVERIFY(samePoints(pointsOf(state.surface("segmentation")), flipped));

    // Every other holder of the segment shows the saved geometry too.
    QVERIFY(samePoints(pointsOf(pkg->getSurface(kSegmentId)), flipped));
    QVERIFY(samePoints(pointsOf(state.surface(kSegmentId)), flipped));
    QVERIFY(samePoints(pointsOf(state.activeSurface().lock()), flipped));

    // Selecting the segment again in the panel keeps showing it flipped.
    selectSegment(state, kSegmentId);
    QVERIFY(samePoints(pointsOf(state.surface("segmentation")), flipped));

    // A second flip starts from the saved geometry: two flips give the original back.
    flipAndApply(original);
    QVERIFY(samePoints(pointsOf(state.surface("segmentation")), original));
}

QTEST_MAIN(SurfaceRotationApplyTest)
#include "test_surface_rotation_apply.moc"
