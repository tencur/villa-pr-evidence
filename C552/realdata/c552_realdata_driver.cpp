// Real-data driver for C552 (not part of the PR). A project with the public PHerc0800 auto-grown segment
// 20251028220955 in paths/, opened with VC3D's own objects: VolumePkg, CState, SurfacePanelController (tree), and the
// FileWatcherService on the real inotify watch, as CWindow wires them. The segment is selected the way
// CWindow::onSurfaceActivated selects it; the Rotate/Flip tool (SurfaceRotationOverlayController) is started as the
// Segment > Rotate menu starts it, the Flip box is ticked and Apply is pressed (the widget's applyRequested callback;
// called directly here because no viewer is open). The event loop runs while the save worker and the file watcher work.
// Round 1: one flip. Then the user selects the segment again. Round 2: a second flip (two flips = the original).
#include <QApplication>
#include <QElapsedTimer>
#include <QTreeWidget>
#include <QMetaObject>
#include <QObject>
#include <QPointer>

#include <opencv2/core.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "CState.hpp"
#include "FileWatcherService.hpp"
#include "SurfacePanelController.hpp"
#include "overlays/SurfaceRotationTransform.hpp"
#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/util/QuadSurface.hpp"

#define private public
#include "overlays/SurfaceRotationOverlayController.hpp"
#undef private

namespace fs = std::filesystem;

static double maxDiff(const cv::Mat_<cv::Vec3f>& a, const cv::Mat_<cv::Vec3f>& b)
{
    if (a.size() != b.size()) return 1e30;
    double m = 0;
    for (int r = 0; r < a.rows; ++r)
        for (int c = 0; c < a.cols; ++c) {
            const auto& p = a(r, c); const auto& q = b(r, c);
            if (p[0] == -1 && q[0] == -1) continue;
            for (int k = 0; k < 3; ++k) m = std::max(m, (double)std::abs(p[k] - q[k]));
        }
    return m;
}

static const char* which(const cv::Mat_<cv::Vec3f>& p, const cv::Mat_<cv::Vec3f>& orig, const cv::Mat_<cv::Vec3f>& flipped)
{
    if (p.empty()) return "(none)";
    if (maxDiff(p, orig) < 1e-3) return "ORIGINAL";
    if (maxDiff(p, flipped) < 1e-3) return "FLIPPED";
    return "OTHER";
}

static cv::Mat_<cv::Vec3f> pts(const std::shared_ptr<Surface>& s)
{
    auto q = std::dynamic_pointer_cast<QuadSurface>(s);
    return q ? q->rawPoints().clone() : cv::Mat_<cv::Vec3f>();
}

static cv::Mat_<cv::Vec3f> disk(const fs::path& p)
{
    QuadSurface s(p);
    return s.rawPoints().clone();
}

static void pump(int ms)
{
    QElapsedTimer t; t.start();
    while (t.elapsed() < ms) { QCoreApplication::processEvents(QEventLoop::AllEvents, 50); }
}

static void activate(CState& state, const std::shared_ptr<VolumePkg>& pkg, const std::string& id)
{
    // CWindow::onSurfaceActivated (a4979b99): CState entry by id first, then VolumePkg.
    auto s = std::dynamic_pointer_cast<QuadSurface>(state.surface(id));
    if (!s) s = pkg->getSurface(id);
    state.setActiveSurface(id, s);
    state.setSurface("segmentation", state.activeSurface().lock(), false, false);
}

int main(int argc, char** argv)
{
    QApplication app(argc, argv);
    const fs::path root(argv[1]);
    const std::string id = argv[2];
    auto pkg = VolumePkg::newEmpty();
    pkg->save(root / "project.volpkg.json");
    pkg->addSegmentsEntry("paths");
    pkg->setSegmentationDirectory("paths");

    CState state;
    state.setVpkg(pkg);
    QTreeWidget tree;
    tree.setColumnCount(2);
    SurfacePanelController panel(SurfacePanelController::UiRefs{&tree, nullptr}, &state, nullptr,
                                 [] { return nullptr; }, {}, nullptr);
    panel.setVolumePkg(pkg);
    panel.loadSurfaces(false);
    FileWatcherService watcher(&state);
    watcher.setSurfacePanel(&panel);
    watcher.startWatching();

    const fs::path segPath = root / "paths" / id;
    const auto orig = disk(segPath);
    cv::Mat_<cv::Vec3f> flipped;
    {
        QuadSurface tmp(orig.clone(), cv::Vec2f(1, 1));
        vc3d::surface_rotation::Transform{0.0f, true}.applyInMemory(tmp);
        flipped = tmp.rawPoints().clone();
    }
    int valid = 0;
    for (auto& p : orig) valid += p[0] != -1;
    std::printf("RESULT segment %s: grid %dx%d, %d valid vertices; flipped differs from original by up to %.1f vx\n",
                id.c_str(), orig.cols, orig.rows, valid, maxDiff(orig, flipped));

    activate(state, pkg, id);
    SurfaceRotationOverlayController ctrl(&state);

    auto report = [&](const char* when) {
        std::printf("RESULT %s:\n", when);
        std::printf("RESULT   on disk (x/y/z.tif)              %s\n", which(disk(segPath), orig, flipped));
        std::printf("RESULT   viewer slot \"segmentation\"       %s\n", which(pts(state.surface("segmentation")), orig, flipped));
        std::printf("RESULT   active surface                   %s\n", which(pts(state.activeSurface().lock()), orig, flipped));
        std::printf("RESULT   VolumePkg::getSurface(id)        %s\n", which(pts(pkg->getSurface(id)), orig, flipped));
        auto byId = state.surface(id);
        std::printf("RESULT   CState entry for the id          %s\n", byId ? which(pts(byId), orig, flipped) : "(none)");
        std::printf("RESULT   segment listed (VolumePkg/tree)  %s/%d\n", pkg->segmentation(id) ? "yes" : "no", tree.topLevelItemCount());
    };

    auto applyFlip = [&](int round) {
        ctrl.beginRotate();                 // Segment > Rotate (MenuActionController::beginRotateSurfaceTransform)
        ctrl.setHorizontalFlip(true);       // Flip box
        ctrl.applyRotation();               // Apply button
        QElapsedTimer t; t.start();
        while (ctrl._saveInFlight && t.elapsed() < 60000) QCoreApplication::processEvents(QEventLoop::AllEvents, 50);
        pump(8000);                         // file watcher batches and processes its events
        std::printf("RESULT round %d: Apply finished (save in flight: %s)\n", round, ctrl._saveInFlight ? "yes" : "no");
    };

    report("before");
    applyFlip(1);
    report("after Apply (flip 1)");
    activate(state, pkg, id);
    report("after selecting the segment again in the panel");
    applyFlip(2);
    report("after a second Apply (flip 2; two flips = original)");
    std::fflush(stdout);
    return 0;
}
