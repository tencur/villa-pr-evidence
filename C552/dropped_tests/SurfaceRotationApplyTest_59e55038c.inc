// Built only with VC_TESTING (VC3D --test-surface-rotation). The Rotate/Flip tool over a real object store, wired as
// CWindow wires it: the surface panel opens a stored surface (onSurfaceActivated: active surface + "segmentation"),
// replacing the shown surface drops it as the active one (onSurfaceWillBeDeleted), the overlay saves through
// SegmentationModule::saveSurface, and a commit to the active surface by someone else reopens it. "Flip horizontally"
// is ticked and "Apply" clicked in the overlay widget of a real viewer, twice, without selecting the segment again in
// between: the tool opens both times, two flips give the original back, and the segment stays the selected one (id
// and surface). An Apply with no change (Flip ticked and unticked) also leaves the segment selected.
#include "../CState.hpp"
#include "../SurfaceCatalog.hpp"
#include "../SurfacePanelController.hpp"
#include "../SurfaceTreeWidget.hpp"
#include "../ViewerManager.hpp"
#include "../segmentation/SegmentationModule.hpp"
#include "../segmentation/tools/SegmentationEditManager.hpp"
#include "../volume_viewers/VolumeViewerBase.hpp"
#include "vc/core/types/ProjectDocument.hpp"
#include "vc/core/types/VolumePkg.hpp"
#include "support/TestZarrVolume.hpp"
#include <vc_object_store/Uuid.hpp>
#include <vc_object_store/objects/GeometryCoordinates.hpp>

#include <QElapsedTimer>
#include <QTemporaryDir>
#include <QThread>
#include <iostream>

int runSurfaceRotationApplyTests()
{
    using namespace vc::objects;
    namespace fs = std::filesystem;
    auto require = [](bool value, const char* message) { if (!value) throw std::runtime_error(message); };
    auto same = [](const cv::Mat_<cv::Vec3f>& a, const cv::Mat_<cv::Vec3f>& b) {
        return a.size() == b.size() && !a.empty() && cv::norm(a.reshape(1), b.reshape(1), cv::NORM_INF) < 1e-4;
    };
    auto pointsOf = [](const std::shared_ptr<Surface>& s) {
        auto quad = std::dynamic_pointer_cast<QuadSurface>(s);
        return quad && quad->rawPointsPtr() ? quad->rawPointsPtr()->clone() : cv::Mat_<cv::Vec3f>();
    };
    try {
        QTemporaryDir dir;
        vc::test::ZarrVolumeSpec spec;
        spec.shapeZYX = {16, 16, 16}; spec.chunkShapeZYX = {16, 16, 16}; spec.uuid = "rotation-volume";
        const auto volume = vc::test::createZarrVolume(fs::path(dir.path().toStdString()) / "ct.zarr", spec);
        const auto storePath = fs::path(dir.path().toStdString()) / "objects.vcobj";
        OpenOptions options; options.author = "surface-rotation-test";
        { auto created = ObjectStore::create(storePath, options); }
        CState state;
        auto projectPath = storePath; projectPath.replace_extension(".vcproj");
        vc::project::ProjectDocument::create(projectPath, "test", storePath.string(), false);
        state.setVpkg(VolumePkg::open(projectPath));
        const auto attachment = *state.storeFor("surface");

        // Asymmetric along both axes, so a flip moves every vertex.
        cv::Mat_<cv::Vec3f> original(4, 6);
        for (int r = 0; r < original.rows; ++r)
            for (int c = 0; c < original.cols; ++c) original(r, c) = {2.f + c, 3.f + 1.5f * r, 4.f + 0.5f * c + 0.25f * r};
        cv::Mat_<cv::Vec3f> flipped;
        {
            QuadSurface copy(original.clone(), {1.f, 1.f});
            vc3d::surface_rotation::Transform{0.0f, true}.applyInMemory(copy);
            flipped = copy.rawPointsPtr()->clone();
        }
        require(!same(original, flipped), "a flip moves the test grid");
        const auto uuid = util::newUuid();
        {
            QuadSurface grid(original.clone(), {1.f, 1.f});
            state.objectStoreSession()->act(attachment, {{uuid, std::nullopt}}, [&](auto& db, auto& txn) {
                create_quad_in_store(db, txn, uuid, grid, "segment", {}, volumeCoordinates(volume->id(), 1),
                                     {"surface-rotation-test"});
            });
        }

        SurfaceCatalog catalog(state.objectStoreSession());
        QObject::connect(&state, &CState::objectStoreChanged, &catalog,
                         [&](const StoreSessionEvent& event) { catalog.handle(event); });
        catalog.setSource(attachment, volume, {volume->id()});
        SurfaceTreeWidget tree;
        tree.setColumnCount(TIMESTAMP_COLUMN + 1);
        SurfacePanelController panel({&tree}, &state, &catalog, nullptr, {});
        // CWindow::onSurfaceActivated
        QObject::connect(&panel, &SurfacePanelController::surfaceActivated,
                         [&](const QString& id, const std::shared_ptr<QuadSurface>& surface) {
                             state.setActiveSurface(id.toStdString(), surface);
                             state.setSurface("segmentation", state.activeSurface().lock(), false, false);
                         });
        // CWindow::onSurfaceWillBeDeleted
        QObject::connect(&state, &CState::surfaceWillBeDeleted, [&](std::string, std::shared_ptr<Surface> surface) {
            const auto current = state.activeSurface().lock();
            if (current && current == surface) state.clearActiveSurface();
        });
        SegmentationEditManager edits;
        SegmentationModule module(nullptr, &edits, nullptr, nullptr, &state, nullptr, false);
        QString error;
        module.setEditingDestinationResolver([&](const auto& surface) { return panel.prepareSurfaceEditing(surface->id, &error); });
        module.setStorePinResolver([&](const QuadSurface* surface) -> std::optional<SegmentationModule::StoreOrigin> {
            const auto& opened = panel.opened();
            if (opened && surface && opened->surface->id == surface->id && opened->editable)
                return SegmentationModule::StoreOrigin{opened->pin, opened->view};
            return std::nullopt;
        });
        // CWindow: another replica, tool or service changed the open surface: reopen it.
        QObject::connect(&catalog, &SurfaceCatalog::surfacesCommitted, [&](const std::vector<std::string>& uuids) {
            const auto active = state.activeSurface().lock();
            const auto* row = active ? catalog.row(active->id) : nullptr;
            if (!row || std::find(uuids.begin(), uuids.end(), active->id) == uuids.end()) return;
            auto known = module.storePin(active.get());
            if (!known && panel.opened()) known = panel.opened()->pin;
            std::optional<std::string> head;
            catalog.session()->read(row->source, [&](const auto& db, const auto& txn, const auto&) {
                if (const auto revision = db.currentObjectRevision(txn, active->id)) head = revision->uuid;
            });
            if (!head || (known && known->objectRevision == *head)) return;
            panel.activateSurfaceById(active->id);
        });

        auto pump = [&](const std::function<bool()>& done, const char* message) {
            QElapsedTimer timer; timer.start();
            while (!done() && timer.elapsed() < 20000) {
                QApplication::processEvents();
                QThread::msleep(1);
            }
            require(done(), message);
        };
        auto stored = [&] { return pointsOf(catalog.open(uuid).surface); };
        pump([&] { return tree.topLevelItemCount() == 1; }, "the panel lists the stored surface");
        require(panel.activateSurfaceById(uuid, &error), "the stored surface opens");
        require(same(pointsOf(state.surface("segmentation")), original), "the segment is shown as stored");

        QWidget host;
        host.resize(640, 480);
        ViewerManager viewers(&state, state.pointCollection());
        auto* viewer = viewers.createViewerInWidget("segmentation", &host);
        require(viewer, "a segmentation viewer exists");
        host.show();
        SurfaceRotationOverlayController tool(&state);
        tool.setViewerManager(&viewers);
        tool.setSaver([&](QuadSurface* surface) { return module.saveSurface(surface); });

        auto inOverlay = [&](const char* name) -> QWidget* {
            auto* scene = viewer->graphicsView() ? viewer->graphicsView()->scene() : nullptr;
            for (auto* item : scene ? scene->items() : QList<QGraphicsItem*>{})
                if (auto* proxy = dynamic_cast<QGraphicsProxyWidget*>(item); proxy && proxy->widget())
                    if (auto* child = proxy->widget()->findChild<QWidget*>(QString::fromLatin1(name))) return child;
            return nullptr;
        };
        auto selected = [&](const char* message) {
            const auto active = state.activeSurface().lock();
            require(active && active == state.surface("segmentation"), message);
            // MenuActionController::surfaceFromSelection's check.
            require(state.activeSurfaceId() == uuid && panel.opened() && panel.opened()->surface->id == state.activeSurfaceId(),
                    message);
        };
        auto flipAndApply = [&](const cv::Mat_<cv::Vec3f>& expected, const char* message) {
            tool.beginRotate();
            auto* flip = qobject_cast<QCheckBox*>(inOverlay(kHorizontalFlipObjectName));
            auto* apply = qobject_cast<QPushButton*>(inOverlay(kApplyTransformObjectName));
            require(flip && apply, "Actions > Transforms > Rotate opens the tool (Flip and Apply shown)");
            flip->setChecked(true);
            require(apply->isEnabled(), "Apply is enabled for a stored surface");
            apply->click();
            pump([&] { return inOverlay(kApplyTransformObjectName) == nullptr; }, "the save finishes and the overlay closes");
            require(same(stored(), expected), message);
        };

        flipAndApply(flipped, "the first Apply stores the flipped segment");
        require(same(pointsOf(state.surface("segmentation")), flipped), "the viewer shows the flipped segment");
        // The tool is started again on the segment as it is now shown, without selecting it again.
        flipAndApply(original, "a second flip starts from the saved geometry: two flips give the original back");
        require(same(pointsOf(state.surface("segmentation")), original), "the viewer shows the original again");
        selected("after Apply the segment is still the selected one (active surface and id)");

        // Flip ticked and unticked, then Apply: nothing to store, and the segment stays selected.
        tool.beginRotate();
        {
            auto* flip = qobject_cast<QCheckBox*>(inOverlay(kHorizontalFlipObjectName));
            auto* apply = qobject_cast<QPushButton*>(inOverlay(kApplyTransformObjectName));
            require(flip && apply, "the tool opens a third time");
            flip->setChecked(true);
            require(same(pointsOf(state.surface("segmentation")), flipped), "the viewer previews the flip");
            flip->setChecked(false);
            apply->click();
            pump([&] { return inOverlay(kApplyTransformObjectName) == nullptr; }, "Apply with no change closes the tool");
        }
        require(same(stored(), original), "Apply with no change stores nothing");
        selected("after Apply with no change the segment is still the selected one (active surface and id)");
        tool.beginRotate();
        require(inOverlay(kApplyTransformObjectName) != nullptr, "after Apply with no change, Rotate opens the tool again");
        tool.cancelRotate();
        require(panel.activateSurfaceById(uuid, &error) && same(pointsOf(state.surface("segmentation")), original),
                "selecting the segment again shows what is stored");
        state.closeProject();
        std::cerr << "surface rotation apply tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "surface rotation apply test: " << error.what() << '\n';
        return 1;
    }
}
