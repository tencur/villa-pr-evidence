// Built only with VC_TESTING (VC3D --test-surface-rotation). The Rotate/Flip tool over a real object store, wired as
// CWindow wires it: the surface panel opens a stored surface (onSurfaceActivated: active surface + "segmentation"),
// replacing the shown surface drops it as the active one (onSurfaceWillBeDeleted), the overlay saves through
// SegmentationModule::saveSurface, and a commit to the active surface by someone else reopens it. "Flip horizontally"
// is ticked and "Apply" clicked in the overlay widget of a real viewer, twice, without selecting the segment again in
// between: the tool opens both times, two flips give the original back, and the segment stays the selected one (id
// and surface). An Apply with no change (Flip ticked and unticked) also leaves the segment selected.
// Then one case per review point (#2466): Apply respects a newer selection (another surface, a cleared selection, the
// segment deleted), does not overwrite a newer save (the surface panel's Flip, another client), and after a failed
// save does not leave an old revision selected (the agent bridge then could not reopen the current one). Every case
// runs and reports; the test fails if any case fails.
// 22 Nov review: an autosave through the segment's store edit that finishes while Apply saves (6a), the shown preview
// edited and autosaved before Apply, then Rotate again (6b), the surface panel's Flip during a pending Apply must leave
// Rotate able to reopen on the saved surface (4b, stricter than 20 Nov), and revision lookups that return nothing
// (7a: unknown at Apply, 7b: unknown at open and at Apply) neither save nor deselect.
// The tool is wired as CWindow wires it at the variant under test (requires checks pick the API it has).
#include "../CState.hpp"
#include "../SurfaceCatalog.hpp"
#include "../SurfacePanelController.hpp"
#include "../SurfaceTreeWidget.hpp"
#include "../ViewerManager.hpp"
#include "../segmentation/SegmentationModule.hpp"
#include "../segmentation/SurfaceStoreEdit.hpp"
#include "../segmentation/tools/SegmentationEditManager.hpp"
#include "../volume_viewers/VolumeViewerBase.hpp"
#include "vc/core/types/ProjectDocument.hpp"
#include "vc/core/types/VolumePkg.hpp"
#include "test/support/TestVolume.hpp"
#include <vc_object_store/Uuid.hpp>
#include <vc_object_store/objects/GeometryCoordinates.hpp>

#include <QElapsedTimer>
#include <QTemporaryDir>
#include <QTimer>
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
        vc3d::test::VolumeSpec spec;
        spec.shapeZYX = {16, 16, 16}; spec.chunkShapeZYX = {16, 16, 16};
        const auto volume = vc3d::test::createVolume(fs::path(dir.path().toStdString()) / "ct.zarr", spec);
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
        // The Rotate tool's horizontal flip, and the surface panel's Flip Surface > V Flip (QuadSurface::flipU).
        auto flippedH = [](const cv::Mat_<cv::Vec3f>& points) {
            QuadSurface copy(points.clone(), {1.f, 1.f});
            vc3d::surface_rotation::Transform{0.0f, true}.applyInMemory(copy);
            return cv::Mat_<cv::Vec3f>(copy.rawPointsPtr()->clone());
        };
        auto flippedU = [](const cv::Mat_<cv::Vec3f>& points) {
            QuadSurface copy(points.clone(), {1.f, 1.f});
            copy.flipU();
            return cv::Mat_<cv::Vec3f>(copy.rawPointsPtr()->clone());
        };
        require(!same(flippedU(original), flippedH(original)) && !same(flippedU(original), original),
                "the panel's V Flip and the tool's flip give different grids");
        const auto uuid = util::newUuid();
        const auto uuidB = util::newUuid();
        for (const auto& [id, name, offset] : {std::tuple{uuid, "segment", 0.f}, std::tuple{uuidB, "other", 7.f}}) {
            QuadSurface grid(cv::Mat_<cv::Vec3f>(original + cv::Scalar(offset, offset, offset)), {1.f, 1.f});
            // 7e6b82164 (#2462): a checked request, as SurfacePanelStoreTest.inc creates its surfaces.
            auto request = state.objectStoreSession()->newRequest(attachment);
            request.expectedObjects = {{id, std::nullopt}};
            request.operations = createSurfaceOperations(id, name, {}, volumeCoordinates(volume->id(), 1),
                                                         {"surface-rotation-test"}, grid.meta, quad_surface_geometry(grid));
            state.objectStoreSession()->submit(attachment, request);
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
        // CWindow::clearSurfaceSelection (the panel cleared its selection, or the open surface left the store)
        QObject::connect(&panel, &SurfacePanelController::surfaceSelectionCleared, [&] { state.clearActiveSurface(); });
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
            // 1386f072e: as CWindow reads the head and compares the pins by reference().
            const auto head = catalog.session()->currentObject(row->source, active->id);
            if (!head || (known && known->reference() == head->reference())) return;
            if (module.editingEnabled()) return;
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
        pump([&] { return tree.topLevelItemCount() == 2; }, "the panel lists the stored surfaces");
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
        // Another client saves the segment (the panel's V Flip of what is stored), through its own store edit.
        auto commitElsewhere = [&] {
            auto opened = catalog.open(uuid);
            opened.surface->flipU();
            const auto* points = opened.surface->rawPointsPtr();
            SurfaceStoreEdit other(state.objectStoreSession(), opened.pin, opened.view);
            other.save(*opened.surface, cv::Rect(0, 0, points->cols, points->rows), {}, std::nullopt);
            return cv::Mat_<cv::Vec3f>(points->clone());
        };
        bool commitInsideSave = false;
        // Runs in the saver, i.e. after the tool's last check and before SegmentationModule::saveSurface.
        std::function<void()> insideSave;
        auto beforeSave = [&] {
            if (std::exchange(commitInsideSave, false)) commitElsewhere();
            if (auto hook = std::exchange(insideSave, nullptr)) hook();
        };
        auto saveThrough = [&](auto& segmentation, QuadSurface* surface, const std::optional<std::string>& revision) {
            if constexpr (requires { segmentation.saveSurface(surface, true, std::vector<std::string>{}, revision); })
                return segmentation.saveSurface(surface, true, std::vector<std::string>{}, revision);
            else
                return segmentation.saveSurface(surface);
        };
        // The store does not answer (a hosted lookup after an error): the revision lookups return nothing.
        bool headUnknown = false, sourceUnknown = false;
        // CWindow at b860d5aa0: the stored head's revision uuid.
        auto revisionOf = [&](const std::string& id) -> std::optional<std::string> {
            if (headUnknown) return std::nullopt;
            const auto* row = catalog.row(id);
            const auto head = row ? catalog.session()->currentObject(row->source, id) : std::nullopt;
            return head ? std::optional(head->uuid) : std::nullopt;
        };
        // CWindow after the 22 Nov change: the stored head's geometry revision, the geometry revision the
        // segment's store edit holds for the surface Rotate opens on, and reopening through the surface panel.
        auto geometryRevisionOf = [&](const std::string& id) -> std::optional<std::string> {
            if (headUnknown) return std::nullopt;
            const auto* row = catalog.row(id);
            const auto head = row ? catalog.session()->currentObject(row->source, id) : std::nullopt;
            return head ? head->geometryRevision : std::nullopt;
        };
        auto sourceRevisionOf = [&](QuadSurface* surface) -> std::optional<std::string> {
            if (sourceUnknown) return std::nullopt;
            const auto pin = module.storePin(surface);
            return pin ? pin->geometryRevision : std::nullopt;
        };
        int reopened = 0;
        auto reopen = [&](const std::string& id) { ++reopened; return panel.activateSurfaceById(id); };
        auto wireTool = [&](auto& overlay) {
            if constexpr (requires { overlay.setReopener(reopen); }) {
                overlay.setSaver([&](QuadSurface* surface, const std::optional<std::string>& revision) {
                    beforeSave();
                    return saveThrough(module, surface, revision);
                });
                overlay.setSourceRevisionReader(sourceRevisionOf);
                overlay.setRevisionReader(geometryRevisionOf);
                overlay.setReopener(reopen);
                std::cerr << "wired: saver with the transformed revision, source and stored geometry revisions, reopen\n";
            } else {
                overlay.setSaver([&](QuadSurface* surface) {
                    beforeSave();
                    return saveThrough(module, surface, std::nullopt);
                });
                if constexpr (requires { overlay.setRevisionReader(revisionOf); }) {
                    overlay.setRevisionReader(revisionOf);
                    std::cerr << "wired: saver, stored revision\n";
                } else {
                    std::cerr << "wired: saver\n";
                }
            }
        };
        wireTool(tool);

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
        // The selected surface is the stored segment (what the next Rotate, the panel and the agent bridge start from).
        auto selectedAsStored = [&](const char* message) {
            selected(message);
            require(same(pointsOf(state.activeSurface().lock()), stored()), message);
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

        // ---- Review cases. Message boxes (the tool's warnings) are answered at once.
        auto stale = [&] {
            // The agent bridge reopens a segment only when it is not the active id: an active id must be the stored one.
            const auto active = state.activeSurface().lock();
            return state.activeSurfaceId() == uuid && (!active || !same(pointsOf(active), stored()));
        };
        std::vector<std::string> messages;
        bool watchStale = false, staleAtMessage = false;
        QTimer answer;
        answer.setInterval(2);
        QObject::connect(&answer, &QTimer::timeout, [&] {
            if (auto* box = qobject_cast<QMessageBox*>(QApplication::activeModalWidget())) {
                messages.push_back(box->windowTitle().toStdString());
                if (watchStale && stale()) staleAtMessage = true;
                box->done(QMessageBox::Ok);
            }
        });
        answer.start();
        auto settle = [&](int ms) {
            QElapsedTimer timer; timer.start();
            while (timer.elapsed() < ms) { state.pollObjectStoreChanges(); QApplication::processEvents(); QThread::msleep(1); }
        };
        auto openTool = [&](bool flip) {
            tool.beginRotate();
            auto* check = qobject_cast<QCheckBox*>(inOverlay(kHorizontalFlipObjectName));
            auto* apply = qobject_cast<QPushButton*>(inOverlay(kApplyTransformObjectName));
            require(check && apply, "Actions > Transforms > Rotate opens the tool (Flip and Apply shown)");
            if (flip) check->setChecked(true);
            return apply;
        };
        auto waitForApply = [&](std::size_t before) {
            pump([&] { return inOverlay(kApplyTransformObjectName) == nullptr || messages.size() > before; },
                 "Apply finishes (the tool closes or reports)");
        };
        // SegmentationCommandHandler::onFlipSurface (Flip Surface > V Flip): only on the active surface of that id
        // (requireSurfaceAndRunner); flips it in place, saves it and shows it.
        auto panelFlip = [&] {
            const auto surface = state.activeSurface().lock();
            if (!surface || surface->id != uuid) return false;
            surface->flipU();
            require(module.saveSurface(surface.get()), "the surface panel's Flip saves");
            state.setSurface("segmentation", surface, false, false);
            return true;
        };
        int passed = 0, failed = 0;
        auto runCase = [&](const char* name, const std::function<void()>& body) {
            try {
                tool.cancelRotate();
                require(panel.activateSurfaceById(uuid, &error), "the segment opens");
                body();
                ++passed;
                std::cerr << "case passed: " << name << '\n';
            } catch (const std::exception& e) {
                ++failed;
                std::cerr << "case FAILED: " << name << ": " << e.what() << '\n';
            }
            watchStale = staleAtMessage = false;
            headUnknown = sourceUnknown = false;
            commitInsideSave = false;
            insideSave = nullptr;
            tool.cancelRotate();
            settle(250);
        };

        runCase("1 Rotate on A, select B, Apply", [&] {
            const auto base = stored();
            auto* apply = openTool(true);
            require(panel.activateSurfaceById(uuidB, &error), "B opens");
            const auto before = messages.size();
            apply->click();
            waitForApply(before);
            const auto active = state.activeSurface().lock();
            require(state.activeSurfaceId() == uuidB && active && active->id == uuidB &&
                        active == state.surface("segmentation"),
                    "after Apply, B is still the selected and shown surface");
            require(same(stored(), base), "Apply after selecting B does not save A");
        });
        runCase("2 Apply, Rotate again, clear the selection, Apply", [&] {
            const auto base = stored();
            auto* apply = openTool(true);
            auto before = messages.size();
            apply->click();
            waitForApply(before);
            require(same(stored(), flippedH(base)), "the first Apply stores the flip (the store edit is cached)");
            apply = openTool(true);
            tree.clearSelection();
            require(state.activeSurfaceId().empty(), "clearing the panel selection clears the active surface");
            before = messages.size();
            apply->click();
            waitForApply(before);
            require(state.activeSurfaceId().empty() && !state.activeSurface().lock(), "after Apply no surface is selected");
            require(same(stored(), flippedH(base)), "Apply after clearing the selection saves nothing");
        });
        runCase("4a Apply pending, A selected again and flipped from the surface panel", [&] {
            const auto base = stored();
            auto* apply = openTool(true);
            const auto before = messages.size();
            apply->click();  // the result is saved when the event loop delivers it
            require(panel.activateSurfaceById(uuid, &error), "the segment opens again");
            require(panelFlip(), "the surface panel's Flip runs on the selected segment");
            waitForApply(before);
            require(same(stored(), flippedU(base)), "the pending Apply does not overwrite the surface panel's newer save");
            require(!stale(), "the segment is not selected at an old revision");
        });
        runCase("4b Apply pending, the surface panel's Flip saves first, Rotate again", [&] {
            const auto base = stored();
            auto* apply = openTool(true);
            auto before = messages.size();
            apply->click();  // the result is saved when the event loop delivers it
            require(panelFlip(), "the surface panel's Flip runs on the selected segment during the pending Apply");
            waitForApply(before);
            require(same(stored(), flippedU(base)), "the pending Apply does not overwrite the surface panel's newer save");
            require(!stale(), "the segment is not selected at an old revision");
            selectedAsStored("after the cancelled Apply the surface panel's saved surface is the selected one");
            apply = openTool(true);
            before = messages.size();
            apply->click();
            waitForApply(before);
            require(same(stored(), flippedH(flippedU(base))), "Rotate again starts from the surface panel's save");
            selectedAsStored("after Rotate again the saved segment is selected");
        });
        runCase("5a another client saves during the preview, then Apply", [&] {
            auto* apply = openTool(true);
            const auto newer = commitElsewhere();
            settle(400);  // the store's change notice reaches CWindow's reopen rule
            watchStale = true;
            const auto before = messages.size();
            apply->click();
            waitForApply(before);
            require(same(stored(), newer), "Apply does not overwrite the other client's newer save");
            require(!stale() && !staleAtMessage,
                    "after Apply the segment is not selected at an old revision (the agent bridge can reopen it)");
        });
        runCase("5b the save fails (another client saved first)", [&] {
            auto* apply = openTool(true);
            commitInsideSave = true;
            watchStale = true;
            const auto before = messages.size();
            apply->click();
            waitForApply(before);
            require(!commitInsideSave, "the save was attempted");
            require(std::find(messages.begin() + before, messages.end(), "Surface Transform Failed") != messages.end(),
                    "the failed save is reported");
            require(!staleAtMessage,
                    "after the failed save the segment is not selected at an old revision (the agent bridge can reopen it)");
            settle(400);
            require(!stale(), "after the store's change notice the segment is not selected at an old revision");
        });
        runCase("6a an autosave through the segment's store edit finishes while Apply saves", [&] {
            const auto base = stored();
            auto* apply = openTool(true);
            cv::Mat_<cv::Vec3f> autosaved;
            insideSave = [&] {
                // An edit of the shown surface and its autosave (SegmentationModule::flushAutosave: the
                // "segmentation" surface through the segment's store edit, on a worker), started after the
                // tool's last check; saveSurface waits for it before it saves the transform.
                auto shown = std::dynamic_pointer_cast<QuadSurface>(state.surface("segmentation"));
                if (!shown) return;  // (no throwing out of the tool's finished slot)
                (*shown->rawPointsPtr())(1, 2) += cv::Vec3f(0.5f, 0.f, 0.f);
                shown->invalidateCache();
                autosaved = shown->rawPointsPtr()->clone();
                module.flushAutosave();
            };
            watchStale = true;
            auto before = messages.size();
            apply->click();
            waitForApply(before);
            require(!insideSave && !autosaved.empty(), "the save was attempted");
            settle(300);
            require(same(stored(), autosaved) && !same(stored(), flippedH(base)),
                    "Apply does not overwrite the autosave that finished while it saved");
            require(std::find(messages.begin() + before, messages.end(), "Surface Transform Failed") != messages.end(),
                    "the refused save is reported");
            require(!staleAtMessage, "after the refused save the segment is not selected at an old revision");
            selectedAsStored("after the refused save the autosaved segment is the selected one");
            watchStale = false;
            apply = openTool(true);
            before = messages.size();
            apply->click();
            waitForApply(before);
            require(same(stored(), flippedH(autosaved)), "Rotate again starts from the autosaved segment");
            selectedAsStored("after Rotate again the saved segment is selected");
        });
        runCase("6b the shown preview is edited and autosaved, Apply, Rotate again", [&] {
            auto* apply = openTool(true);
            auto shown = std::dynamic_pointer_cast<QuadSurface>(state.surface("segmentation"));
            require(shown && shown != state.activeSurface().lock(), "the viewer shows the tool's preview");
            (*shown->rawPointsPtr())(2, 3) += cv::Vec3f(0.f, 0.75f, 0.f);
            shown->invalidateCache();
            const cv::Mat_<cv::Vec3f> edited = shown->rawPointsPtr()->clone();
            module.flushAutosave();
            pump([&] { return same(stored(), edited); }, "the edit of the shown preview is autosaved");
            settle(300);  // the autosave completes; the store's change notice reaches the window's reopen rule
            auto before = messages.size();
            apply->click();
            waitForApply(before);
            settle(100);
            require(same(stored(), edited), "Apply does not overwrite the autosaved edit");
            require(std::find(messages.begin() + before, messages.end(), "Surface Transform Cancelled") != messages.end(),
                    "the cancelled Apply is reported");
            selectedAsStored("after the cancelled Apply the autosaved segment is the selected one");
            apply = openTool(true);
            before = messages.size();
            apply->click();
            waitForApply(before);
            require(same(stored(), flippedH(edited)), "Rotate again starts from the autosaved edit, not the old geometry");
            selectedAsStored("after Rotate again the saved segment is selected");
        });
        runCase("7a the stored revision cannot be read at Apply", [&] {
            const auto base = stored();
            auto* apply = openTool(true);
            headUnknown = true;
            auto before = messages.size();
            apply->click();
            waitForApply(before);
            headUnknown = false;
            require(same(stored(), base), "Apply does not save over a stored revision it cannot read");
            selectedAsStored("the segment stays selected (an unreadable revision is not a newer one)");
            apply = openTool(true);
            before = messages.size();
            apply->click();
            waitForApply(before);
            require(same(stored(), flippedH(base)), "Rotate again saves once the revision can be read");
            selectedAsStored("after Rotate again the saved segment is selected");
        });
        runCase("7b the revision cannot be read when Rotate opens nor at Apply", [&] {
            const auto base = stored();
            headUnknown = sourceUnknown = true;
            auto* apply = openTool(true);
            const auto before = messages.size();
            apply->click();
            waitForApply(before);
            headUnknown = sourceUnknown = false;
            require(same(stored(), base), "Apply does not save when no revision is known (two unknown revisions do not match)");
            selectedAsStored("the segment stays selected (an unreadable revision is not a newer one)");
        });
        runCase("3 Rotate on A, A deleted, Apply with no change", [&] {
            auto* apply = openTool(false);
            require(panel.retireSurfaces({QString::fromStdString(uuid)}, &error), "the segment is deleted");
            pump([&] { return tree.topLevelItemCount() == 1; }, "the panel drops the deleted segment");
            require(state.activeSurfaceId().empty(), "deleting the selected segment clears the selection");
            const auto before = messages.size();
            apply->click();
            waitForApply(before);
            require(state.activeSurfaceId() != uuid,
                    "after Apply with no change the deleted segment is not the active one (the agent bridge reads its row)");
        });
        answer.stop();
        tool.cancelRotate();
        state.closeProject();
        std::cerr << "reopened through the surface panel: " << reopened << "\n";
        std::cerr << "surface rotation apply cases: " << passed << " passed, " << failed << " failed\n";
        if (failed) return 1;
        std::cerr << "surface rotation apply tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "surface rotation apply test: " << error.what() << '\n';
        return 1;
    }
}
