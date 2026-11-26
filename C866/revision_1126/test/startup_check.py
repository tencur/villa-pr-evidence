"""Start vesuvius.napari_trainer (main_window.main, the console script) from whatever vesuvius is on
sys.path, with napari offscreen; instead of handing control to a person, report what started."""
import json, sys, traceback
import napari

result = {}


def started():
    from vesuvius.napari_trainer import main_window as mw
    viewer = napari.current_viewer()
    result["docked_widgets"] = sorted(viewer.window.dock_widgets)
    result["config"] = str(mw._config_manager._config_path)


if __name__ == "__main__":
    from napari._vispy.layers import base, image, labels  # no OpenGL here: answer the canvas capability queries
    image.get_gl_extensions = lambda: "texture_float"
    for mod in (base, image, labels):
        mod.get_max_texture_sizes = lambda: (16384, 2048)
    napari.run = started
    import vesuvius.napari_trainer.main_window as mw
    result["main_window"] = mw.__file__
    try:
        mw.main()
        result["main"] = "returned"
    except Exception as e:
        traceback.print_exc()
        result["main"] = f"{type(e).__name__}: {e}"
    print("RESULT " + json.dumps(result))
