"""Save a PNG of the whole X screen (includes popup menus and dialogs, which a widget grab misses)."""
import numpy as np, cv2
from Xlib import display, X
def shot(path, disp=None):
    d = display.Display(disp); r = d.screen().root; g = r.get_geometry()
    raw = r.get_image(0, 0, g.width, g.height, X.ZPixmap, 0xffffffff)
    a = np.frombuffer(raw.data, dtype=np.uint8).reshape(g.height, g.width, 4)[:, :, :3]
    cv2.imwrite(path, a); d.close()
