"""Mouse clicks and typing on an X display with XTEST (stands in for the user's mouse and keyboard)."""
import sys, time
from Xlib import X, XK, display
from Xlib.ext import xtest
def click(x, y, button=1, disp=None):
    d = display.Display(disp)
    xtest.fake_input(d, X.MotionNotify, x=int(x), y=int(y)); d.sync(); time.sleep(0.15)
    xtest.fake_input(d, X.ButtonPress, button); d.sync(); time.sleep(0.08)
    xtest.fake_input(d, X.ButtonRelease, button); d.sync(); time.sleep(0.1)
    d.close()
_NAMES = {"/": "slash", ".": "period", "_": "underscore", "-": "minus", " ": "space", ":": "colon"}
def _key(d, keysym_name, shift=False):
    kc = d.keysym_to_keycode(XK.string_to_keysym(keysym_name))
    sk = d.keysym_to_keycode(XK.string_to_keysym("Shift_L"))
    if shift: xtest.fake_input(d, X.KeyPress, sk)
    xtest.fake_input(d, X.KeyPress, kc); xtest.fake_input(d, X.KeyRelease, kc)
    if shift: xtest.fake_input(d, X.KeyRelease, sk)
    d.sync(); time.sleep(0.02)
def type_text(text, disp=None):
    d = display.Display(disp)
    for ch in text:
        name = _NAMES.get(ch, ch)
        shift = ch.isupper() or ch in "_:"
        _key(d, name, shift)
    d.close()
def key(name, disp=None):
    d = display.Display(disp); _key(d, name); d.close()
if __name__ == "__main__":
    click(int(sys.argv[1]), int(sys.argv[2]))
def move(x, y, disp=None):
    d = display.Display(disp); xtest.fake_input(d, X.MotionNotify, x=int(x), y=int(y)); d.sync(); d.close()
