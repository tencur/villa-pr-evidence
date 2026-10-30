F=/home/pi/fixwork; W=$F/wt-new-surface-tracks-dust-threshold-downsampled/spiral-fitting
mkdir -p $F/realdata/c341; cp rd_c341.py $F/realdata/c341/
cd $W; git stash push -q extract_surface_tracks.py tests/test_extract_surface_tracks.py
echo "== C341 on main (villa main 4eecd08f)"
$F/venv-data/bin/python $F/realdata/c341/rd_c341.py $W $F/realdata/pred_9um_z3456_y2048_x2048.npy 2>&1 | grep -v -i warn | tail -4
git stash pop -q
echo "== C341 on fix (fix/surface-tracks-dust-threshold-downsampled)"
$F/venv-data/bin/python $F/realdata/c341/rd_c341.py $W $F/realdata/pred_9um_z3456_y2048_x2048.npy 2>&1 | grep -v -i warn | tail -4
