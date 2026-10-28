"""Convert the Chikusei cube (128 bands, 2517 x 2335) into the repo's CASSI scene format.

Source: HSI_raw/chikusei/Chikusei_MATLAB/HyperspecVNIR_Chikusei_20140729.mat (MATLAB v7.3; h5py
layout (128, W, H) float64, chunked along H, so the cube is read in row blocks, never all at once).

Output (default ./Chikusei_128/):
    test/scene_rRRRR_cCCCC.mat   key 'img'        float32 (256, 256, 128) in [0, 1]   -> test_list.txt
    train/tile_rRRRR_cCCCC.mat   key 'img_expand' uint16 (512, 512, 128) = img*65535 -> train_list.txt
    meta.json                    scale, tile lists, valid fraction
    ../mask_256_128.mat, ../mask_128_128.mat   the 2-D CASSI mask of mask_256_28.mat repeated to 128 bands

Split: the top `--test_rows` rows (default 1024, the usual Chikusei SR split) are the test region;
only fully valid (no zero-padding) 256 x 256 tiles on a 256 grid are kept. Training tiles are
512 x 512, fully valid, on a 512 grid over the remaining rows with the last row/column of tiles
shifted inward so the whole region is covered.

Normalisation: one global scale for the whole cube (no per-band normalisation, so the spectral
shape is preserved) = the 99.99th percentile of valid pixels over all bands, then clip to [0, 1].
This gives a value distribution close to CAVE (median ~0.1, p99 ~0.6, max 1).
"""
import argparse
import json
import os

import h5py
import numpy as np
import scipy.io as sio

parser = argparse.ArgumentParser()
parser.add_argument('--src', default='./HSI_raw/chikusei/Chikusei_MATLAB/HyperspecVNIR_Chikusei_20140729.mat')
parser.add_argument('--out', default='./Chikusei_128')
parser.add_argument('--mask_src', default='./mask_256_28.mat')
parser.add_argument('--test_rows', default=1024, type=int)
parser.add_argument('--test_size', default=256, type=int)
parser.add_argument('--train_size', default=512, type=int)
parser.add_argument('--scale_pct', default=99.99, type=float)
parser.add_argument('--block', default=128, type=int, help='rows per read block')
opt = parser.parse_args()


def read_rows(ds, r0, r1):
    """(128, W, rows) -> (rows, W, 128) float32."""
    return np.ascontiguousarray(np.transpose(ds[:, :, r0:r1], (2, 1, 0)).astype(np.float32))


def grid(start, stop, size):
    """Tile origins covering [start, stop) with stride `size`; the last tile is shifted to end at stop."""
    o = list(range(start, stop - size + 1, size))
    if o and o[-1] + size < stop:
        o.append(stop - size)
    return o


if __name__ == '__main__':
    h = h5py.File(opt.src, 'r')
    ds = h['chikusei']
    C, W, H = ds.shape
    print('cube (C, W, H) =', ds.shape)

    # pass 1: valid mask and global scale from a pixel subsample
    valid = np.zeros((H, W), bool)
    samples = []
    for r0 in range(0, H, opt.block):
        blk = read_rows(ds, r0, min(H, r0 + opt.block))
        v = (blk != 0).any(axis=2)
        valid[r0:r0 + blk.shape[0]] = v
        samples.append(blk[v][::37].ravel())
    samples = np.concatenate(samples)
    scale = float(np.percentile(samples, opt.scale_pct))
    print('valid fraction %.3f, scale (p%.2f) = %.1f, max = %.1f' % (valid.mean(), opt.scale_pct, scale, samples.max()))
    del samples

    test_dir, train_dir = os.path.join(opt.out, 'test'), os.path.join(opt.out, 'train')
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(train_dir, exist_ok=True)

    # test tiles: top region, 256 grid, fully valid
    ts = opt.test_size
    test_tiles = [(r, c) for r in range(0, opt.test_rows - ts + 1, ts) for c in range(0, W - ts + 1, ts)
                  if valid[r:r + ts, c:c + ts].all()]
    # train tiles: remaining rows, 512 grid with inward-shifted last tiles, fully valid
    tr = opt.train_size
    train_tiles = [(r, c) for r in grid(opt.test_rows, H, tr) for c in grid(0, W, tr)
                   if valid[r:r + tr, c:c + tr].all()]
    print('%d test tiles (%dx%d), %d train tiles (%dx%d)' % (len(test_tiles), ts, ts, len(train_tiles), tr, tr))

    def write_tiles(tiles, size, folder, prefix, key, as_uint16):
        names = []
        # group by row origin so each row band of the cube is read once
        for r in sorted(set(t[0] for t in tiles)):
            blk = read_rows(ds, r, r + size)
            blk = np.clip(blk / scale, 0., 1.)
            for (rr, c) in [t for t in tiles if t[0] == r]:
                tile = blk[:, c:c + size, :]
                name = '%s_r%04d_c%04d.mat' % (prefix, rr, c)
                if as_uint16:
                    sio.savemat(os.path.join(folder, name), {key: np.round(tile * 65535).astype(np.uint16)}, do_compression=False)
                else:
                    sio.savemat(os.path.join(folder, name), {key: tile.astype(np.float32)}, do_compression=False)
                names.append(os.path.join(folder, name))
            print('  wrote row block %d (%d tiles)' % (r, len([t for t in tiles if t[0] == r])), flush=True)
        return names

    test_names = write_tiles(test_tiles, ts, test_dir, 'scene', 'img', as_uint16=False)
    train_names = write_tiles(train_tiles, tr, train_dir, 'tile', 'img_expand', as_uint16=True)
    with open(os.path.join(opt.out, 'test_list.txt'), 'w') as f:
        f.write('\n'.join(test_names) + '\n')
    with open(os.path.join(opt.out, 'train_list.txt'), 'w') as f:
        f.write('\n'.join(train_names) + '\n')

    # masks: the 2-D CASSI mask repeated over 128 bands (all 28 slices of mask_256_28 are identical)
    m2d = sio.loadmat(opt.mask_src)['mask'][:, :, 0]
    for s in sorted(set([256, ts, 128])):
        sio.savemat('./mask_%d_%d.mat' % (s, C), {'mask': np.repeat(m2d[:s, :s, None], C, axis=2)})
    json.dump({'src': opt.src, 'scale': scale, 'scale_pct': opt.scale_pct, 'valid_fraction': float(valid.mean()),
               'test_rows': opt.test_rows, 'test_tiles': test_tiles, 'train_tiles': train_tiles,
               'bands': int(C), 'wavelength_nm': '363-1018 (128 bands, Headwall Hyperspec-VNIR-C)'},
              open(os.path.join(opt.out, 'meta.json'), 'w'), indent=1)
    print('done ->', opt.out)
