"""npz (Jev labels) -> binary labels file for the CUDA trainer.
Format: int32 hdr[8] {0x4c41424c, 1, n, K, nfacets, split, 0, 0}; int32 fsize[nf]; int64 end[n]; int32 len[n]; f32 y[n*K]
usage: make_labels_bin.py in.npz out.bin split(0=train,1=val) [shuffle_seed]"""
import numpy as np, sys
z = np.load(sys.argv[1]); out = sys.argv[2]; split = int(sys.argv[3])
Y = np.concatenate([z["topic"], z["fmt"]], axis=1).astype(np.float32)
Y /= np.maximum(Y[:, :160].sum(1, keepdims=True), 1e-9) * np.r_[np.ones(160), np.zeros(40)][None] + np.maximum(Y[:, 160:].sum(1, keepdims=True), 1e-9) * np.r_[np.zeros(160), np.ones(40)][None]
if len(sys.argv) > 4:   # control: shuffle label rows across passages (destroys meaning, keeps marginals)
    Y = Y[np.random.default_rng(int(sys.argv[4])).permutation(len(Y))]
end = z["end"].astype(np.int64); ln = (z["end"] - z["start"]).astype(np.int32)
with open(out, "wb") as f:
    np.array([0x4c41424c, 1, len(Y), Y.shape[1], 2, split, 0, 0], dtype=np.int32).tofile(f)
    np.array([160, 40], dtype=np.int32).tofile(f); end.tofile(f); ln.tofile(f); Y.tofile(f)
print(out, Y.shape, "len mean", ln.mean(), "topic argmax classes used", len(np.unique(Y[:, :160].argmax(1))), "fmt", len(np.unique(Y[:, 160:].argmax(1))))
