import h5py, os, numpy as np

def h5_report(path, max_keys=5):
    sz = os.path.getsize(path) / (10**9)   # show in GB (SI)
    print(f"\n{path} ≈ {sz:.2f} GB")
    with h5py.File(path, "r") as f:
        keys = list(f.keys())[:max_keys]   # only first N keys
        for k in keys:
            d = f[k]
            nbytes = d.size * d.dtype.itemsize
            print(f"  {k:20s} shape={d.shape} dtype={d.dtype} ~{nbytes/2**20:.1f} MiB")
for i in range(6):
    h5_report(f"/mnt/data/gene_data/embeddings_{i}.h5")