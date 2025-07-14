#!/usr/bin/env python3
import argparse, os, glob, pickle, h5py, pandas as pd, webdataset as wds
from concurrent.futures import ProcessPoolExecutor, as_completed
from utils import cache_to_pyg_data
from tqdm import tqdm

# ─────────────────────────── CPU detection ────────────────────────────
def detect_num_cpus(arg_override: int | None) -> int:
    if arg_override is not None:
        return max(1, arg_override)
    env_override = os.getenv("NUM_CPUS")
    if env_override and env_override.isdigit():
        return max(1, int(env_override))
    cpus = os.cpu_count()
    return max(1, cpus or 1)

parser = argparse.ArgumentParser()
parser.add_argument("--cpus", type=int, help="Number of worker processes")
args = parser.parse_args()
NUM_WORKERS = detect_num_cpus(args.cpus)
print(f"[info] Launching ProcessPoolExecutor with {NUM_WORKERS} workers")

# ───────────────────────────── paths ──────────────────────────────────
out_dir   = "/mnt/data/wds_shards"
tar_pat   = os.path.join(out_dir, "shard-%06d.tar")
progress  = os.path.join(out_dir, "progress.txt")
failed_log = os.path.join(out_dir, "failed_samples.txt")
processed_log = os.path.join(out_dir, "processed_rows.txt")

csv_path  = "/mnt/data/tier1.csv"
h5_path   = "/mnt/data/gene_data/embeddings_0.h5"
smiles_cache_path = "/mnt/data/smiles_cache.pkl"

# ───────────────────────────── data load ──────────────────────────────
df = pd.read_csv(csv_path)

# ─────────────── restart logic & next shard number ────────────────────
os.makedirs(out_dir, exist_ok=True)
next_shard_id = len(glob.glob(os.path.join(out_dir, "shard-*.tar")))

start_idx = 0
if os.path.exists(progress):
    start_idx = int(open(progress).read().strip()) + 1
print(f"[resume] starting at CSV row {start_idx}, shard {next_shard_id}")

# ─────────────────────── worker helpers ───────────────────────────────
def _init_worker(_h5_path, _smiles_pkl):
    global H5F, SMILES_CACHE
    H5F = h5py.File(_h5_path, "r")
    with open(_smiles_pkl, "rb") as f:
        SMILES_CACHE = pickle.load(f)

def _process_row(args):
    idx, gid, smiles, interaction = args
    try:
        phrase = interaction.split("[DRUG]", 1)[1].split("[PROTEIN]", 1)[0].strip()
    except Exception:
        snippet = interaction.replace("\n", " ")[:120]
        return (idx, None, f"FAILED_PHRASE_EXTRACTION|{snippet}")

    h5_key = f"genes_{gid}"
    if h5_key not in H5F:
        return (idx, None, "MISSING_H5")

    embed = H5F[h5_key][...]
    if embed.ndim == 2:
        embed = embed.mean(axis=0)
    embed = embed.astype("float32")

    try:
        drug_np = cache_to_pyg_data(SMILES_CACHE[smiles]).x.numpy()
    except Exception as e:
        return (idx, None, f"SMILES_GRAPH_FAIL:{e}")

    sample = {
        "__key__": f"{gid}_{idx}",
        "protein.npy": embed,
        "drug.npy":    drug_np,
        "interaction.txt": phrase,
        "smiles.txt":  smiles,
    }
    return (idx, sample)

# ───────────────────────── main loop ──────────────────────────────────
written = failed = 0
total_rows = len(df)

with open(failed_log, "a") as flog, open(processed_log, "a") as plog, \
     wds.ShardWriter(tar_pat, maxsize=1_073_741_824,
                     start_shard=next_shard_id) as shard_writer, \
     ProcessPoolExecutor(max_workers=NUM_WORKERS,
                         initializer=_init_worker,
                         initargs=(h5_path, smiles_cache_path)) as pool:

    # build argument tuples once
    job_args = [
        (
            i,
            str(int(df.loc[i, "GeneID"])),
            df.loc[i, "SMILES"],
            df.loc[i, "Interaction"],
        )
        for i in range(start_idx, total_rows)
    ]

    futures = {pool.submit(_process_row, arg): arg[0] for arg in job_args}
    pbar = tqdm(total=len(job_args), desc="Converting rows", unit="row")

    for fut in as_completed(futures):
        idx = futures[fut]
        try:
            result = fut.result()
        except Exception as e:
            flog.write(f"{idx},CRASH_IN_WORKER,{e}\n")
            failed += 1
            pbar.update()
            continue

        if len(result) == 3:                 # failure tuple
            _, _, reason = result
            if reason.startswith("FAILED_PHRASE_EXTRACTION"):
                tag, snippet = reason.split("|", 1)
                flog.write(f"{idx},{tag},{snippet}\n")
                print(f"[SKIP] row {idx}: phrase-extraction failed → \"{snippet}\"")
            else:
                flog.write(f"{idx},{reason}\n")
            failed += 1
        else:                                # success tuple
            _, sample = result
            shard_writer.write(sample)
            written += 1
            print(f"row {idx} of {total_rows} written")
            plog.write(f"{idx}\n")
            if idx % 1_000 == 0:
                with open(progress, "w") as p:
                    p.write(str(idx))

        pbar.update()

# ───────────────────────── cleanup / summary ──────────────────────────
if os.path.exists(progress):
    os.remove(progress)

print(f"✅  {written} new samples, ❌  {failed} failures  | shards in {out_dir}")