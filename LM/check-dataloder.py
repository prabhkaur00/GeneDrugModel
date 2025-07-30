from torch.utils.data import DataLoader
import time

dataset = ProteinDrugDataset(h5_file, csv_subset, graph_cache, tokenizer)
loader = DataLoader(dataset, batch_size=32, num_workers=4, collate_fn=collate_fn)

start = time.time()
for batch in loader:
    pass
print("Dataset iteration time:", time.time() - start)