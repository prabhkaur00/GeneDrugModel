import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import gc, json
import os
from datetime import datetime
from utils import print_gpu_memory, collate_fn
from logger import log_batch_predictions
import time
import psutil, threading

def log_bad_batch(epoch, batch_idx, batch, loss, log_dir="/mnt/data/logs"):
    os.makedirs(log_dir, exist_ok=True)
    prot_lens = list(batch['protein_embeddings'].shape)
    graph_nodes = [g.num_nodes for g in batch['drug_graphs'].to_data_list()]
    text_lens = None
    if batch['encoded_texts'] is not None:
        text_lens = batch['encoded_texts']['input_ids'].shape[1]
    bad = {
        "epoch": epoch,
        "batch_idx": batch_idx,
        "loss_shape": tuple(loss.shape),
        "loss_nan": torch.isnan(loss).sum().item(),
        "loss_inf": torch.isinf(loss).sum().item(),
        "gene_ids": batch['gene_ids'],
        "smiles": batch['smiles'],
        "prot_batch_shape": prot_lens,
        "graph_nodes": graph_nodes,
        "text_seq_len": text_lens
    }
    with open(os.path.join(log_dir, "bad_batches.jsonl"), "a") as f:
        f.write(json.dumps(bad) + "\n")


def eval_loss(model, loader, device):
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            protein = batch['protein_embeddings'].to(device)
            drug    = batch['drug_graphs'].to(device)
            txt     = batch['encoded_texts']
            if txt: txt = {k: v.to(device) for k, v in txt.items()}
            loss = model(protein, drug, txt)
            tot += loss.item(); n += 1
    model.train()
    return tot / max(n, 1)


def train_model(
    model,
    train_dataset,
    val_dataset,
    tokenizer,
    batch_size=32,
    num_epochs=10,
    learning_rate=5e-7,
    warmup_ratio=0.1,
    device="cuda" if torch.cuda.is_available() else "cpu",
    grad_accum_steps=1,
    max_len=25,
    log_predictions=True,
    log_frequency=10,
    checkpoint_interval=100,   # kept but unused now
    num_workers=1,
    prefetch_factor=1
):
    # dirs
    LOG_DIR = "/mnt/data/logs"
    CKPT_DIR = "/mnt/data/checkpoints"
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)

    log_file = f"training_loss_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    log_path = os.path.join(LOG_DIR, log_file)
    with open(log_path, "w") as f:
        f.write("step,epoch,batch,loss\n")

    # show trainables
    for n, p in model.named_parameters():
        if p.requires_grad:
            print(f"[Trainable] {n}: {p.numel()}")

    print("[Before empty_cache()]"); print_gpu_memory()
    torch.cuda.empty_cache()
    print("[After empty_cache()]");  print_gpu_memory()
    gc.collect()

    model.to(device)
    model.train()
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn,
        num_workers=num_workers, persistent_workers=True, pin_memory=True,
        prefetch_factor=prefetch_factor, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn,
        num_workers=num_workers, persistent_workers=True, pin_memory=True,
        prefetch_factor=prefetch_factor, drop_last=False,
    )
    print(f"[Info] Each epoch will have ~{len(train_loader)} batches.")

    # mem watch
    def mem_watch():
        p = psutil.Process(os.getpid())
        while True:
            print(f"[MEM RSS] {p.memory_info().rss / (1024**3):.2f} GiB")
            time.sleep(600)
    threading.Thread(target=mem_watch, daemon=True).start()

    # quick loader/step profile
    start = time.time()
    for _ in range(100):
        next(iter(train_loader))
    print(f"[Profiler] DataLoader avg: {(time.time()-start)/100:.3f} s")

    batch = next(iter(train_loader))
    t0, t1 = torch.cuda.Event(True), torch.cuda.Event(True)
    t0.record()
    loss = model(batch['protein_embeddings'].to(device),
                 batch['drug_graphs'].to(device),
                 batch['encoded_texts'])
    loss.backward()
    t1.record(); torch.cuda.synchronize()
    print(f"[Profiler] Step time: {t0.elapsed_time(t1)/1000:.3f} s")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    total_steps  = len(train_loader) * num_epochs // grad_accum_steps
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler = torch.amp.GradScaler()

    # ckpt helpers
    LAST_CKPT = os.path.join(CKPT_DIR, "last.ckpt")
    BEST_CKPT = os.path.join(CKPT_DIR, "best.ckpt")
    best_val = float("inf")

    def save_ckpt(path, epoch, step):
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "epoch": epoch,
            "step": step
        }, path)

    # train loop
    for epoch in range(num_epochs):
        print(f"\n[Epoch {epoch+1}] Starting")
        print_gpu_memory()
        total_loss, step_count = 0.0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        loss_print_interval = 1
        prediction_log_interval = max(len(train_loader) // 5, 1)

        try:
            for batch_idx, batch in enumerate(pbar):
                protein_data = batch['protein_embeddings'].to(device)
                drug_graph   = batch['drug_graphs'].to(device)
                target_text  = batch['encoded_texts']
                if target_text:
                    target_text = {k: v.to(device) for k, v in target_text.items()}

                loss = model(protein_data, drug_graph, target_text)

                if loss is None or loss.dim() != 0:
                    msg = f"[BadLoss] epoch={epoch+1} batch={batch_idx} loss_shape={None if loss is None else tuple(loss.shape)}"
                    print(msg, flush=True)
                    with open(log_path, "a") as f: f.write(msg + "\n")
                    log_bad_batch(epoch+1, batch_idx, batch, loss)
                    continue

                loss = loss / grad_accum_steps
                if not torch.isfinite(loss).item():
                    log_bad_batch(epoch+1, batch_idx, batch, loss)
                    continue

                scaler.scale(loss).backward()

                if (batch_idx + 1) % grad_accum_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()
                    avg_loss = total_loss / step_count if step_count > 0 else 0
                    with open(log_path, "a") as f:
                        f.write(f"{step_count},{epoch+1},{batch_idx+1},{avg_loss:.4f}\n")

                total_loss += loss.item() * grad_accum_steps
                step_count += 1

                if batch_idx % loss_print_interval == 0 and step_count > 0:
                    avg_loss = total_loss / step_count
                    pbar.set_postfix({"loss": f"{avg_loss:.4f}"})
                    log_line = f"[Epoch {epoch+1} | Batch {batch_idx}] Loss: {avg_loss:.4f}"
                    print(log_line, flush=True)
                    with open(log_path, "a") as f: f.write(log_line + "\n")

                if log_predictions and batch_idx % prediction_log_interval == 0:
                    model.eval()
                    try:
                        log_batch_predictions(model, batch, tokenizer, device, batch_idx, epoch)
                    except Exception as e:
                        print(f"[Prediction log error] {e}")
                    model.train()

                if batch_idx % (len(train_loader) // 5) == 0:
                    torch.cuda.empty_cache()

                if batch_idx % 10 == 0:
                    gc.collect()

        except Exception as e:
            err_msg = f"[Training loop exception] Epoch {epoch+1} | Step {step_count} | Error: {str(e)}"
            print(err_msg, flush=True)
            with open(log_path, "a") as f: f.write(err_msg + "\n")
            error_ckpt_path = os.path.join(CKPT_DIR, f"error_model_epoch{epoch}_step{step_count}.pt")
            torch.save(model.state_dict(), error_ckpt_path)
            raise e

        avg_epoch_loss = total_loss / step_count if step_count > 0 else 0
        print(f"[Epoch {epoch+1}] Finished. Avg loss: {avg_epoch_loss:.4f}")
        torch.cuda.empty_cache()

        # ---- validation + save last/best ----
        val_loss = eval_loss(model, val_loader, device)
        print(f"[Epoch {epoch+1}] Val loss: {val_loss:.4f}")
        save_ckpt(LAST_CKPT, epoch, step_count)
        if val_loss < best_val:
            best_val = val_loss
            save_ckpt(BEST_CKPT, epoch, step_count)
            print(f"[Best updated] val={val_loss:.4f} → {BEST_CKPT}")

    return model