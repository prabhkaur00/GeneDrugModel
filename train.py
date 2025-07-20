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

def log_bad_batch(batch_idx, epoch, batch, loss, log_dir="/mnt/data/logs"):
    os.makedirs(log_dir, exist_ok=True)
    bad = {
        "epoch": epoch,
        "batch_idx": batch_idx,
        "loss_shape": tuple(loss.shape),
        "loss_nan": torch.isnan(loss).sum().item(),
        "loss_inf": torch.isinf(loss).sum().item(),
        "sample_ids": batch["sample_ids"],
        "prot_lens": [p.shape[0] for p in batch["protein_embeddings"]],
        "graph_nodes": [g.num_nodes for g in batch["drug_graphs"].to_data_list()],
        "text_lens": batch["encoded_texts"]["input_ids"].shape[1]
                     if batch["encoded_texts"] else None
    }
    with open(os.path.join(log_dir, "bad_batches.jsonl"), "a") as f:
        f.write(json.dumps(bad) + "\n")

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
    stage_name="",
    checkpoint_interval=100
):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"[Trainable] {name}: {param.numel()}")

    log_file = f"training_loss_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    log_path = os.path.join("logs", log_file)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    with open(log_path, "w") as f:
        f.write("step,epoch,batch,loss\n")

    print("[Before empty_cache()]")
    print_gpu_memory()
    torch.cuda.empty_cache()
    print("[After empty_cache()]")
    print_gpu_memory()
    gc.collect()

    model.train()
    model = model.to(device)
    if hasattr(model, "gradient_checkpointing_enable"):
        print("[Enabling gradient checkpointing]")
        model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        print("[Enabling input require grads]")
        model.enable_input_require_grads()
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )
    start = time.time()
    for _ in range(100):
        next(iter(train_loader))
    print(f"[Profiler] Loader avg: {(time.time()-start)/100:.3f} s")

    batch = next(iter(train_loader))
    t0 = torch.cuda.Event(True); t1 = torch.cuda.Event(True)
    t0.record()
    ## for profiling purposes, we run a single batch
    loss = model(batch['protein_embeddings'].to(device),
                batch['drug_graphs'].to(device),
                batch['encoded_texts'])
    loss.backward()
    t1.record(); torch.cuda.synchronize()
    print(f"[Profiler] Step time: {t0.elapsed_time(t1)/1000:.3f} s")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_loader) * num_epochs // grad_accum_steps
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler = torch.amp.GradScaler()
    for epoch in range(num_epochs):
        print(f"\n[Epoch {epoch+1}] Starting")
        print_gpu_memory()
        total_loss = 0.0
        step_count = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        loss_print_interval = 1
        prediction_log_interval = max(len(train_loader) // 5, 1)  # ~5x per epoch
        try:
            for batch_idx, batch in enumerate(pbar):
                protein_data = batch['protein_embeddings'].to(device)
                drug_graph = batch['drug_graphs'].to(device)
                target_text = batch['encoded_texts']
                if target_text:
                    target_text = {k: v.to(device) for k, v in target_text.items()}

                
                loss = model(protein_data, drug_graph, target_text)
                if loss is None or loss.dim() != 0:
                    msg = f"[BadLoss] epoch={epoch+1} batch={batch_idx} loss_shape={None if loss is None else tuple(loss.shape)}"
                    print(msg, flush=True)
                    with open(log_path, "a") as f:
                        f.write(msg + "\n")
                    log_bad_batch(epoch+1, batch_idx, batch, loss)
                    continue
                loss = loss / grad_accum_steps
                
                if not torch.isfinite(loss).item():
                    log_bad_batch(epoch+1, batch_idx, batch, loss)                       # see step 6
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
                
                del protein_data, drug_graph, target_text
                torch.cuda.empty_cache()
                gc.collect()

                total_loss += loss.item() * grad_accum_steps
                step_count += 1

                if batch_idx % loss_print_interval == 0 and step_count > 0:
                    avg_loss = total_loss / step_count
                    pbar.set_postfix({"loss": f"{avg_loss:.4f}"})
                    log_line = f"[Epoch {epoch+1} | Batch {batch_idx}] Loss: {avg_loss:.4f}"
                    print(log_line, flush=True)
                    with open(log_path, "a") as f:
                        f.write(log_line + "\n")


                if log_predictions and batch_idx % prediction_log_interval == 0:
                    model.eval()
                    try:
                        log_batch_predictions(model, batch, tokenizer, device, batch_idx, epoch)
                    except Exception as e:
                        print(f"[Prediction log error] {e}")
                    model.train()

                if batch_idx % (len(train_loader) // 5) == 0:
                    torch.cuda.empty_cache()

                if (batch_idx + 1) % checkpoint_interval == 0:
                    ckpt_path = f"checkpoints/ckpt_epoch{epoch}_step{step_count}_{stage_name}_batch{batch_idx + 1}.pt"
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'epoch': epoch,
                        'step': step_count
                    }, ckpt_path)
                    print(f"[Checkpoint saved] {ckpt_path}")
                
                if batch_idx % 10 == 0:
                    gc.collect()

        except Exception as e:
            err_msg = f"[Training loop exception] Epoch {epoch+1} | Step {step_count} | Error: {str(e)}"
            print(err_msg, flush=True)
            with open(log_path, "a") as f:
                f.write(err_msg + "\n")
            torch.save(model.state_dict(), f"checkpoints/error_model_epoch{epoch}_step{step_count}.pt")
            raise e

        avg_epoch_loss = total_loss / step_count if step_count > 0 else 0
        print(f"[Epoch {epoch+1}] Finished. Avg loss: {avg_epoch_loss:.4f}")
        torch.cuda.empty_cache()

        final_ckpt = f"checkpoints/ckpt_epoch{epoch}_final_{stage_name}.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'epoch': epoch,
            'step': step_count
        }, final_ckpt)
        print(f"[Checkpoint saved] {final_ckpt}")

    return model

import torch
from torch.utils.data import DataLoader
from utils import collate_fn          # same collate you already have

def decode(tokenizer, ids):
    """Turn a tensor of token-ids into a clean string."""
    txt = tokenizer.decode(ids, skip_special_tokens=True)
    return txt.replace("[PROTEIN]", "").replace("[DRUG]", "").strip()

@torch.no_grad()
def print_predictions(model, batch, tokenizer, device, n=3):
    # take first n items for display
    prot  = batch["protein_embeddings"][:n].to(device)
    graph = batch["drug_graphs"][:n].to(device)

    # forward pass in eval mode
    model.eval()
    
    generated_ids = model.generate(prot, graph, max_new_tokens=15)
    model.train()

    expected_ids = batch["encoded_texts"][:n]
    for i in range(len(expected_ids)):
        exp = decode(tokenizer, expected_ids[i].cpu())
        pred = decode(tokenizer, generated_ids[i].cpu())
        print(f"  · expected: «{exp}»")
        print(f"    predicted: «{pred}»\n")

def train_minimal(
    model,
    train_data,
    tokenizer,
    device=None,
    batch_size=8,
    epochs=1,
    lr=1e-4,
    show_every=5,
    preview_k=3,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    loader  = DataLoader(train_data,
                         batch_size=batch_size,
                         shuffle=True,
                         collate_fn=collate_fn,
                         num_workers=0,
                         pin_memory=False)

    optim   = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler  = torch.amp.GradScaler()

    for epoch in range(epochs):
        running = 0.0
        for b, batch in enumerate(loader, 1):
            optim.zero_grad()
            loss = model(batch["protein_embeddings"].to(device),
                            batch["drug_graphs"].to(device),
                            batch["encoded_texts"])
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()

            running += loss.item()
            print(f"epoch {epoch+1}  batch {b}  loss {loss.item():.4f}")

            if b % show_every == 0:
                print_predictions(model, batch, tokenizer, device, n=preview_k)

        print(f"epoch {epoch+1} finished – avg loss {(running/len(loader)):.4f}")

    torch.save(model.state_dict(), "model_min.pt")
    print("✓ training complete")

def validate_model(model, val_dataset, tokenizer, device):
    model.eval()
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True
    )
    total_val_loss = 0.0
    count = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="[Validation]"):
            try:
                protein_data = batch['protein_embeddings'].to(device)
                drug_graph = batch['drug_graphs'].to(device)
                target_text = batch['encoded_texts']
                if target_text:
                    target_text = {k: v.to(device) for k, v in target_text.items()}
                loss = model(protein_data, drug_graph, target_text)
                total_val_loss += loss.item()
                count += 1
            except Exception as e:
                print(f"[Validation error] {e}")
                continue
    avg_val_loss = total_val_loss / count if count > 0 else float("inf")
    val_log = f"[Validation] Avg loss: {avg_val_loss:.4f}"
    print(val_log, flush=True)
    with open("logs/validation_loss_log.txt", "a") as f:
        f.write(val_log + "\n")
    model.train()
    return avg_val_loss