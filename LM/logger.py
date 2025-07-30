import sys
import torch   

def log_batch_predictions(model, batch, tokenizer, device, batch_idx, epoch):
    print(f"EPOCH {epoch+1} - BATCH {batch_idx} - PREDICTION LOG")
    model.eval()
    try:
        with torch.no_grad():
            protein_embeddings = batch['protein_embeddings'].to(device, dtype=torch.bfloat16)
            drug_graphs = batch['drug_graphs'].to(device)
            target_text = batch['encoded_texts']
            try:
                predictions = model(
                    protein_embeddings,
                    drug_graphs,
                    target_text=None,
                    max_len=25
                )
            except RuntimeError as e:
                if "device-side assert" in str(e) or "CUDA" in str(e):
                    print(f"CUDA error during generation: {e}")
                    print("Skipping prediction logging for this batch.")
                    model.train()
                    return
                else:
                    raise e

            batch_size = protein_embeddings.size(0)

            for i in range(min(batch_size, 2)):
                print(f"\n--- SAMPLE {i+1} ---")
                prompt_template = "###Human: Describe the interaction between this protein [PROTEIN] and this drug [DRUG] ###Assistant:"
                print(f"PROMPT: {prompt_template}")
                if target_text and 'input_ids' in target_text:
                    actual_ids = target_text['input_ids'][i]
                    actual_ids = actual_ids[actual_ids != tokenizer.pad_token_id]
                    actual_ids = actual_ids[actual_ids != -100]
                    try:
                        actual_text = tokenizer.decode(actual_ids, skip_special_tokens=True)
                        print(f"ACTUAL: {actual_text}")
                    except Exception as e:
                        print(f"Error decoding actual text: {e}")
                if i < predictions.shape[0]:
                    pred_ids = predictions[i]
                    vocab_size = len(tokenizer)
                    valid_mask = (pred_ids >= 0) & (pred_ids < vocab_size)

                    if not valid_mask.all():
                        invalid_count = (~valid_mask).sum().item()
                        print(f"WARNING: Found {invalid_count} invalid token IDs in predictions")
                        pred_ids = pred_ids[valid_mask]

                    pred_ids = pred_ids[pred_ids != tokenizer.pad_token_id]
                    pred_ids = pred_ids[pred_ids != tokenizer.eos_token_id]
                    pred_ids = pred_ids.cpu()

                    try:
                        predicted_text = tokenizer.decode(pred_ids, skip_special_tokens=True)
                        if "###Assistant:" in predicted_text:
                            assistant_response = predicted_text.split("###Assistant:")[-1].strip()
                            print(f"PREDICTED: {assistant_response}")
                        else:
                            print(f"PREDICTED: {predicted_text}")

                    except Exception as decode_error:
                        print(f"DECODE ERROR: {decode_error}")
                        print(f"Attempting character-by-character decode...")
                        try:
                            decoded_tokens = []
                            for token_id in pred_ids[:20]:  # Only first 20 tokens
                                try:
                                    token = tokenizer.decode([token_id], skip_special_tokens=True)
                                    decoded_tokens.append(token)
                                except:
                                    decoded_tokens.append(f"<UNK_{token_id}>")
                            print(f"PREDICTED (partial): {''.join(decoded_tokens)}")
                        except Exception as e2:
                            print(f"Fallback decode also failed: {e2}")

                print("-" * 50)

    except Exception as e:
        print(f"Error during prediction logging: {e}")
        import traceback
        traceback.print_exc()