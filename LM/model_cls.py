import torch
from torch import nn
from torch.amp import autocast
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from gnn import GNN_graphpred
import time
import torch.nn.functional as F

class ProteinDrugLLMModelCls(torch.nn.Module):
    def __init__(self,
                 tokenizer,
                 llm_model_name="/mnt/data/vicuna-13b-v1.5",
                 gnn_ckpt=None,
                 freeze_gnn=True,
                 freeze_llm=True,
                 protein_hidden_dim=768,
                 lora_r=8,
                 lora_alpha=16,
                 lora_dropout=0.05,
                 num_labels=14,                
                 class_weights=None): 
     

        super().__init__()
        self.tokenizer = tokenizer
        self.protein_token_id = tokenizer.convert_tokens_to_ids("[PROTEIN]")
        self.drug_token_id = tokenizer.convert_tokens_to_ids("[DRUG]")
        print("PROTEIN token ID:", tokenizer.convert_tokens_to_ids("[PROTEIN]"))
        print("DRUG token ID:", tokenizer.convert_tokens_to_ids("[DRUG]"))
        self.vocab_size = len(tokenizer)
        self.freeze_llm = freeze_llm

        self.gnn = self.create_gnn(gnn_ckpt, freeze_gnn)
        self.drug_hidden_dim = 300
        self.protein_hidden_dim = protein_hidden_dim

        t0 = time.time(); print("Loading model+resizing tokenizer...")
        t1 = time.time(); print("Loading model...")
        base_llm = AutoModelForCausalLM.from_pretrained(
            llm_model_name,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=False,    # faster; needs host RAM
            local_files_only=True,
        )
        print(f"Model load: {time.time()-t1:.1f}s")
    
        t2 = time.time(); print("Resizing token embeddings...")
        base_llm.resize_token_embeddings(len(tokenizer))
        print(f"Resize: {time.time()-t2:.1f}s")

        print(f"TOTAL: {time.time()-t0:.1f}s")
        if freeze_llm:
            for param in base_llm.parameters():
                param.requires_grad = False
            base_llm.eval()
            self.llm = base_llm
        else:
            base_llm.gradient_checkpointing_enable()
            base_llm.enable_input_require_grads()
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM
            )
            self.llm = get_peft_model(base_llm, lora_config)

        self.llm_hidden_size = self.llm.config.hidden_size
        self.protein_proj = nn.Sequential(
            nn.Linear(self.protein_hidden_dim, self.llm_hidden_size),
            nn.LayerNorm(self.llm_hidden_size, eps=1e-5)
        )
        self.drug_proj = nn.Sequential(
            nn.Linear(self.drug_hidden_dim, self.llm_hidden_size),
            nn.LayerNorm(self.llm_hidden_size, eps=1e-5)
        )

        self.print_trainable_parameters()
        self.cls_head = nn.Sequential(
            nn.Linear(self.llm_hidden_size, self.llm_hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.llm_hidden_size // 2, num_labels)
        )

    def create_gnn(self, model_path, freeze):
        from gnn import GNN_graphpred
        emb_dim = 300
        gnn = GNN_graphpred(5, emb_dim, emb_dim, graph_pooling='attention', gnn_type='gcn')
        if model_path:
            gnn.from_pretrained(model_path)
        if freeze:
            for param in gnn.parameters():
                param.requires_grad = False
            gnn.eval()
        return gnn

    def print_trainable_parameters(self):
        print("\n[Trainable Parameters]")
        total = 0
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(f"- {name}: {param.numel()} params")
                total += param.numel()
        print(f"Total trainable parameters: {total}\n")

    def get_embeddings_layer(self):
        return self.llm.get_input_embeddings() if self.freeze_llm else self.llm.get_base_model().get_input_embeddings()

    def encode_drug(self, graph):
        device = next(self.parameters()).device
        graph = graph.to(device)
        graph_feat = self.gnn(graph)
        if graph_feat.ndim == 2:
            graph_feat = graph_feat.unsqueeze(1)
        return graph_feat

    def process_embeddings(self, protein_embeds, drug_embeds):
        device = next(self.parameters()).device
        protein_embeds = protein_embeds.to(device).to(torch.bfloat16)
        drug_embeds = drug_embeds.to(device).to(torch.bfloat16)
        if protein_embeds.ndim == 3:
            print("[PKW] Warning: protein_embeds has 3D shape, mean pooling will be applied")
            protein_embeds = protein_embeds.mean(dim=1)
        if drug_embeds.ndim == 3:
            print("[PKW] Warning: drug_embeds has 3D shape, mean pooling will be applied")
            drug_embeds = drug_embeds.mean(dim=1)

        protein_proj = self.protein_proj(protein_embeds)
        drug_proj = self.drug_proj(drug_embeds)
        protein_proj_before = protein_proj.clone().detach()
        drug_proj_before    = drug_proj.clone().detach()
        with torch.no_grad():
            ref_std = self.get_embeddings_layer().weight.std().item()

        protein_proj = F.layer_norm(protein_proj, protein_proj.shape[-1:]) * ref_std
        drug_proj    = F.layer_norm(drug_proj,    drug_proj.shape[-1:])    * ref_std
        print(f"[ScaleMatch] prot std {protein_proj_before.std():.4e}→{protein_proj.std():.4e}, "
            f"drug std {drug_proj_before.std():.4e}→{drug_proj.std():.4e}, "
            f"ref std {ref_std:.4e}")

        return protein_proj, drug_proj


    def forward(self, protein_data, drug_graph, target_text=None, max_len=25):
        with autocast(device_type="cuda", dtype=torch.bfloat16):
            device = next(self.parameters()).device
            protein_data = protein_data.to(device)
            drug_graph = drug_graph.to(device)
            drug_embed = self.encode_drug(drug_graph)
            protein_proj, drug_proj = self.process_embeddings(protein_data, drug_embed)
            batch_size = protein_data.size(0)
            prompt_template = "###Human: Describe the interaction between this protein [PROTEIN] and this drug [DRUG] ###Assistant:"
            prompts = [prompt_template] * batch_size
            tokenized = self.tokenizer(prompts, return_tensors='pt', padding=True, truncation=True).to(device)
            input_ids = tokenized['input_ids']
            attention_mask = tokenized['attention_mask']
            embed_layer = self.get_embeddings_layer()
            inputs_embeds = embed_layer(input_ids).clone()

            for i in range(batch_size):
                p_pos = (input_ids[i] == self.protein_token_id).nonzero(as_tuple=True)[0]
                d_pos = (input_ids[i] == self.drug_token_id).nonzero(as_tuple=True)[0]
                if len(p_pos) == 0 or len(d_pos) == 0:
                    msg = f"[TokenInjectionWarning] Sample {i} missing token(s) | protein_tokens={len(p_pos)} drug_tokens={len(d_pos)}"
                    print(msg)
                    with open("logs/error_log.txt", "a") as f: f.write(msg + "\n")
                if len(p_pos) > 0:
                    inputs_embeds[i, p_pos[0]] = protein_proj[i]
                if len(d_pos) > 0:
                    inputs_embeds[i, d_pos[0]] = drug_proj[i]

            outputs = self.llm(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_hidden_states=False,
                return_dict=True
            )
            seq_last = outputs.last_hidden_state  # [B, T, H]

            mask = attention_mask.unsqueeze(-1)   # [B, T, 1]
            pooled = (seq_last * mask).sum(1) / mask.sum(1)  # [B, H]

            logits = self.cls_head(pooled)        # [B, C]

            if labels is not None:
                labels = labels.to(device).float()          # multi-hot
                if self.class_weights is not None:
                    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=self.class_weights.to(device))
                else:
                    loss_fn = torch.nn.BCEWithLogitsLoss()
                loss = loss_fn(logits, labels)
                return loss, logits
            else:
                probs = torch.sigmoid(logits)
                return probs

