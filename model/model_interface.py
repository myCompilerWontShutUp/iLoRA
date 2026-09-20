import inspect
import torch
import importlib
from torch import nn
from torch.nn import functional as F
import torch.optim.lr_scheduler as lrs

import pytorch_lightning as pl

from transformers import LlamaForCausalLM, LlamaTokenizer
import random
from pandas.core.frame import DataFrame
import os.path as op
import os
import csv
import json
import time
import subprocess
from optims import LinearWarmupCosineLRScheduler
import numpy as np
from .peft import get_peft_config, get_peft_model, get_peft_model_state_dict, LoraConfig, TaskType, PeftModel, MoeLoraConfig, MoeLoraModel
import pickle
from .router.nlpr import LambdaLayer, ResidualBlock, GateFunction, NLPRecommendationRouter, build_router
from .routing_utils import (
    parse_cluster_expert_mapping,
    validate_expert_mapping,
    build_onehot_gate,
    load_assignment_csv,
    gate_diagnostics,
)


# from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, LoraConfig, TaskType, PeftModel
class MInterface(pl.LightningModule):
    def __init__(self, 
                 **kargs):
        super().__init__()
        self.save_hyperparameters()
        self.load_llm(self.hparams.llm_path)
        
        if self.hparams.router == 'share':
            self.router = build_router()

        self.load_rec_model(self.hparams.rec_model_path)
        self.load_projector()
        self.gradient_storage = {}
        self._setup_routing_mode()

    def _setup_routing_mode(self):
        """Set up --routing_mode state. dynamic (default) touches nothing here; cluster_hard
        loads fixed cluster-assignment lookup tables. See
        experiments/IMPLEMENTATION_NOTES.md sections 4-6 for the design rationale."""
        self.routing_mode = getattr(self.hparams, 'routing_mode', 'dynamic')
        self._router_call_count = 0
        self._gate_export_rows = []
        self._cluster_mapping = None
        self._cluster_assignments = None

        if self.routing_mode == 'cluster_hard':
            if self.hparams.router != 'share':
                raise ValueError(
                    "--routing_mode cluster_hard requires --router share: the gate produced by "
                    "_resolve_gate is only wired into the MoE-LoRA layers on the 'share' path "
                    "(see CODEBASE_ANALYSIS.md section A/B)."
                )
            self._cluster_mapping = parse_cluster_expert_mapping(self.hparams.cluster_expert_mapping)
            num_clusters = max(self._cluster_mapping.keys()) + 1
            validate_expert_mapping(self._cluster_mapping, num_clusters=num_clusters, num_moe=self.hparams.num_moe)

            assignment_dir = self.hparams.cluster_assignment_dir
            self._cluster_assignments = {
                'train': load_assignment_csv(op.join(assignment_dir, 'train_assignments.csv')),
                'validation': load_assignment_csv(op.join(assignment_dir, 'validation_assignments.csv')),
                'test': load_assignment_csv(op.join(assignment_dir, 'test_assignments.csv')),
            }

            if self.hparams.cluster_model_path:
                if not op.exists(self.hparams.cluster_model_path):
                    raise FileNotFoundError(
                        f"--cluster_model_path {self.hparams.cluster_model_path} does not exist. "
                        "Run scripts/fit_kmeans.py first."
                    )

            if hasattr(self, 'router'):
                for p in self.router.parameters():
                    p.requires_grad = False
                print('[cluster_hard] router parameters frozen (requires_grad=False); '
                      'router.forward is never called in this routing_mode, see '
                      'experiments/IMPLEMENTATION_NOTES.md section 5.')

    def _resolve_gate(self, user_embeds, batch, split):
        """Single point where the gate fed into the MoE-LoRA layers is produced.

        dynamic: identical to the original `self.router(user_embeds)` call.
        cluster_hard: one-hot gate from a fixed, precomputed cluster assignment looked up by
        `sample_idx`, never recomputed on the fly (experiments/IMPLEMENTATION_NOTES.md section 4).
        """
        if self.routing_mode == 'dynamic':
            self._router_call_count += 1
            return self.router(user_embeds)

        if self.routing_mode == 'cluster_hard':
            assignments = self._cluster_assignments[split]
            sample_idx = batch['sample_idx'].tolist()
            expert_ids = []
            for idx in sample_idx:
                if idx not in assignments:
                    raise KeyError(f"No cluster assignment for sample_idx={idx} split={split} "
                                    f"in {self.hparams.cluster_assignment_dir}/{split}_assignments.csv")
                cluster_id = assignments[idx]
                expert_ids.append(self._cluster_mapping[cluster_id])
            return build_onehot_gate(expert_ids, self.hparams.num_moe,
                                      device=user_embeds.device, dtype=user_embeds.dtype)

        raise ValueError(f"Unknown routing_mode {self.routing_mode!r}")

    def _record_gate_export(self, sample_idx, gate_weights, user_embeds, split):
        """Buffer one batch's worth of gate/representation rows in memory. Flushed once per
        epoch by _flush_gate_export — no per-step CSV I/O (spec section 4)."""
        gate = gate_weights.detach().float().cpu().squeeze(1)  # [batch, num_moe]
        rep = user_embeds.detach().float().cpu().squeeze(1)    # [batch, sasrec_dim]
        idx = sample_idx.detach().cpu().tolist() if torch.is_tensor(sample_idx) else list(sample_idx)
        argmax_expert = gate.argmax(dim=-1).tolist()
        for i in range(gate.shape[0]):
            row = {
                'sample_id': idx[i],
                'user_id': 'NOT_AVAILABLE',
                'split': split,
                'argmax_expert': argmax_expert[i],
            }
            for k in range(gate.shape[1]):
                row[f'gate_{k}'] = gate[i, k].item()
            for d in range(rep.shape[1]):
                row[f'sasrec_{d}'] = rep[i, d].item()
            self._gate_export_rows.append(row)

    def _flush_gate_export(self, split):
        """Write results/<results_dir>/gates_<split>.csv from the in-memory buffer."""
        if not self._gate_export_rows:
            return
        os.makedirs(self.hparams.results_dir, exist_ok=True)
        out_path = op.join(self.hparams.results_dir, f'gates_{split}.csv')
        fieldnames = list(self._gate_export_rows[0].keys())
        with open(out_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self._gate_export_rows)

        gate_tensor = torch.tensor([[row[f'gate_{k}'] for k in range(self.hparams.num_moe)]
                                     for row in self._gate_export_rows])
        diag = gate_diagnostics(gate_tensor)
        if not diag['sum_ok'] or diag['has_nan'] or diag['has_inf']:
            print(f"[export_gates][WARNING] {out_path}: {diag['num_rows_violating']} row(s) "
                  f"failed gate validation (has_nan={diag['has_nan']}, has_inf={diag['has_inf']})")
        print(f"[export_gates] wrote {len(self._gate_export_rows)} rows to {out_path}")
        self._gate_export_rows = []

    def _write_metrics_json(self, prediction_valid_ratio, hr, metric, elapsed_seconds):
        """Write results/<results_dir>/metrics.json with the real, measured values only.
        No number here is invented; anything not measurable locally is 'NOT_RUN'."""
        try:
            commit_hash = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'], cwd=op.dirname(op.abspath(__file__))
            ).decode().strip()
        except Exception:
            commit_hash = 'NOT_RUN'

        try:
            import transformers as _tf
            transformers_version = _tf.__version__
        except Exception:
            transformers_version = 'NOT_RUN'

        try:
            from . import peft as _peft
            peft_version = getattr(_peft, '__version__', 'NOT_RUN')
        except Exception:
            peft_version = 'NOT_RUN'

        if torch.cuda.is_available():
            gpu_model = torch.cuda.get_device_name(0)
        else:
            gpu_model = 'NOT_RUN (no CUDA device visible)'

        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        metrics = {
            'git_commit_hash': commit_hash,
            'seed': self.hparams.seed,
            'dataset': self.hparams.dataset,
            'routing_mode': self.routing_mode,
            'num_moe': self.hparams.num_moe,
            'lora_r': self.hparams.lora_r,
            'epochs': self.hparams.max_epochs,
            'learning_rate': self.hparams.lr,
            'batch_size': self.hparams.batch_size,
            'accumulate_grad_batches': self.hparams.accumulate_grad_batches,
            'candidate_count': self.hparams.cans_num,
            'trainable_params': int(trainable_params),
            'llm_path_basename': os.path.basename(os.path.normpath(self.hparams.llm_path)),
            'test_prediction_valid': prediction_valid_ratio,
            'test_hr': hr,
            'metric': metric,
            'elapsed_seconds': elapsed_seconds,
            'gpu_model': gpu_model,
            'torch_version': torch.__version__,
            'transformers_version': transformers_version,
            'peft_version': peft_version,
            'pytorch_lightning_version': pl.__version__,
        }
        os.makedirs(self.hparams.results_dir, exist_ok=True)
        out_path = op.join(self.hparams.results_dir, 'metrics.json')
        with open(out_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"[metrics] wrote {out_path}")

    def forward(self, batch):
        targets = batch["tokens"].input_ids.masked_fill(
            batch["tokens"].input_ids == self.llama_tokenizer.pad_token_id, -100
        ) # [batch_size, max_len]
        
        targets = targets.masked_fill((batch["tokens"].token_type_ids == 0)[:,1:], -100)
        # targets = targets.masked_fill((batch["tokens"].token_type_ids == 0)[:,:], -100)
        
        input_embeds, user_embeds = self.wrap_emb(batch)

        if self.hparams.router == 'share':
            gate_weights = self._resolve_gate(user_embeds, batch, split='train')
            outputs = self.llama_model(
                inputs_embeds=input_embeds,
                attention_mask=batch["tokens"].attention_mask,
                return_dict=True,
                labels=targets,
                use_cache=False,
                user_embeds=user_embeds,
                gate_weights=gate_weights
            )
            return outputs

        outputs = self.llama_model(
            inputs_embeds=input_embeds,
            attention_mask=batch["tokens"].attention_mask,
            return_dict=True,
            labels=targets,
            use_cache=False,
            user_embeds=user_embeds
        )
        return outputs

    def generate(self, batch,temperature=0.8,do_sample=False,num_beams=1,max_gen_length=64,min_gen_length=1,repetition_penalty=1.0,length_penalty=1.0, num_return_sequences=1, split=None):
        input_embeds, user_embeds = self.wrap_emb(batch)
        if self.hparams.router == 'share':
            gate_weights = self._resolve_gate(user_embeds, batch, split=split)
            if self.hparams.export_gates and split is not None:
                self._record_gate_export(batch['sample_idx'], gate_weights, user_embeds, split)
            generate_ids = self.llama_model.generate(
                inputs_embeds=input_embeds,
                attention_mask=batch["tokens"].attention_mask,
                temperature=temperature,
                do_sample=do_sample,
                num_beams=num_beams,
                max_new_tokens=max_gen_length,
                min_new_tokens=min_gen_length,
                pad_token_id=self.llama_tokenizer.pad_token_id,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_return_sequences,
                user_embeds=user_embeds,
                gate_weights = gate_weights
            )
            output_text=self.llama_tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            outputs=[text.strip() for text in output_text]
            return outputs
            
        gate_weights = self.router(user_embeds)
        
        generate_ids = self.llama_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=batch["tokens"].attention_mask,
            temperature=temperature,
            do_sample=do_sample,
            num_beams=num_beams,
            max_new_tokens=max_gen_length,
            min_new_tokens=min_gen_length,
            pad_token_id=self.llama_tokenizer.pad_token_id,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            num_return_sequences=num_return_sequences,
            
            user_embeds=user_embeds, 
            gate_weights = gate_weights
            )
        output_text=self.llama_tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        outputs=[text.strip() for text in output_text]
        return outputs
    
    def capture_and_store_gradients(self):
        for name, param in self.llama_model.named_parameters():
            if "lora" in name and param.grad is not None:
                if name not in self.gradient_storage:
                    self.gradient_storage[name] = []
                self.gradient_storage[name].append(param.grad.clone().detach())
        
        if self.trainer.global_step % 10 == 0: 
            self.save_gradients_to_file()
            
    def save_gradients_to_file(self):
        directory = self.hparams.capture_dir
        if not os.path.exists(directory):
            os.makedirs(directory)
        file_path = os.path.join(directory, f'gradients_step_{self.trainer.global_step}.pkl')
        with open(file_path, 'wb') as f:
            pickle.dump(self.gradient_storage, f)
        self.gradient_storage = {} 
            
    def training_step(self, batch, batch_idx):
        if self.scheduler:
            self.scheduler.step(self.trainer.global_step, self.current_epoch, self.trainer.max_steps)
        if batch["flag"]:
            for name, param in self.projector.named_parameters():
                param.requires_grad = False
        else:
            for name, param in self.projector.named_parameters():
                param.requires_grad = True
        out = self(batch)
        loss = self.configure_loss(out)
        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('lr', self.scheduler.optimizer.param_groups[0]['lr'], on_step=True, on_epoch=True, prog_bar=True)
        self.log('global_step_num', self.trainer.global_step, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
            
    def on_validation_epoch_start(self):
        self.val_content={
            "generate":[],
            "real":[],
            "cans":[],
        }
        self._gate_export_rows = []
        self._router_call_count = 0

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        generate_output = self.generate(batch, split='validation')
        output=[]
        for i,generate in enumerate(generate_output):
            real=batch['correct_answer'][i]
            cans=batch['cans_name'][i]
            generate=generate.strip().split("\n")[0]
            output.append((generate,real,cans))
        return output

    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        for generate,real,cans in outputs:
            self.val_content["generate"].append(generate)
            self.val_content["real"].append(real)
            self.val_content["cans"].append(cans)

    def on_validation_epoch_end(self):
        df=DataFrame(self.val_content)
        if not os.path.exists(self.hparams.output_dir):
            os.makedirs(self.hparams.output_dir)
        df.to_csv(op.join(self.hparams.output_dir, 'valid.csv'))
        prediction_valid_ratio,hr=self.calculate_hr1(self.val_content)
        metric=hr*prediction_valid_ratio
        self.log('val_prediction_valid', prediction_valid_ratio, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_hr', hr, on_step=False, on_epoch=True, prog_bar=True)
        self.log('metric', metric, on_step=False, on_epoch=True, prog_bar=True)
        print(f"[routing_mode={self.routing_mode}] router forward calls this validation epoch: "
              f"{self._router_call_count}")
        if self.hparams.export_gates:
            self._flush_gate_export('validation')

    def on_test_epoch_start(self):
        self.test_content={
            "generate":[],
            "real":[],
            "cans":[],
        }
        self._gate_export_rows = []
        self._router_call_count = 0
        self._test_epoch_start_time = time.time()

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        generate_output = self.generate(batch, split='test')
        output=[]
        for i,generate in enumerate(generate_output):
            real=batch['correct_answer'][i]
            cans=batch['cans_name'][i]
            generate=generate.strip().split("\n")[0]
            output.append((generate,real,cans))
        return output
    
    def on_test_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        for generate,real,cans in outputs:
            self.test_content["generate"].append(generate)
            self.test_content["real"].append(real)
            self.test_content["cans"].append(cans)

    def on_test_epoch_end(self):
        df=DataFrame(self.test_content)
        if not os.path.exists(self.hparams.output_dir):
            os.makedirs(self.hparams.output_dir)
        df.to_csv(op.join(self.hparams.output_dir, 'test.csv'))
        prediction_valid_ratio,hr=self.calculate_hr1(self.test_content)
        metric=hr*prediction_valid_ratio

        self.log('test_prediction_valid', prediction_valid_ratio, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test_hr', hr, on_step=False, on_epoch=True, prog_bar=True)
        self.log('metric', metric, on_step=False, on_epoch=True, prog_bar=True)
        print(f"[routing_mode={self.routing_mode}] router forward calls this test epoch: "
              f"{self._router_call_count}")
        if self.hparams.export_gates:
            self._flush_gate_export('test')
        elapsed_seconds = time.time() - self._test_epoch_start_time
        self._write_metrics_json(prediction_valid_ratio, hr, metric, elapsed_seconds)

    def configure_optimizers(self):
        if hasattr(self.hparams, 'weight_decay'):
            weight_decay = self.hparams.weight_decay
        else:
            weight_decay = 0
        optimizer = torch.optim.Adam([
            {'params': self.projector.parameters(), 'lr': self.hparams.lr, 'weight_decay':weight_decay},
            
            {'params': self.router.parameters(), 'lr': self.hparams.lr * 0.3, 'weight_decay':weight_decay},
            
            {'params': [p for n, p in self.llama_model.named_parameters() if "gating" not in n], 'lr': self.hparams.lr},
            # {'params': [p for n, p in self.llama_model.named_parameters() if "gating" in n], 'lr': self.hparams.lr * 1, 'weight_decay':weight_decay}
           
            # {'params': self.llama_model.parameters(), 'lr': self.hparams.lr},
        ])
        
        
        for i, param_group in enumerate(optimizer.param_groups):
            print(f"Initial LR for group {i}: {param_group['lr']}")   
            total_params = sum(p.numel() for p in param_group['params'])
            print(f"Parameter Group {i}: {total_params} parameters")

        if self.hparams.lr_scheduler is None:
            return optimizer
        else:
            max_step = self.trainer.max_steps
            warmup_steps = max_step // 20
            print(f'max_step: {max_step}')
            print(f'warmup_steps: {warmup_steps}')
            if self.hparams.lr_scheduler == 'cosine':
                
                init_lr_list = [
                    self.hparams.lr,  
                    self.hparams.lr * 0.3, 
                    self.hparams.lr * 1
                ]
                min_lr_list = [
                    self.hparams.lr_decay_min_lr,  
                    self.hparams.lr_decay_min_lr * 0.3,  
                    self.hparams.lr_decay_min_lr * 1  
                ]
                warmup_start_lr_list = [
                    self.hparams.lr_warmup_start_lr, 
                    self.hparams.lr_warmup_start_lr * 0.3, 
                    self.hparams.lr_warmup_start_lr * 1 
                ]
                self.scheduler = LinearWarmupCosineLRScheduler(
                    optimizer=optimizer,
                    max_step=max_step,
                    min_lr_list=min_lr_list,
                    init_lr_list=init_lr_list,
                    warmup_steps=warmup_steps,
                    warmup_start_lr_list=warmup_start_lr_list
                )
                                
                
                for i, param_group in enumerate(optimizer.param_groups):
                    print(f"Initial LR for group {i}: {param_group['lr']}")   
                    total_params = sum(p.numel() for p in param_group['params'])
                    print(f"Parameter Group {i}: {total_params} parameters")
                    
                    
            else:
                self.scheduler = None
                raise ValueError('Invalid lr_scheduler type!')
            return optimizer

    def configure_loss(self, out, labels=None):
        loss = self.hparams.loss.lower()
        if loss == 'lm':
            return out.loss
        else:
            raise ValueError("Invalid Loss Type!")

    def on_save_checkpoint(self, checkpoint):
        if self.hparams.save == 'part':
            checkpoint.pop('optimizer_states')
            to_be_removed = []
            for key, value in checkpoint['state_dict'].items():
                try:
                    if not self.get_parameter(key).requires_grad:
                        to_be_removed.append(key)
                except AttributeError:
                    to_be_removed.append(key)
            for key in to_be_removed:
                checkpoint['state_dict'].pop(key)
        elif self.hparams.save == 'all':
            pass
        
    def load_llm(self, llm_path):
        print('Loading LLAMA')
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(llm_path, use_fast=False)
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
        self.llama_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.llama_tokenizer.padding_side = "right"
        self.llama_tokenizer.add_special_tokens({'additional_special_tokens': ['[PH]','[HistoryEmb]','[CansEmb]','[ItemEmb]']})
        self.llama_model = LlamaForCausalLM.from_pretrained(llm_path,torch_dtype=torch.bfloat16)
        self.llama_model.resize_token_embeddings(len(self.llama_tokenizer))
        if self.hparams.llm_tuning == 'lora':
            if self.hparams.peft_dir:
                self.llama_model = PeftModel.from_pretrained(self.llm_model, self.hparams.peft_dir, is_trainable=True)
            else:
                if self.hparams.peft_config:
                    peft_config = LoraConfig(**LoraConfig.from_json_file(self.hparams.peft_config))
                else:
                    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM,
                                             inference_mode=False,
                                             r=self.hparams.lora_r,
                                             lora_alpha=self.hparams.lora_alpha,
                                             lora_dropout=self.hparams.lora_dropout,
                                             target_modules=['k_proj', 'v_proj', 'q_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'])
                self.peft_config = peft_config
                self.llama_model = get_peft_model(self.llama_model, peft_config)
            self.llama_model.print_trainable_parameters()
        elif self.hparams.llm_tuning == 'freeze':
            for name, param in self.llama_model.named_parameters():
                param.requires_grad = False
        elif self.hparams.llm_tuning == 'freeze_lora':
            if self.hparams.peft_dir:
                self.llama_model = PeftModel.from_pretrained(self.llm_model, self.hparams.peft_dir, is_trainable=True)
            else:
                if self.hparams.peft_config:
                    peft_config = LoraConfig(**LoraConfig.from_json_file(self.hparams.peft_config))
                else:
                    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM,
                                             inference_mode=False,
                                             r=self.hparams.lora_r,
                                             lora_alpha=self.hparams.lora_alpha,
                                             lora_dropout=self.hparams.lora_dropout,
                                             target_modules=['k_proj', 'v_proj', 'q_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'])
                self.peft_config = peft_config
                self.llama_model = get_peft_model(self.llama_model, peft_config)
            for name, param in self.llama_model.named_parameters():
                param.requires_grad = False
            self.llama_model.print_trainable_parameters()
        elif self.hparams.llm_tuning == 'moelora':
            if self.hparams.peft_dir:
                self.llama_model = PeftModel.from_pretrained(self.llm_model, self.hparams.peft_dir, is_trainable=True)
            else:
                if self.hparams.peft_config:
                    peft_config = MoeLoraConfig(**MoeLoraConfig.from_json_file(self.hparams.peft_config))
                else:
                    peft_config = MoeLoraConfig(task_type=TaskType.CAUSAL_LM,
                                                inference_mode=False,
                                                r=self.hparams.lora_r,
                                                lora_alpha=self.hparams.lora_alpha,
                                                lora_dropout=self.hparams.lora_dropout,
                                                target_modules=['k_proj', 'v_proj', 'q_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
                                                num_moe=self.hparams.num_moe,
                                                gating=self.hparams.gating)
                self.peft_config = peft_config
                self.llama_model = get_peft_model(self.llama_model, peft_config)
                
                """for name, param in self.llama_model.named_parameters():
                    if "gating" not in name:
                        param.requires_grad = False"""
            self.llama_model.print_trainable_parameters()
        else:
            raise NotImplementedError()
 
        print('Loading LLAMA Done')

    def load_projector(self):
        name = self.hparams.model_name
        camel_name = ''.join([i.capitalize() for i in name.split('_')])
        try:
            Model = getattr(importlib.import_module(
                '.'+name, package=__package__), camel_name)
        except:
            raise ValueError(
                f'Invalid Module File Name or Invalid Class Name {name}.{camel_name}!')
        self.projector = self.instancialize(Model, rec_size=self.hparams.rec_size, llm_size=self.llama_model.config.hidden_size)

    def instancialize(self, Model, **other_args):
        class_args = inspect.getargspec(Model.__init__).args[1:]
        inkeys = self.hparams.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = getattr(self.hparams, arg)
        args1.update(other_args)
        # args1: args在hparams中有的部分
        return Model(**args1)

    def load_rec_model(self, rec_model_path):
        print('Loading Rec Model')
        self.rec_model = torch.load(rec_model_path, map_location="cpu")
        self.rec_model.eval()
        for name, param in self.rec_model.named_parameters():
            param.requires_grad = False
        print('Loding Rec model Done')

    def encode_items(self, seq):
        if self.hparams.rec_embed=="SASRec":
            item_rec_embs=self.rec_model.cacu_x(seq)
        elif self.hparams.rec_embed in ['Caser','GRU']:
            item_rec_embs=self.rec_model.item_embeddings(seq)
        item_txt_embs=self.projector(item_rec_embs)
        return item_txt_embs

    def encode_users(self, seq, len_seq):
        if self.hparams.rec_embed=="SASRec":
            user_rec_embs=self.rec_model.cacul_h(seq, len_seq)
        elif self.hparams.rec_embed in ['Caser','GRU']:
            user_rec_embs=self.rec_model.item_embeddings(seq)
        
        user_txt_embs=self.projector(user_rec_embs)    
        return user_rec_embs
    
    def embed_tokens(self, token_ids):
        embeds = self.llama_model.base_model.embed_tokens(token_ids)
        return embeds

    # batch -> embeds
    def wrap_emb(self, batch):
        input_embeds = self.llama_model.get_input_embeddings()(batch["tokens"].input_ids)
        
           
        
        his_token_id=self.llama_tokenizer("[HistoryEmb]", return_tensors="pt",add_special_tokens=False).input_ids.item()
        cans_token_id=self.llama_tokenizer("[CansEmb]", return_tensors="pt",add_special_tokens=False).input_ids.item()
        item_token_id=self.llama_tokenizer("[ItemEmb]", return_tensors="pt",add_special_tokens=False).input_ids.item()
         
        
        his_item_embeds = self.encode_items(batch["seq"])
        cans_item_embeds = self.encode_items(batch["cans"])
        item_embeds=self.encode_items(batch["item_id"])
        
        
        user_embeds=self.encode_users(batch["seq"], batch["len_seq"])

        for i in range(len(batch["len_seq"])):
            if (batch["tokens"].input_ids[i]==his_token_id).nonzero().shape[0]>0:
                idx_tensor=(batch["tokens"].input_ids[i]==his_token_id).nonzero().view(-1)
                for idx, item_emb in zip(idx_tensor,his_item_embeds[i,:batch["len_seq"][i].item()]):
                    input_embeds[i,idx]=item_emb
            if (batch["tokens"].input_ids[i]==cans_token_id).nonzero().shape[0]>0:
                idx_tensor=(batch["tokens"].input_ids[i]==cans_token_id).nonzero().view(-1)
                for idx, item_emb in zip(idx_tensor,cans_item_embeds[i,:batch["len_cans"][i].item()]):
                    input_embeds[i,idx]=item_emb
            if (batch["tokens"].input_ids[i]==item_token_id).nonzero().shape[0]>0:
                idx=(batch["tokens"].input_ids[i]==item_token_id).nonzero().item()
                input_embeds[i,idx]=item_embeds[i]
        
        return input_embeds, user_embeds
     
    def calculate_hr1(self,eval_content):
        correct_num=0
        valid_num=0
        total_num=0
        for i,generate in enumerate(eval_content["generate"]):
            real=eval_content["real"][i]
            cans=eval_content["cans"][i]
            total_num+=1
            generate=generate.strip().lower().strip()
            real=real.strip().lower().strip()
            cans=[item.strip().lower().strip() for item in cans]
            gen_cans_list=[]
            for cans_item in cans:
                if cans_item in generate:
                    gen_cans_list.append(cans_item)
            if len(gen_cans_list)==1:
                valid_num+=1
                if real == gen_cans_list[0]:
                    correct_num+=1
        valid_ratio=valid_num/total_num
        if valid_num>0:
            hr1=correct_num/valid_num
        else:
            hr1=0
        return valid_ratio,hr1
