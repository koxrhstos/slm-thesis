import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
import torch
torch._dynamo.config.disable = True
torch._dynamo.config.suppress_errors = True
import json
import random
import numpy as np
from typing import List, Dict, Tuple, Any
from datetime import datetime

import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    TrainerCallback,
)
from sklearn.model_selection import train_test_split
import logging
from datetime import datetime
from tqdm import tqdm


# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("trainer")


class GreekLegalDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = f"Ερώτηση: {item['input']}\nΑπάντηση: {item['output']}{self.tokenizer.eos_token}"
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].flatten()
        attention_mask = encoding['attention_mask'].flatten()

        # Mask padding tokens in labels to -100 so they don't contribute to the loss
        labels = input_ids.clone()
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else -100
        if pad_id != -100:
            labels = torch.where(attention_mask == 1, labels, torch.full_like(labels, -100))
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


class SimplifiedMetricsComputer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.tensorboard_writer = None

    def set_tensorboard_writer(self, writer):
        self.tensorboard_writer = writer

    def normalize_text(self, text: str) -> str:
        if not text or not isinstance(text, str):
            return ""
        text = ' '.join(text.split())
        text = (text
                .replace('ά', 'α').replace('έ', 'ε').replace('ή', 'η')
                .replace('ί', 'ι').replace('ό', 'ο').replace('ύ', 'υ').replace('ώ', 'ω'))
        text = text.lower()
        return text.strip()

    def compute_perplexity(self, eval_preds) -> Dict[str, float]:
        predictions, labels = eval_preds
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        if not isinstance(predictions, torch.Tensor):
            predictions = torch.tensor(predictions)
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels)
        if predictions.device != labels.device:
            predictions = predictions.to(labels.device)
        if len(predictions.shape) != 3:
            return {"perplexity": float('inf')}
        shift_logits = predictions[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else -100
        mask = (shift_labels != -100) & (shift_labels != pad_id)
        if mask.sum() == 0:
            return {"perplexity": float('inf')}
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        losses = losses.view(shift_labels.shape)
        masked_losses = losses * mask.float()
        mean_loss = masked_losses.sum() / mask.sum()
        perplexity = torch.exp(torch.clamp(mean_loss, max=20))
        return {"perplexity": float(perplexity.item())}

    def exact_match_score(self, predictions: List[str], references: List[str]) -> float:
        if not predictions or not references:
            return 0.0
        exact_matches = 0
        valid = 0
        for pred, ref in zip(predictions, references):
            if not pred or not ref:
                continue
            pred_norm = self.normalize_text(pred)
            ref_norm = self.normalize_text(ref)
            if pred_norm and ref_norm:
                valid += 1
                if pred_norm == ref_norm:
                    exact_matches += 1
        return exact_matches / valid if valid > 0 else 0.0

    def token_overlap_stats(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        if not predictions or not references:
            return {"f1": 0.0, "precision": 0.0, "recall": 0.0, "fpr": 0.0, "fnr": 0.0}
        total_p = total_r = total_f1 = total_fpr = total_fnr = 0.0
        n = 0
        for pred, ref in zip(predictions, references):
            if pred is None or ref is None:
                continue
            pred_tokens = set(self.normalize_text(pred).split())
            ref_tokens = set(self.normalize_text(ref).split())
            if not pred_tokens and not ref_tokens:
                continue
            n += 1
            common = pred_tokens & ref_tokens
            precision = len(common) / len(pred_tokens) if len(pred_tokens) > 0 else 0.0
            recall = len(common) / len(ref_tokens) if len(ref_tokens) > 0 else 0.0
            f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
            # Define false rates on token sets
            fpr = 1.0 - precision if len(pred_tokens) > 0 else 0.0
            fnr = 1.0 - recall if len(ref_tokens) > 0 else 0.0
            total_p += precision
            total_r += recall
            total_f1 += f1
            total_fpr += fpr
            total_fnr += fnr
        if n == 0:
            return {"f1": 0.0, "precision": 0.0, "recall": 0.0, "fpr": 0.0, "fnr": 0.0}
        return {
            "f1": total_f1 / n,
            "precision": total_p / n,
            "recall": total_r / n,
            "fpr": total_fpr / n,
            "fnr": total_fnr / n,
        }

    def random_accuracy_baseline(self, references: List[str]) -> float:
        vals = []
        for ref in references:
            tokens = self.normalize_text(ref).split()
            vals.append(1.0 / max(1, len(tokens)))
        return float(np.mean(vals)) if vals else 0.0

    def compute_eval_text_metrics(self, decoded_preds: List[str], decoded_labels: List[str]) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        metrics["accuracy"] = self.exact_match_score(decoded_preds, decoded_labels)
        ov = self.token_overlap_stats(decoded_preds, decoded_labels)
        metrics["f1"] = ov["f1"]
        metrics["precision"] = ov["precision"]
        metrics["recall"] = ov["recall"]
        metrics["false_positive_rate"] = ov["fpr"]
        metrics["false_negative_rate"] = ov["fnr"]
        metrics["random_accuracy"] = self.random_accuracy_baseline(decoded_labels)
        return metrics


def decode_labels(tokenizer, labels: torch.Tensor) -> List[str]:
    labels = labels.clone()
    labels[labels == -100] = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    return tokenizer.batch_decode(labels, skip_special_tokens=True)


def decode_argmax_predictions(tokenizer, logits: torch.Tensor) -> List[str]:
    pred_ids = torch.argmax(logits, dim=-1)
    return tokenizer.batch_decode(pred_ids, skip_special_tokens=True)

def build_compute_metrics(tokenizer, metrics_computer: SimplifiedMetricsComputer):
    def _compute(eval_preds):
        predictions, labels = eval_preds

        # transform to torch tensors
        if not isinstance(predictions, torch.Tensor):
            predictions = torch.tensor(predictions)
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels)

        #if already have ids from preprocess_logits_for_metrics (2D), keep them
        # else (3D logits) take argmax here (fallback).
        if predictions.ndim == 3:
            pred_ids = predictions.argmax(dim=-1)
        else:
            pred_ids = predictions

        # decode predictions and labels
        decoded_preds = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)

        lab = labels.clone()
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        lab[lab == -100] = pad_id
        decoded_labels = tokenizer.batch_decode(lab, skip_special_tokens=True)

        # Text metrics only (accuracy, f1, precision, recall, fpr, fnr, random_accuracy)
        return {k: float(v) for k, v in metrics_computer.compute_eval_text_metrics(decoded_preds, decoded_labels).items()}
    return _compute


class EnhancedTrainer(Trainer):
    def __init__(self, *args, metrics_computer: SimplifiedMetricsComputer = None, tokenizer=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics_computer = metrics_computer
        self.step_count = 0
        self.best_metrics: Dict[str, Any] = {}
        self.tokenizer = tokenizer
        
        #create unique run name with timestamp and config info
        log_dir = self.args.logging_dir
        os.makedirs(log_dir, exist_ok=True)
        self.tensorboard_writer = SummaryWriter(log_dir=log_dir)
        if self.metrics_computer:
            self.metrics_computer.set_tensorboard_writer(self.tensorboard_writer)

    def preprocess_logits_for_metrics(self, logits, labels):
        logits = logits[0] if isinstance(logits, tuple) else logits
        return logits.argmax(dim=-1)

    def compute_metrics(self, eval_preds):
        predictions, labels = eval_preds
        metrics: Dict[str, float] = {}
        if self.metrics_computer:
            metrics.update(self.metrics_computer.compute_perplexity(eval_preds))
        try:
            if isinstance(predictions, tuple):
                predictions = predictions[0]
            if not isinstance(predictions, torch.Tensor):
                predictions = torch.tensor(predictions)
            if not isinstance(labels, torch.Tensor):
                labels = torch.tensor(labels)
            decoded_preds = decode_argmax_predictions(self.tokenizer, predictions)
            decoded_labels = decode_labels(self.tokenizer, labels)
            text_metrics = self.metrics_computer.compute_eval_text_metrics(decoded_preds, decoded_labels)
            for k, v in text_metrics.items():
                metrics[k] = float(v)
        except Exception as e:
            logger.warning(f"Failed to compute text metrics on eval: {e}")
        return metrics

    def _remap_eval_keys_with_slash(self, logs: Dict[str, Any]) -> Dict[str, Any]:
        # Convert eval_* to eval/* names as requested
        remapped = {}
        for key, value in logs.items():
            if key.startswith("eval_"):
                metric_name = key[len("eval_"):]
                remapped[f"eval/{metric_name}"] = value
        return remapped

    def log(self, logs, start_time=None):
        self.step_count += 1
        if 'train_loss' in logs:
            logs['step'] = self.step_count
            logs['lr'] = self.get_lr()
            if torch.cuda.is_available():
                logs['gpu_memory_allocated'] = torch.cuda.memory_allocated() / 1024**3
                logs['gpu_memory_reserved'] = torch.cuda.memory_reserved() / 1024**3
            self.tensorboard_writer.add_scalar("Training/loss", logs['train_loss'], self.state.global_step)
            self.tensorboard_writer.add_scalar("Training/learning_rate", logs['lr'], self.state.global_step)

        # Mirror eval_* keys to eval/* keys for logging
        eval_slash_logs = self._remap_eval_keys_with_slash(logs)
        for k, v in eval_slash_logs.items():
            # Also write to TensorBoard
            self.tensorboard_writer.add_scalar(k, v, self.state.global_step)
            logs.setdefault(k, v)

        if 'eval_loss' in logs:
            self.tensorboard_writer.add_scalar("Evaluation/loss", logs['eval_loss'], self.state.global_step)
            if 'eval_perplexity' in logs:
                self.tensorboard_writer.add_scalar("Evaluation/perplexity", logs['eval_perplexity'], self.state.global_step)
                # Keep slash-style too
                self.tensorboard_writer.add_scalar("eval/perplexity", logs['eval_perplexity'], self.state.global_step)

        if start_time is not None:
            super().log(logs, start_time)
        else:
            super().log(logs)

    def get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']
        return 0.0

    def __del__(self):
        if hasattr(self, 'tensorboard_writer'):
            self.tensorboard_writer.close()


class TrainMetricsCallback(TrainerCallback):
    def __init__(self, tokenizer, metrics_computer: SimplifiedMetricsComputer, sample_size: int = 64, generate_max_new_tokens: int = 64):
        self.tokenizer = tokenizer
        self.metrics_computer = metrics_computer
        self.sample_size = sample_size
        self.generate_max_new_tokens = generate_max_new_tokens

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.total_flos is not None:
            logs = logs or {}
            logs["train/total_flos"] = float(state.total_flos)
        return

    def on_step_end(self, args, state, control, **kwargs):
        trainer: Trainer = kwargs.get('trainer')
        if trainer is None or trainer.train_dataset is None:
            return

        # Evaluate only at eval_steps intervals
        if state.global_step % max(1, args.eval_steps) != 0:
            return

        try:
            dataset = trainer.train_dataset
            indices = random.sample(range(len(dataset)), k=min(self.sample_size, len(dataset)))
            prompts, references = [], []
            for idx in indices:
                item = dataset.data[idx]
                prompts.append(f"Ερώτηση: {item['input']}\nΑπάντηση:")
                references.append(item['output'])

            model = trainer.model
            model.eval()
            decoded_preds = []
            gen_cfg = dict(
                max_new_tokens=self.generate_max_new_tokens,
                temperature=0.0,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )

            with torch.inference_mode():
                for prompt in prompts:
                    inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=getattr(trainer.args, 'model_max_length', 512))
                    inputs = {k: v.to(model.device) for k, v in inputs.items()}
                    outputs = model.generate(**inputs, **gen_cfg)
                    full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    pred = full_text[len(prompt):].strip().split('\n')[0].strip()
                    decoded_preds.append(pred if pred else "")

            metrics = self.metrics_computer.compute_eval_text_metrics(decoded_preds, references)
            to_log = {f"train/{k}": float(v) for k, v in metrics.items()}
            if state.total_flos is not None:
                to_log["train/total_flos"] = float(state.total_flos)
            trainer.log(to_log)

            # Free GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            logger.warning(f"Train metrics computation failed: {e}")


def simplified_model_evaluation(model, tokenizer, test_data, metrics_computer, max_samples=50):
    logger.info("=" * 60)
    logger.info("ΑΞΙΟΛΟΓΗΣΗ ΜΟΝΤΕΛΟΥ")
    logger.info("=" * 60)
    model.eval()
    predictions = []
    references = []
    eval_samples = test_data[:max_samples] if len(test_data) > max_samples else test_data
    logger.info(f"Αξιολόγηση με {len(eval_samples)} δείγματα...")

    generation_config = {
        "max_new_tokens": 200,
        "temperature": 0.4,
        "do_sample": True,
        "top_p": 0.9,
        "top_k": 50,
        "repetition_penalty": 1.1,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id
    }

    for i, item in enumerate(tqdm(eval_samples, desc="Generating predictions")):
        try:
            prompt = f"Ερώτηση: {item['input']}\nΑπάντηση:"
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=400)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model.generate(**inputs, **generation_config)
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            prediction = generated_text[len(prompt):].strip()
            prediction = prediction.split('\n')[0].strip()
            if len(prediction) == 0:
                prediction = "Δεν μπορώ να απαντήσω σε αυτή την ερώτηση."
            predictions.append(prediction)
            references.append(item['output'])
        except Exception as e:
            logger.warning(f"Σφάλμα στο δείγμα {i}: {e}")
            predictions.append("Σφάλμα κατά τη γένεση.")
            references.append(item['output'])

    logger.info("Υπολογισμός μετρικών...")
    metrics = metrics_computer.compute_eval_text_metrics(predictions, references)

    logger.info("\n" + "=" * 60)
    logger.info("ΑΠΟΤΕΛΕΣΜΑΤΑ ΑΞΙΟΛΟΓΗΣΗΣ")
    logger.info("=" * 60)
    if 'accuracy' in metrics:
        logger.info(f"  • Exact Match: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    logger.info(f"  • Token F1-Score: {metrics.get('f1', 0):.4f} ({metrics.get('f1', 0)*100:.2f}%)")
    logger.info(f"  • Token Precision: {metrics.get('precision', 0):.4f} ({metrics.get('precision', 0)*100:.2f}%)")
    logger.info(f"  • Token Recall: {metrics.get('recall', 0):.4f} ({metrics.get('recall', 0)*100:.2f}%)")
    logger.info(f"  • False Positive Rate: {metrics.get('false_positive_rate', 0):.4f}")
    logger.info(f"  • False Negative Rate: {metrics.get('false_negative_rate', 0):.4f}")
    logger.info(f"  • Random Accuracy: {metrics.get('random_accuracy', 0):.6f}")
    logger.info("=" * 60)

    n_samples = min(5, len(predictions))
    return metrics, predictions[:n_samples], references[:n_samples]


def load_and_prepare_data(file_path, test_size=0.1, val_size=0.10):
    logger.info(f"Φόρτωση δεδομένων από {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Το αρχείο {file_path} δεν βρέθηκε!")
    
    data = []
    invalid_samples = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                if not all(key in item for key in ['input', 'output']):
                    invalid_samples += 1
                    continue
                if not item['input'].strip() or not item['output'].strip():
                    invalid_samples += 1
                    continue
                data.append(item)
            except json.JSONDecodeError:
                invalid_samples += 1
                continue
    
    logger.info(f"Συνολικά έγκυρα δεδομένα: {len(data)}")
    if len(data) < 10:
        raise ValueError("Πολύ λίγα δεδομένα για training!")
    
    # Split data
    train_data, temp_data = train_test_split(data, test_size=test_size+val_size, random_state=42, shuffle=True)
    val_data, test_data = train_test_split(temp_data, test_size=test_size/(test_size+val_size), random_state=42, shuffle=True)
    
    logger.info(f"Κατανομή δεδομένων - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    # Save test data to a separate file
    base_name = os.path.splitext(os.path.basename(file_path))[0]  # without extension
    test_file_path = os.path.join(os.path.dirname(file_path), f"{base_name}_test.jsonl")
    
    logger.info(f"Αποθήκευση test data στο: {test_file_path}")
    with open(test_file_path, 'w', encoding='utf-8') as f:
        for item in test_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    logger.info(f" Test data αποθηκεύτηκαν επιτυχώς: {len(test_data)} samples")
    logger.info(f" Test file: {test_file_path}")
    
    return train_data, val_data, test_data


def setup_model_and_tokenizer(model_name="google/gemma-3-4b-it"):
    logger.info(f"Φόρτωση μοντέλου: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    logger.info(f"Μοντέλο φορτώθηκε επιτυχώς. Vocabulary size: {len(tokenizer)}")
    return model, tokenizer


def main():
    config = {
        "model_name": "google/gemma-3-4b-it",
        "data_file": "./datasets/train_data_qa.jsonl",
        "output_dir": "./models/gemma-greek-4b-legal-with-metrics-5-epochs",
        "max_length": 512,
        "batch_size": 1,
        "gradient_accumulation_steps": 8,
        "learning_rate": 4e-5,
        "num_epochs": 5,
        "warmup_steps": 400,
        "weight_decay": 0.01,
        "logging_steps": 5,
        "save_steps": 400,
        "eval_steps": 100,
        "use_wandb": False,
        "bf16": True,
        "max_eval_samples": 50,
        "generation_max_length": 200,
        "early_stopping_patience": 3,
        "save_total_limit": 5,
        "load_best_model_at_end": True,
        "eval_strategy": "steps",
        "save_strategy": "steps",
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
    }

    os.makedirs(config["output_dir"], exist_ok=True)

    train_data, val_data, test_data = load_and_prepare_data(config["data_file"], test_size=0.1, val_size=0.08)
    model, tokenizer = setup_model_and_tokenizer(config["model_name"])

    train_dataset = GreekLegalDataset(train_data, tokenizer, config["max_length"])
    val_dataset = GreekLegalDataset(val_data, tokenizer, config["max_length"])

    metrics_computer = SimplifiedMetricsComputer(tokenizer)

    run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_epochs{config['num_epochs']}_bs{config['batch_size']}_lr{config['learning_rate']}"
    
    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        overwrite_output_dir=True,
        num_train_epochs=config["num_epochs"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        warmup_steps=config["warmup_steps"],
        logging_steps=config["logging_steps"],
        save_steps=config["save_steps"],
        eval_steps=config["eval_steps"],
        eval_strategy=config["eval_strategy"],
        save_strategy=config["save_strategy"],
        load_best_model_at_end=config["load_best_model_at_end"],
        metric_for_best_model=config["metric_for_best_model"],
        greater_is_better=config["greater_is_better"],
        bf16=config["bf16"],
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        report_to=["tensorboard"],
        run_name=f"legal-metrics-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        logging_dir=os.path.join("./tensorboard_logs", run_name),  # use το run_name
        seed=42,
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        save_total_limit=config["save_total_limit"],
        logging_first_step=True,
        dataloader_num_workers=2,
        label_smoothing_factor=0.1,
        warmup_ratio=0.1,
        ddp_find_unused_parameters=False,
        prediction_loss_only=False,
        push_to_hub=False,
        eval_accumulation_steps=1,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8)

    compute_metrics_fn = build_compute_metrics(tokenizer, metrics_computer)

    trainer = EnhancedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        metrics_computer=metrics_computer,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_fn,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=config["early_stopping_patience"]),
            TrainMetricsCallback(tokenizer, metrics_computer, sample_size=100, generate_max_new_tokens=64),
        ],
    )

    logger.info("\n" + "=" * 80)
    logger.info("ΕΚΚΙΝΗΣΗ FINE-TUNING ΜΕ ΠΛΗΡΕΙΣ ΜΕΤΡΙΚΕΣ")
    logger.info("=" * 80)
    logger.info(f"Training samples: {len(train_data):,}")
    logger.info(f"Validation samples: {len(val_data):,}")

    training_result = trainer.train()
    logger.info("Αποθήκευση τελικού μοντέλου...")
    trainer.save_model()
    tokenizer.save_pretrained(config["output_dir"])

    logger.info("Τελική αξιολόγηση στο validation set...")
    final_eval_results = trainer.evaluate()

    logger.info("Αξιολόγηση στο test set...")
    test_metrics, sample_predictions, sample_references = simplified_model_evaluation(
        model, tokenizer, test_data, metrics_computer, config["max_eval_samples"],
    )

    final_results = {
        "final_training_metrics": training_result.metrics,
        "final_validation_metrics": final_eval_results,
        "test_metrics": test_metrics,
        "config": config,
    }
    results_path = os.path.join(config["output_dir"], "results.json")
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)
    logger.info(f"Αποτελέσματα αποθηκεύτηκαν στο: {config['output_dir']}")
    logger.info(f"Λεπτομερή αποτελέσματα: {results_path}")


if __name__ == "__main__":
    main()
 
