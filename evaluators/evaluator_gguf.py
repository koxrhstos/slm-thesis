import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import json
import math
import random
import re
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import logging
from tqdm import tqdm

# llama.cpp for GGUF
try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except Exception:
    LLAMA_CPP_AVAILABLE = False

# Optional evaluation libs
try:
    import evaluate
    EVALUATE_AVAILABLE = True
except Exception:
    evaluate = None
    EVALUATE_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


class GGUFMetricsAligned:
    # Ίδια σημασιολογία/ονόματα με τον “σωστό”
    _WORD_REGEX = re.compile(r"[Α-ΩΆ-Ώα-ωά-ώA-Za-z0-9]+|[.,;:!?]")

    def __init__(self):
        self.use_library = EVALUATE_AVAILABLE
        self.bleu = None
        self.rouge = None
        self.meteor = None
        self.bertscore = None
        if self.use_library:
            try:
                self.bleu = evaluate.load("sacrebleu")
            except Exception as e:
                logger.warning(f"Failed to load sacrebleu: {e}")
            try:
                self.rouge = evaluate.load("rouge")
            except Exception as e:
                logger.warning(f"Failed to load rouge: {e}")
            try:
                self.meteor = evaluate.load("meteor")
            except Exception as e:
                logger.warning(f"Failed to load meteor: {e}")
            try:
                self.bertscore = evaluate.load("bertscore")
            except Exception as e:
                logger.warning(f"Failed to load bertscore: {e}")
                self.bertscore = None

    def normalize_text(self, text: str) -> str:
        if not text or not isinstance(text, str):
            return ""
        text = " ".join(text.split())
        artifacts = [
            "Δεν μπορώ να απαντήσω",
            "Σφάλμα κατά τη γένεση",
            "Ερώτηση:",
            "Απάντηση:",
            "<pad>", "<unk>", "<s>", "</s>",
            "Human:", "Assistant:",
        ]
        for artifact in artifacts:
            text = text.replace(artifact, "")
        text = re.sub(r"\s+([.,;:!?])", r"\1", text)
        text = re.sub(r"([.,;:!?])\s*", r"\1 ", text)
        return text.strip()

    def tokenize_words(self, text: str) -> List[str]:
        if not text:
            return []
        text = self.normalize_text(text)
        return [t for t in self._WORD_REGEX.findall(text) if t.strip()]

    # Exact match
    def exact_match_score(self, predictions: List[str], references: List[str]) -> float:
        if not predictions or not references:
            return 0.0
        exact_matches = 0
        valid = 0
        for pred, ref in zip(predictions, references):
            p = self.normalize_text(pred).lower()
            r = self.normalize_text(ref).lower()
            if p or r:
                valid += 1
                if p and r and p == r:
                    exact_matches += 1
        return exact_matches / valid if valid > 0 else 0.0

    # Token-level F1 (με Counters)
    def token_level_f1(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        if not predictions or not references:
            return {"token_f1": 0.0, "token_precision": 0.0, "token_recall": 0.0}
        total_precision = 0.0
        total_recall = 0.0
        total_f1 = 0.0
        valid = 0
        for pred, ref in zip(predictions, references):
            pt = Counter(self.tokenize_words(pred))
            rt = Counter(self.tokenize_words(ref))
            if not pt and not rt:
                continue
            all_keys = set(pt) | set(rt)
            overlap = sum(min(pt[k], rt[k]) for k in all_keys)
            p_tot = sum(pt.values())
            r_tot = sum(rt.values())
            precision = overlap / p_tot if p_tot > 0 else 0.0
            recall = overlap / r_tot if r_tot > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            total_precision += precision
            total_recall += recall
            total_f1 += f1
            valid += 1
        if valid == 0:
            return {"token_f1": 0.0, "token_precision": 0.0, "token_recall": 0.0}
        return {
            "token_f1": total_f1 / valid,
            "token_precision": total_precision / valid,
            "token_recall": total_recall / valid,
        }

    # BLEU (sacrebleu ή corpus fallback)
    def compute_bleu(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        preds = [self.normalize_text(p) for p in predictions]
        refs = [self.normalize_text(r) for r in references]
        pairs = [(p, r) for p, r in zip(preds, refs) if p and r]
        if not pairs:
            return {"bleu": 0.0, "bleu_p1": 0.0, "bleu_p2": 0.0, "bleu_p3": 0.0, "bleu_p4": 0.0}
        pp, rr = zip(*pairs)
        if self.bleu is not None:
            try:
                res = self.bleu.compute(predictions=list(pp), references=[[r] for r in rr], tokenize="intl")
                score = res.get("score", 0.0) / 100.0
                precisions = [p / 100.0 for p in res.get("precisions", [0.0, 0.0, 0.0, 0.0])]
                out = {"bleu": score}
                for i, p in enumerate(precisions, 1):
                    out[f"bleu_p{i}"] = p
                return out
            except Exception as e:
                logger.warning(f"sacrebleu failed, using fallback: {e}")
        return self._compute_bleu_corpus_fallback(list(pp), list(rr))

    def _compute_bleu_corpus_fallback(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        def ngrams(tokens: List[str], n: int) -> Counter:
            if len(tokens) < n:
                return Counter()
            return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))

        tot_pred_len, tot_ref_len = 0, 0
        tot_pred_ng = [0, 0, 0, 0]
        tot_match_ng = [0, 0, 0, 0]

        for p, r in zip(predictions, references):
            pt = self.tokenize_words(p)
            rt = self.tokenize_words(r)
            tot_pred_len += len(pt)
            tot_ref_len += len(rt)
            for n in range(1, 5):
                p_ng = ngrams(pt, n)
                r_ng = ngrams(rt, n)
                tot_pred_ng[n - 1] += sum(p_ng.values())
                tot_match_ng[n - 1] += sum(min(c, r_ng.get(g, 0)) for g, c in p_ng.items())

        precisions = []
        for i in range(4):
            denom = tot_pred_ng[i]
            prec = (tot_match_ng[i] / denom) if denom > 0 else 0.0
            precisions.append(max(prec, 1e-16))

        if tot_pred_len == 0:
            bp = 0.0
        elif tot_pred_len > tot_ref_len:
            bp = 1.0
        else:
            bp = math.exp(1 - (tot_ref_len / max(tot_pred_len, 1)))

        bleu = bp * math.exp(sum(math.log(p) for p in precisions) / 4.0)
        return {
            "bleu": float(bleu),
            "bleu_p1": float(precisions[0]),
            "bleu_p2": float(precisions[1]),
            "bleu_p3": float(precisions[2]),
            "bleu_p4": float(precisions[3]),
        }

    # ROUGE
    def compute_rouge(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        preds = [self.normalize_text(p) for p in predictions]
        refs = [self.normalize_text(r) for r in references]
        pairs = [(p, r) for p, r in zip(preds, refs) if p and r]
        if not pairs:
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "rougeLsum": 0.0}
        pp, rr = zip(*pairs)
        if self.rouge is not None:
            try:
                scores = self.rouge.compute(predictions=list(pp), references=list(rr))
                return {
                    "rouge1": float(scores.get("rouge1", 0.0)),
                    "rouge2": float(scores.get("rouge2", 0.0)),
                    "rougeL": float(scores.get("rougeL", 0.0)),
                    "rougeLsum": float(scores.get("rougeLsum", scores.get("rougeL", 0.0))),
                }
            except Exception as e:
                logger.warning(f"ROUGE failed, using custom: {e}")
        return self._compute_rouge_f1_custom(list(pp), list(rr))

    def _compute_rouge_f1_custom(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        def rouge_n_f1(p_tokens: List[str], r_tokens: List[str], n: int) -> float:
            def ngrams(tokens: List[str], n: int) -> Counter:
                if len(tokens) < n:
                    return Counter()
                return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))

            p_ng = ngrams(p_tokens, n)
            r_ng = ngrams(r_tokens, n)
            if not r_ng:
                return 0.0
            overlap = sum(min(c, p_ng.get(g, 0)) for g, c in r_ng.items())
            precision = overlap / sum(p_ng.values()) if sum(p_ng.values()) > 0 else 0.0
            recall = overlap / sum(r_ng.values()) if sum(r_ng.values()) > 0 else 0.0
            return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        def rouge_l_f1(p_tokens: List[str], r_tokens: List[str]) -> float:
            if not p_tokens or not r_tokens:
                return 0.0
            m, n = len(p_tokens), len(r_tokens)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if p_tokens[i - 1] == r_tokens[j - 1]:
                        dp[i][j] = dp[i - 1][j - 1] + 1
                    else:
                        dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
            lcs = dp[m][n]
            precision = lcs / m
            recall = lcs / n
            return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        r1, r2, rl = [], [], []
        for p, r in zip(predictions, references):
            pt = self.tokenize_words(p)
            rt = self.tokenize_words(r)
            r1.append(rouge_n_f1(pt, rt, 1))
            r2.append(rouge_n_f1(pt, rt, 2))
            rl.append(rouge_l_f1(pt, rt))
        return {
            "rouge1": float(np.mean(r1) if r1 else 0.0),
            "rouge2": float(np.mean(r2) if r2 else 0.0),
            "rougeL": float(np.mean(rl) if rl else 0.0),
            "rougeLsum": float(np.mean(rl) if rl else 0.0),
        }

    # METEOR
    def compute_meteor(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        preds = [self.normalize_text(p) for p in predictions]
        refs = [self.normalize_text(r) for r in references]
        pairs = [(p, r) for p, r in zip(preds, refs) if p and r]
        if not pairs:
            return {"meteor": 0.0} if self.meteor is not None else {"meteor_like": 0.0}
        pp, rr = zip(*pairs)
        if self.meteor is not None:
            try:
                res = self.meteor.compute(predictions=list(pp), references=list(rr))
                return {"meteor": float(res.get("meteor", 0.0))}
            except Exception as e:
                logger.warning(f"METEOR failed; using heuristic: {e}")
        # heuristic meteor_like
        scores = []
        for p, r in zip(pp, rr):
            pt = self.tokenize_words(p)
            rt = self.tokenize_words(r)
            if not pt or not rt:
                scores.append(0.0)
                continue
            p_set = set(pt)
            r_set = set(rt)
            exact = len(p_set & r_set)
            partial = 0.0
            for a in pt:
                if a in r_set:
                    continue
                for b in rt:
                    if b in p_set:
                        continue
                    if len(a) >= 4 and len(b) >= 4 and a[:4] == b[:4]:
                        partial += 0.5
                        break
            total = exact + partial
            precision = total / len(pt)
            recall = total / len(rt)
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            scores.append(f1)
        return {"meteor_like": float(np.mean(scores) if scores else 0.0)}

    # Semantic (BERTScore ή Jaccard)
    def compute_semantic(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        preds = [self.normalize_text(p) for p in predictions]
        refs = [self.normalize_text(r) for r in references]
        pairs = [(p, r) for p, r in zip(preds, refs) if p and r]
        if not pairs:
            return {
                "bertscore_precision": 0.0,
                "bertscore_recall": 0.0,
                "bertscore_f1": 0.0,
                "token_jaccard": 0.0,
                "semantic_similarity": 0.0,
            }
        pp, rr = zip(*pairs)

        bert_metrics = None
        if self.bertscore is not None:
            try:
                res = self.bertscore.compute(
                    predictions=list(pp),
                    references=list(rr),
                    model_type="bert-base-multilingual-cased",
                    verbose=False,
                )
                precision = float(np.mean(res.get("precision", [0.0])))
                recall = float(np.mean(res.get("recall", [0.0])))
                f1 = float(np.mean(res.get("f1", [0.0])))
                bert_metrics = {
                    "bertscore_precision": precision,
                    "bertscore_recall": recall,
                    "bertscore_f1": f1,
                }
            except Exception as e:
                logger.warning(f"BERTScore failed: {e}")

        jaccs = []
        for p, r in zip(pp, rr):
            p_tok = set(self.tokenize_words(p))
            r_tok = set(self.tokenize_words(r))
            if not p_tok and not r_tok:
                jaccs.append(1.0)
            elif not p_tok or not r_tok:
                jaccs.append(0.0)
            else:
                inter = len(p_tok & r_tok)
                uni = len(p_tok | r_tok)
                jaccs.append(inter / uni if uni > 0 else 0.0)
        token_jaccard = float(np.mean(jaccs)) if jaccs else 0.0

        out = {"token_jaccard": token_jaccard}
        if bert_metrics is not None:
            out.update(bert_metrics)
            out["semantic_similarity"] = bert_metrics["bertscore_f1"]
        else:
            out.update({"bertscore_precision": 0.0, "bertscore_recall": 0.0, "bertscore_f1": 0.0})
            out["semantic_similarity"] = token_jaccard
        return out

    def compute_all_metrics(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        if not predictions or not references or len(predictions) != len(references):
            logger.warning("Invalid data for metric computation")
            return {}
        metrics: Dict[str, float] = {}
        metrics["exact_match"] = self.exact_match_score(predictions, references)
        metrics.update(self.token_level_f1(predictions, references))
        metrics.update(self.compute_bleu(predictions, references))
        metrics.update(self.compute_rouge(predictions, references))
        metrics.update(self.compute_meteor(predictions, references))
        metrics.update(self.compute_semantic(predictions, references))

        # length/diversity
        pred_lengths: List[int] = []
        ref_lengths: List[int] = []
        ratios: List[float] = []
        all_pred_tokens: List[str] = []
        unique_preds: set = set()
        for p, r in zip(predictions, references):
            pt = self.tokenize_words(p)
            rt = self.tokenize_words(r)
            if pt and rt:
                pred_lengths.append(len(pt))
                ref_lengths.append(len(rt))
                ratios.append(len(pt) / len(rt) if len(rt) > 0 else 0.0)
            all_pred_tokens.extend(pt)
            unique_preds.add(self.normalize_text(p))
        if pred_lengths:
            unique_tokens = set(all_pred_tokens)
            ttr = len(unique_tokens) / len(all_pred_tokens) if all_pred_tokens else 0.0
            pred_uniqueness = len(unique_preds) / len(predictions) if predictions else 0.0
            metrics.update({
                "avg_pred_length": float(np.mean(pred_lengths)),
                "avg_ref_length": float(np.mean(ref_lengths)),
                "length_ratio": float(np.mean(ratios) if ratios else 0.0),
                "length_ratio_std": float(np.std(ratios) if ratios else 0.0),
                "pred_length_std": float(np.std(pred_lengths) if pred_lengths else 0.0),
                "ref_length_std": float(np.std(ref_lengths) if ref_lengths else 0.0),
                "type_token_ratio": float(ttr),
                "prediction_uniqueness": float(pred_uniqueness),
                "unique_tokens": len(unique_tokens),
                "total_tokens": len(all_pred_tokens),
            })
        return metrics


def clean_prediction(prediction: str) -> str:
    if not prediction:
        return ""
    artifacts = [
        "Απάντηση:",
        "Ερώτηση:",
        "<pad>", "<unk>", "<s>", "</s>",
        "Human:", "Assistant:",
        "Δεν μπορώ να απαντήσω σε αυτή την ερώτηση",
        "Σφάλμα κατά τη γένεση",
    ]
    for a in artifacts:
        prediction = prediction.replace(a, "")
    prediction = prediction.strip()
    if (prediction.startswith('"') and prediction.endswith('"')) or (prediction.startswith("'") and prediction.endswith("'")):
        prediction = prediction[1:-1].strip()
    words = prediction.split()
    if len(words) > 4:
        cleaned: List[str] = []
        i = 0
        while i < len(words):
            w = words[i]
            c = 1
            while i + c < len(words) and words[i + c] == w:
                c += 1
            cleaned.extend([w] * min(c, 2))
            i += c
        prediction = " ".join(cleaned)
    prediction = re.sub(r"\s+", " ", prediction)
    prediction = re.sub(r"([.!?])\1+", r"\1", prediction)
    return prediction.strip()


def load_gguf_model(model_path: str, n_gpu_layers: int = -1, n_ctx: int = 2048, n_threads: int | None = None):
    if not LLAMA_CPP_AVAILABLE:
        raise ImportError("llama-cpp-python is required. Install with: pip install llama-cpp-python")
    logger.info(f"Loading GGUF model: {model_path}")
    params = {
        "model_path": model_path,
        "n_ctx": n_ctx,
        "n_gpu_layers": n_gpu_layers,
        "n_batch": 512,
        "f16_kv": True,
        "verbose": False,
    }
    if n_threads:
        params["n_threads"] = n_threads
    return Llama(**params)


def gguf_generate(model: Llama, prompt: str) -> str:
    # Χρησιμοποιούμε create_completion για σταθερότητα API
    response = model.create_completion(
        prompt=prompt,
        max_tokens=250,
        temperature=0.3,
        top_p=0.9,
        top_k=40,
        repeat_penalty=1.05,
        stop=["\nΕρώτηση:", "\nQuestion:", "\n\n"],
    )
    return response["choices"][0]["text"]


def gguf_model_evaluation(
    model: Llama,
    test_data: List[Dict[str, str]],
    metrics_computer: GGUFMetricsAligned,
    max_samples: int = 200,
    seed: int = 42,
) -> Tuple[Dict[str, float], List[str], List[str]]:
    logger.info("=" * 60)
    logger.info("GGUF MODEL EVALUATION")
    logger.info("=" * 60)

    set_seed(seed)
    predictions: List[str] = []
    references: List[str] = []

    eval_samples = test_data[:max_samples] if len(test_data) > max_samples else test_data
    logger.info(f"Evaluating with {len(eval_samples)} samples...")

    success = 0
    fail = 0
    for i, item in enumerate(tqdm(eval_samples, desc="Generating")):
        try:
            prompt = f"Ερώτηση: {item['input'].strip()}\nΑπάντηση:"
            raw = gguf_generate(model, prompt)
            pred = clean_prediction(raw)
            if not pred:
                pred = "Δεν υπάρχει διαθέσιμη απάντηση."
                fail += 1
            else:
                success += 1
            predictions.append(pred)
            references.append(item["output"].strip())
        except Exception as e:
            logger.warning(f"Error in sample {i}: {e}")
            predictions.append("Σφάλμα κατά τη γένεση.")
            references.append(item["output"].strip())
            fail += 1

    total = len(eval_samples)
    logger.info("Generation Statistics:")
    logger.info(f"  Successful: {success}/{total} ({(success/total)*100:.1f}%)")
    logger.info(f"  Failed: {fail}/{total} ({(fail/total)*100:.1f}%)")

    logger.info("\nFIRST 5 PREDICTION EXAMPLES:")
    logger.info("=" * 60)
    for i in range(min(5, len(predictions))):
        p_len = len(metrics_computer.tokenize_words(predictions[i]))
        r_len = len(metrics_computer.tokenize_words(references[i]))
        logger.info(f"Example {i+1}:")
        logger.info(f"Question: {eval_samples[i]['input'][:100]}...")
        logger.info(f"Reference: {references[i][:100]}...")
        logger.info(f"Prediction: {predictions[i][:100]}...")
        logger.info(f"Pred Length: {p_len} tokens")
        logger.info(f"Ref Length: {r_len} tokens")
        logger.info("-" * 40)

    logger.info("Computing metrics...")
    metrics = metrics_computer.compute_all_metrics(predictions, references)

    key = {
        "Exact Match": metrics.get("exact_match", 0.0),
        "Token F1": metrics.get("token_f1", 0.0),
        "BLEU": metrics.get("bleu", 0.0),
        "ROUGE-1": metrics.get("rouge1", 0.0),
        "METEOR": metrics.get("meteor", metrics.get("meteor_like", 0.0)),
        "Semantic Sim": metrics.get("semantic_similarity", 0.0),
    }
    logger.info("\nKEY METRICS SUMMARY:")
    for name, score in key.items():
        logger.info(f"  {name:15}: {score:.4f} ({score*100:.2f}%)")

    core_scores = [
        metrics.get("token_f1", 0.0),
        metrics.get("bleu", 0.0),
        metrics.get("rouge1", 0.0),
        metrics.get("meteor", metrics.get("meteor_like", 0.0)),
    ]
    positive = [s for s in core_scores if s > 0]
    overall = float(np.mean(positive)) if positive else 0.0
    logger.info(f"\nOVERALL SCORE: {overall:.4f} ({overall*100:.2f}%)")

    return metrics, predictions[:10], references[:10]


def evaluate_gguf_model(
    model_path: str,
    data_file: str,
    n_gpu_layers: int = -1,
    n_ctx: int = 2048,
    max_samples: int = 200,
) -> Dict:
    try:
        if not os.path.exists(model_path):
            logger.error(f"Model not found at: {model_path}")
            return {}
        if not os.path.exists(data_file):
            logger.error(f"Data file not found at: {data_file}")
            return {}

        logger.info("Loading GGUF model...")
        model = load_gguf_model(model_path=model_path, n_gpu_layers=n_gpu_layers, n_ctx=n_ctx)

        logger.info("Loading test data...")
        with open(data_file, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f if line.strip()]

        test_data = data[-225:-25] if len(data) > 50 else data[-200:] # 200 samples
        logger.info(f"Loaded {len(test_data)} test samples")

        logger.info("\nDATA SAMPLE ANALYSIS:")
        logger.info("=" * 50)
        metrics_computer = GGUFMetricsAligned()
        for i in range(min(3, len(test_data))):
            sample = test_data[i]
            logger.info(f"Sample {i+1}:")
            logger.info(f"  Input: {sample['input'][:100]}...")
            logger.info(f"  Output: {sample['output'][:100]}...")
            logger.info(f"  Input length: {len(metrics_computer.tokenize_words(sample['input']))} tokens")
            logger.info(f"  Output length: {len(metrics_computer.tokenize_words(sample['output']))} tokens")

        metrics, predictions, references = gguf_model_evaluation(
            model, test_data, metrics_computer, max_samples=max_samples, seed=42
        )

        out_dir = "./evaluations"
        os.makedirs(out_dir, exist_ok=True)
        results_path = os.path.join(
            out_dir, f"gguf_evaluation_results_{os.path.basename(model_path).replace('.gguf','')}.json"
        )
        results = {
            "metrics": metrics,
            "sample_predictions": predictions,
            "sample_references": references,
            "method": "standards_compliant_gguf",
            "model_path": model_path,
            "test_samples": len(test_data),
            "generation_config": {
                "max_tokens": 250,
                "temperature": 0.3,
                "top_p": 0.9,
                "top_k": 40,
                "repeat_penalty": 1.05,
                "stop": ["\\nΕρώτηση:", "\\nQuestion:", "\\n\\n"],
            },
            "model_config": {"n_gpu_layers": n_gpu_layers, "n_ctx": n_ctx},
        }
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        logger.info("\nEVALUATION COMPLETED SUCCESSFULLY!")
        logger.info(f"Results saved to: {results_path}")
        return results

    except Exception as e:
        logger.error(f"GGUF evaluation error: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    model_path = "./models/gemma-greek-4b-legal-with-metrics------7epochs-tq2_0.gguf"
    data_file = "./datasets/train_data_qa.jsonl"
    n_gpu_layers = -1
    n_ctx = 2048
    max_samples = 200

    evaluate_gguf_model(
        model_path=model_path,
        data_file=data_file,
        n_gpu_layers=n_gpu_layers,
        n_ctx=n_ctx,
        max_samples=max_samples,
    )
