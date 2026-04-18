"""Metric builders for Seq2SeqTrainer: SARI, penalized SARI, and a combined medical score.

Ensures NLTK punkt is available for sentence tokenization where needed.
"""
from __future__ import annotations

import numpy as np
import evaluate
import nltk
from transformers import PreTrainedTokenizer

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    nltk.download("punkt", quiet=True)


def make_sari_metrics(tokenizer: PreTrainedTokenizer):
    """Return a compute_metrics function that reports SARI on decoded generations.

    Clips prediction ids to the vocabulary, decodes predictions, labels, and inputs, and passes
    them to the Hugging Face evaluate SARI implementation. Label and input positions marked
    negative one hundred are treated as pad for decoding.
    """
    sari_metric = evaluate.load("sari")

    def compute_metrics_sari(eval_preds):
        predictions, labels, inputs = eval_preds.predictions, eval_preds.label_ids, eval_preds.inputs
        predictions = np.clip(predictions, 0, tokenizer.vocab_size - 1)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        inputs = np.where(inputs != -100, inputs, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_inputs = tokenizer.batch_decode(inputs, skip_special_tokens=True)
        all_refs = [[lbl] for lbl in decoded_labels]
        sari_results = sari_metric.compute(
            sources=decoded_inputs,
            predictions=decoded_preds,
            references=all_refs,
        )
        return {"sari": sari_results["sari"]}

    return compute_metrics_sari


def make_penalized_sari_metrics(tokenizer: PreTrainedTokenizer, instruction_prefix: str):
    """Return a compute_metrics function that combines SARI with a copying penalty.

    Strips the instruction prefix from decoded inputs before SARI so sources align with content.
    Subsamples at most five hundred examples for speed. Adds Jaccard-style word overlap between
    prediction and input, scales SARI down when overlap is high, and exposes penalized_sari.
    """
    sari_metric = evaluate.load("sari")
    prefix = instruction_prefix

    def compute_metrics_with_penalty(eval_preds):
        predictions, labels, inputs = eval_preds.predictions, eval_preds.label_ids, eval_preds.inputs
        predictions = np.clip(predictions, 0, tokenizer.vocab_size - 1)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        inputs = np.where(inputs != -100, inputs, tokenizer.pad_token_id)

        if len(predictions) > 500:
            idx = np.random.choice(len(predictions), 500, replace=False)
            predictions, labels, inputs = predictions[idx], labels[idx], inputs[idx]

        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_inputs = tokenizer.batch_decode(inputs, skip_special_tokens=True)

        cleaned_inputs = []
        for inp in decoded_inputs:
            if inp.startswith(prefix):
                cleaned_inputs.append(inp[len(prefix) :].strip())
            else:
                cleaned_inputs.append(inp)

        all_refs = [[lbl] for lbl in decoded_labels]
        sari_results = sari_metric.compute(
            sources=cleaned_inputs,
            predictions=decoded_preds,
            references=all_refs,
        )

        similarity_scores = []
        for pred, inp in zip(decoded_preds, cleaned_inputs):
            pred_words = set(pred.lower().split())
            inp_words = set(inp.lower().split())
            if not pred_words or not inp_words:
                similarity_scores.append(0.0)
            else:
                overlap = len(pred_words.intersection(inp_words))
                similarity_scores.append(overlap / len(inp_words))

        avg_similarity = sum(similarity_scores) / max(len(similarity_scores), 1)
        penalty_weight = 0.3
        penalized_sari = sari_results["sari"] * (1 - penalty_weight * avg_similarity)

        return {
            "sari": sari_results["sari"],
            "similarity": avg_similarity,
            "penalized_sari": penalized_sari,
        }

    return compute_metrics_with_penalty


class CombinedScoreMetrics:
    """Callable evaluator for medical continuation: SARI, BLEU, reading proxy, length, similarity.

    Strips simplification_prefix from decoded inputs to recover expert text. Combines metrics into
    combined_score for Trainer selection. Call set_tokenizer before evaluation so batch_decode works.
    """

    def __init__(self, simplification_prefix: str):
        self.tokenizer: PreTrainedTokenizer | None = None
        self.prefix = simplification_prefix
        self._sari = evaluate.load("sari")
        self._bleu = evaluate.load("bleu")

    def set_tokenizer(self, tokenizer: PreTrainedTokenizer) -> None:
        """Attach the tokenizer used to decode predictions and references."""
        self.tokenizer = tokenizer

    def __call__(self, eval_preds):
        """Decode predictions and return rounded scalar metrics including combined_score."""
        tok = self.tokenizer
        if tok is None:
            print("ERROR compute_metrics: tokenizer not set.")
            return {}

        if isinstance(eval_preds.predictions, tuple):
            predictions = eval_preds.predictions[0]
        else:
            predictions = eval_preds.predictions
        labels, inputs = eval_preds.label_ids, eval_preds.inputs
        if predictions.dtype == np.float32 or predictions.dtype == np.float16:
            predictions = np.argmax(predictions, axis=-1)

        try:
            labels = np.where(labels != -100, labels, tok.pad_token_id)
            if inputs is not None:
                inputs = np.where(inputs != -100, inputs, tok.pad_token_id)
            else:
                inputs = np.full(predictions.shape, tok.pad_token_id)
            vocab_size = tok.vocab_size
            invalid = (predictions < 0) | (predictions >= vocab_size)
            if np.any(invalid):
                predictions = np.clip(predictions, 0, vocab_size - 1)
            decoded_preds = tok.batch_decode(predictions, skip_special_tokens=True)
            decoded_labels = tok.batch_decode(labels, skip_special_tokens=True)
            decoded_inputs_raw = tok.batch_decode(inputs, skip_special_tokens=True) if inputs is not None else [""] * len(predictions)
        except Exception as e:
            print(f"FATAL ERROR decoding: {e}")
            return {}

        n = len(decoded_preds)
        if n == 0:
            return {}
        if n > 500:
            idx = np.random.choice(n, 500, replace=False)
            decoded_preds = [decoded_preds[i] for i in idx]
            decoded_labels = [decoded_labels[i] for i in idx]
            decoded_inputs_raw = [decoded_inputs_raw[i] for i in idx]

        plen = len(self.prefix)
        original_expert_texts = []
        for inp in decoded_inputs_raw:
            expert = inp[plen:].strip() if inp.startswith(self.prefix) else inp
            original_expert_texts.append(expert)

        decoded_preds_pp = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
        all_refs_pp = [["\n".join(nltk.sent_tokenize(label.strip()))] for label in decoded_labels]
        cleaned_inputs_pp = ["\n".join(nltk.sent_tokenize(inp.strip())) for inp in original_expert_texts]

        sari_score = 0.0
        if cleaned_inputs_pp and decoded_preds_pp and all_refs_pp:
            try:
                sari_score = self._sari.compute(
                    sources=cleaned_inputs_pp,
                    predictions=decoded_preds_pp,
                    references=all_refs_pp,
                )["sari"]
            except Exception as e:
                print(f"Error SARI: {e}")

        bleu_score = 0.0
        if decoded_preds_pp and all_refs_pp:
            try:
                bleu_score = self._bleu.compute(predictions=decoded_preds_pp, references=all_refs_pp)["bleu"]
            except Exception as e:
                print(f"Error BLEU: {e}")

        avg_reading_level = 0.0
        if decoded_preds and original_expert_texts:
            scores = []
            for pred, inp_orig in zip(decoded_preds, original_expert_texts):
                iw = nltk.word_tokenize(inp_orig)
                pw = nltk.word_tokenize(pred)
                if not iw or not pw:
                    continue
                try:
                    il = sum(len(w) for w in iw) / len(iw)
                    pl = sum(len(w) for w in pw) / len(pw)
                    if pl > 0 and il > 0:
                        rr = il / pl
                        scores.append(max(0, min(1, (rr - 1.0) * 0.5)))
                    else:
                        scores.append(0)
                except ZeroDivisionError:
                    scores.append(0)
            avg_reading_level = sum(scores) / max(len(scores), 1) if scores else 0.0

        avg_similarity = 0.0
        if decoded_preds and original_expert_texts:
            sims = []
            for pred, inp_orig in zip(decoded_preds, original_expert_texts):
                pw = set(nltk.word_tokenize(pred.lower()))
                iw = set(nltk.word_tokenize(inp_orig.lower()))
                if not pw or not iw:
                    sims.append(0.0)
                    continue
                inter = len(pw.intersection(iw))
                union = len(pw.union(iw))
                sims.append(inter / union if union > 0 else 0.0)
            avg_similarity = sum(sims) / max(len(sims), 1) if sims else 0.0

        length_ratio = 0.0
        length_score_component = 0.0
        if decoded_preds and original_expert_texts:
            pred_len = sum(len(nltk.word_tokenize(p)) for p in decoded_preds)
            inp_len = sum(len(nltk.word_tokenize(i)) for i in original_expert_texts)
            length_ratio = pred_len / inp_len if inp_len > 0 else 0.0
            length_penalty = abs(1.0 - length_ratio)
            length_score_component = max(0, 1.0 - length_penalty * 3.0)

        sim_pen = 0.9
        sari_w, bleu_w, read_w, len_w = 0.6, 0.04, 0.06, 0.30
        combined_before = (
            sari_w * sari_score
            + bleu_w * bleu_score
            + read_w * avg_reading_level
            + len_w * length_score_component
        )
        combined_score = combined_before * (1.0 - sim_pen * (avg_similarity**2))

        result = {
            "sari": sari_score,
            "bleu": bleu_score,
            "reading_level": avg_reading_level,
            "similarity": avg_similarity,
            "length_ratio": length_ratio,
            "combined_score": combined_score,
        }
        return {k: round(float(v), 4) for k, v in result.items()}
