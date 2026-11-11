# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Code adapted from https://github.com/microsoft/MInference/tree/main/scbench
# (via snu-mllab/KVzip results/metric.py and results/repo_qa_utils.py @ 5d84729);
# RULER string match from NVIDIA/kvpress evaluation/benchmarks/ruler (Apache-2.0).
"""Upstream-compatible answer metrics for the KV-scoring benchmark."""
import re
import string
from collections import Counter
from typing import Any, Mapping, Sequence

try:
    from rouge import Rouge
except ImportError:  # CPU unit tests which do not install the image dependency.
    Rouge = None


def normalize_answer(s):
    def remove_articles(text): return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text): return " ".join(text.split())
    def remove_punc(text): return "".join(ch for ch in text if ch not in set(string.punctuation))
    def lower(text): return text.lower()
    def replace_num(text):
        nums = {"zero":"0","one":"1","two":"2","three":"3","four":"4","five":"5","six":"6","seven":"7","eight":"8","nine":"9"}
        return re.compile(r"\b(" + "|".join(nums) + r")\b").sub(lambda x: nums[x.group()], text)
    return replace_num(white_space_fix(remove_articles(remove_punc(lower(s)))))


def rouge_score(prediction, ground_truth, **kwargs):
    if Rouge is None:
        return 0.0
    try: return Rouge().get_scores([prediction], [ground_truth], avg=True)["rouge-l"]["f"]
    except Exception: return 0.0


def f1_score(pred, ref, normalize=True):
    if normalize: pred, ref = normalize_answer(pred), normalize_answer(ref)
    pt, rt = pred.split(), ref.split()
    same = sum((Counter(pt) & Counter(rt)).values())
    if same == 0: return 0
    precision, recall = same / len(pt), same / len(rt)
    return 2 * precision * recall / (precision + recall)


def include_score(pred, ref, normalize=True):
    if normalize: pred, ref = normalize_answer(pred), normalize_answer(ref)
    return ref in pred


def _clean(pred: str) -> str:
    return pred[:-4] if pred.endswith("</s>") else pred


def evaluate_one(pred, ref, dataname, format="qa"):
    """v1 metrics, unchanged: include / gsm / exact / many_shot / ROUGE-L."""
    pred = _clean(pred)
    if not pred.strip(): return 0.0
    if format != "qa": return rouge_score(pred, ref)
    if "_vt" in dataname:
        refs = ref.split(", ")
        return sum(r in pred for r in refs) / len(refs)
    if "_mf" in dataname: return float(pred == ref)
    if "_many_shot" in dataname:
        if "(" in pred and "(" in ref: return float(pred.split("(")[1].split(")")[0] == ref.split("(")[1].split(")")[0])
        if ref.startswith("("): ref = ref.split(")", 1)[1].strip()
        return float(include_score(pred, ref))
    if "summary" in dataname: return rouge_score(pred, ref)
    if "qa_eng" in dataname: return max(f1_score(pred, ref), include_score(pred, ref))
    if "choice_eng" in dataname: return float(include_score(pred.split("\n")[0], ref))
    if "gsm" in dataname:
        pred = pred.strip().lower().split("the answer is ")[-1]
        return float(ref.strip().split("#### ")[-1] in pred)
    return float(include_score(pred, ref))


def evaluate_answer(preds, refs, dataname, format="qa"):
    return [evaluate_one(p, r, dataname, format) for p, r in zip(preds, refs)]


_RULER_CONTROL = re.compile(r"[\x00-\x1f]")


def _ruler_pred(pred: str) -> str:
    return _RULER_CONTROL.sub("", _clean(pred).strip()).strip().lower()


def ruler_match_all(pred: str, ref: Any) -> float:
    """kvpress ruler string_match_all for one row: fraction of reference values present."""
    refs = ref if isinstance(ref, list) else [ref]
    text = _ruler_pred(pred)
    return sum(1.0 if r.lower() in text else 0.0 for r in refs) / len(refs)


def ruler_match_part(pred: str, ref: Any) -> float:
    """kvpress ruler string_match_part for one row: any reference value present."""
    refs = ref if isinstance(ref, list) else [ref]
    text = _ruler_pred(pred)
    return max(1.0 if r.lower() in text else 0.0 for r in refs)


# ------------------------------------------------------------------ RepoQA

REPOQA_THRESHOLD = 0.8
FUNCTION_QUERY = {
    "python": "(function_definition name: (_)) @fdef",
    "java": "(method_declaration name: (_)) @fdef",
    "typescript": "(function_declaration name: (_)) @fdef",
    "rust": "(function_item name: (_)) @fdef",
    "cpp": "(function_definition declarator: (function_declarator declarator: (identifier))) @fdef",
    "go": "(function_declaration name: (_)) @fdef",
}


def _sanitize_code(model_output: str, lang: str) -> str:
    """repo_qa_utils.sanitize_output: first function of the first fenced block, else the block, else the text."""
    model_output = model_output.strip()
    blocks = re.findall(r"^```(?:\w+)?\s*\n(.*?)(?=^```)```", model_output, re.DOTALL | re.MULTILINE)
    if not blocks: return model_output
    from tree_sitter_languages import get_language, get_parser
    parser = get_parser(lang); query = get_language(lang).query(FUNCTION_QUERY[lang])
    for block in blocks:
        try:
            data = bytes(block, "utf8"); tree = parser.parse(data)
            for node, _name in query.captures(tree.root_node):
                return data[node.start_byte:node.end_byte].decode("utf8")
        except Exception:
            pass
    return blocks[0]


def _code_similarity(candidate: str, reference: str) -> float:
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
    return sentence_bleu([re.split(r"\s+", reference.strip())], re.split(r"\s+", candidate.strip()),
                         smoothing_function=SmoothingFunction().method4)


def repoqa_scores(preds: Sequence[str], repo: Mapping[str, Any]) -> list[float]:
    """Per-question pass@1 at BLEU >= 0.8 against every needle of the repository.

    Upstream drops empty predictions from the denominator; here an empty
    prediction scores 0 so every question counts once.
    """
    lang = str(repo["lang"]); names = list(repo["func_names"]); needles = list(repo["needles"])
    out = []
    for pred, target in zip(preds, names):
        text = _clean(pred)
        if not text.strip(): out.append(0.0); continue
        code = _sanitize_code(text, lang); best_name, best = None, 0.0
        for name, needle in zip(names, needles):
            similarity = _code_similarity(code, needle)
            if similarity > best: best, best_name = similarity, name
        out.append(1.0 if best_name == target and best >= REPOQA_THRESHOLD else 0.0)
    return out


def score_item(preds: Sequence[str], item: Mapping[str, Any], metric: str, format: str = "qa") -> list[float]:
    """Score one row's predictions (one per question) with the task's declared metric."""
    answers = item["answers"]; task = str(item["task"])
    if len(preds) != len(answers): raise ValueError("prediction count differs from the reference count")
    if metric == "repoqa": return repoqa_scores(preds, item["meta"]["repoqa"])
    if metric == "ruler_all": return [ruler_match_all(p, r) for p, r in zip(preds, answers)]
    if metric == "ruler_part": return [ruler_match_part(p, r) for p, r in zip(preds, answers)]
    if metric in ("include", "gsm", "exact", "many_shot", "rouge_l"):
        return [float(v) for v in evaluate_answer(preds, answers, task, format)]
    raise ValueError(f"unknown metric: {metric}")


def avg_list_of_list(values):
    means = [sum(v) / len(v) if v else 0 for v in values]
    return sum(means) / len(means) if means else 0
