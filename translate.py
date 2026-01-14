# python translate.py --in_files ocnli_train_expl.jsonl ocnli_val_expl_123.jsonl --out_dir ./translated_en --tgt_lang en --batch_size 128 --num_beams 4 --max_new_tokens 160 --device cuda --dtype fp16
# python translate.py --in_files ocnli_train_expl.jsonl ocnli_val_expl_123.jsonl --out_dir ./translated_de --tgt_lang de --batch_size 128 --num_beams 4 --max_new_tokens 160 --device cuda --dtype fp16
# python translate.py --in_files ocnli_train_expl.jsonl ocnli_val_expl_123.jsonl --out_dir ./translated_fr --tgt_lang fr --batch_size 128 --num_beams 4 --max_new_tokens 160 --device cuda --dtype fp16

# translate_ocnli_jsonl_to_en_qc.py
import argparse
import json
import os
import re
from typing import Any, Dict, List, Tuple

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


SRC_LANG = "zho_Hans"
# 支持的目标语言
SUPPORTED_LANGS = {
    "en": "eng_Latn",
    "de": "deu_Latn",
    "fr": "fra_Latn",
}

FIELDS_DEFAULT = ["premise", "hypothesis", "explanation_1", "explanation_2", "explanation_3"]

# --- placeholder patterns ---
RE_URL = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
RE_EMAIL = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b")
RE_NUM = re.compile(r"\d+([.,:/-]\d+)*")  # numbers like 12, 12.5, 2025-01-01, 1/2 etc.

RE_MULTI_SPACE = re.compile(r"[ \t]+")
RE_MULTI_NEWLINE = re.compile(r"\n{2,}")


def normalize_text(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = RE_MULTI_NEWLINE.sub("\n", s)
    s = RE_MULTI_SPACE.sub(" ", s)
    return s.strip()


def protect_spans(text: str) -> Tuple[str, Dict[str, str]]:
    """
    Replace URL/email/number spans with placeholders, return (protected_text, mapping).
    """
    mapping: Dict[str, str] = {}
    counter = 0

    def _sub(pattern: re.Pattern, tag: str, t: str) -> str:
        nonlocal counter
        def repl(m: re.Match) -> str:
            nonlocal counter
            key = f"__{tag}{counter}__"
            mapping[key] = m.group(0)
            counter += 1
            return key
        return pattern.sub(repl, t)

    t = text
    t = _sub(RE_URL, "URL", t)
    t = _sub(RE_EMAIL, "EMAIL", t)
    t = _sub(RE_NUM, "NUM", t)
    return t, mapping


def restore_spans(text: str, mapping: Dict[str, str]) -> str:
    for k in sorted(mapping.keys(), key=len, reverse=True):
        text = text.replace(k, mapping[k])
    return text


def count_chinese_chars(s: str) -> int:
    return sum(1 for ch in s if "\u4e00" <= ch <= "\u9fff")


def sanity_check(src: str, tgt: str, *,
                 min_tgt_chars: int,
                 max_zh_ratio: float,
                 min_len_ratio: float,
                 max_len_ratio: float) -> List[str]:
    """
    Return list of issue tags. Empty => pass.
    """
    issues = []
    src_n = len(src.strip())
    tgt_n = len(tgt.strip())

    if tgt_n < min_tgt_chars:
        issues.append("empty_or_too_short")

    if src_n > 0:
        ratio = tgt_n / max(1, src_n)
        if ratio < min_len_ratio or ratio > max_len_ratio:
            issues.append(f"len_ratio_out_of_range({ratio:.2f})")

    # crude "target language" check: too many Chinese chars in English output
    zh = count_chinese_chars(tgt)
    zh_ratio = zh / max(1, tgt_n)
    if zh_ratio > max_zh_ratio:
        issues.append(f"too_much_chinese({zh_ratio:.2f})")

    return issues


@torch.inference_mode()
def translate_batch(tokenizer, model, texts: List[str], *,
                    max_new_tokens: int,
                    num_beams: int,
                    tgt_lang: str) -> List[str]:
    tokenizer.src_lang = SRC_LANG
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(model.device)
    forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)
    out = model.generate(
        **inputs,
        forced_bos_token_id=forced_bos_token_id,
        do_sample=False,
        num_beams=num_beams,
        max_new_tokens=max_new_tokens,
    )
    return tokenizer.batch_decode(out, skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_files", nargs="+", required=True, help="Input jsonl files (e.g., train.jsonl dev.jsonl)")
    ap.add_argument("--out_dir", required=True, help="Output directory")
    ap.add_argument("--tgt_lang", required=True, choices=list(SUPPORTED_LANGS.keys()), 
                    help="Target language: en (English), de (German), fr (French)")
    ap.add_argument("--model", default="facebook/nllb-200-distilled-600M")
    ap.add_argument("--fields", default=",".join(FIELDS_DEFAULT),
                    help="Comma-separated fields to translate; creates <field>_<lang> by default.")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_new_tokens", type=int, default=160)
    ap.add_argument("--num_beams", type=int, default=4)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "fp32"])
    ap.add_argument("--overwrite", action="store_true", help="Overwrite original fields instead of adding *_<lang>.")
    ap.add_argument("--no_placeholder", action="store_true", help="Disable placeholder protection.")

    # QC thresholds (tunable)
    ap.add_argument("--min_tgt_chars", type=int, default=2, help="Minimum length of translation to be considered valid.")
    ap.add_argument("--max_zh_ratio", type=float, default=0.05, help="Max Chinese-char ratio allowed in target output.")
    ap.add_argument("--min_len_ratio", type=float, default=0.30, help="Min tgt/src length ratio.")
    ap.add_argument("--max_len_ratio", type=float, default=3.50, help="Max tgt/src length ratio.")

    args = ap.parse_args()

    tgt_lang_code = SUPPORTED_LANGS[args.tgt_lang]
    os.makedirs(args.out_dir, exist_ok=True)
    fields = [f.strip() for f in args.fields.split(",") if f.strip()]

    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device
    if device == "auto":
        device = "cpu"
    torch_dtype = torch.float16 if (args.dtype == "fp16" and device == "cuda") else torch.float32

    print(f"Loading model: {args.model}")
    print(f"Device: {device}, Dtype: {torch_dtype}")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    print("Loading model (compatible mode)...")
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    
    if torch_dtype == torch.float16 and device == "cuda":
        print("Converting to FP16...")
        model = model.half()
    
    print(f"Moving model to {device}...")
    model = model.to(device)
    model.eval()
    print("Model loaded successfully!")

    # 添加全局元数据模板
    mt_meta = {
        "mt_model": args.model,
        "src_lang": SRC_LANG,
        "tgt_lang": tgt_lang_code,
        "tgt_lang_code": args.tgt_lang,
        "do_sample": False,
        "num_beams": args.num_beams,
        "max_new_tokens": args.max_new_tokens,
        "placeholder_protection": (not args.no_placeholder),
        "qc": {
            "min_tgt_chars": args.min_tgt_chars,
            "max_zh_ratio": args.max_zh_ratio,
            "min_len_ratio": args.min_len_ratio,
            "max_len_ratio": args.max_len_ratio,
        }
    }

    for in_path in args.in_files:
        base = os.path.basename(in_path)
        stem = base[:-6] if base.endswith(".jsonl") else base
        out_path = os.path.join(args.out_dir, f"{stem}.{args.tgt_lang}.jsonl")
        bad_path = os.path.join(args.out_dir, f"bad_cases.{stem}.{args.tgt_lang}.jsonl")

        with open(in_path, "r", encoding="utf-8") as f:
            raw_lines = [line.strip() for line in f if line.strip()]

        # 收集所有需要翻译的任务
        # jobs: (line_idx, field, original_text, protected_text, mapping)
        jobs: List[Tuple[int, str, str, str, Dict[str, str]]] = []
        
        for li, line in enumerate(raw_lines):
            obj = json.loads(line)
            
            for field in fields:
                if field not in obj:
                    continue
                val = obj[field]
                if not isinstance(val, str):
                    continue
                val_n = normalize_text(val)
                if val_n == "":
                    continue

                if args.no_placeholder:
                    jobs.append((li, field, val_n, val_n, {}))
                else:
                    protected, mapping = protect_spans(val_n)
                    jobs.append((li, field, val_n, protected, mapping))

        # 存储翻译结果
        translations: Dict[Tuple[int, str], str] = {}
        issues_map: Dict[Tuple[int, str], List[str]] = {}

        # 批量翻译
        print(f"Translating {len(jobs)} fields from {base}...")
        for start in tqdm(range(0, len(jobs), args.batch_size), desc=f"Translating {base}"):
            batch = jobs[start:start + args.batch_size]
            texts = [p for (_, _, _, p, _) in batch]
            outs = translate_batch(tokenizer, model, texts, 
                                 max_new_tokens=args.max_new_tokens, 
                                 num_beams=args.num_beams,
                                 tgt_lang=tgt_lang_code)

            for (li, field, src_text, _prot, mapping), tr in zip(batch, outs):
                tr = normalize_text(tr)
                if mapping:
                    tr = restore_spans(tr, mapping)

                translations[(li, field)] = tr

                # QC
                issues = sanity_check(
                    src_text, tr,
                    min_tgt_chars=args.min_tgt_chars,
                    max_zh_ratio=args.max_zh_ratio,
                    min_len_ratio=args.min_len_ratio,
                    max_len_ratio=args.max_len_ratio,
                )
                if issues:
                    issues_map[(li, field)] = issues

        # 逐行写入输出文件
        total_flagged = 0
        with open(out_path, "w", encoding="utf-8") as wf, open(bad_path, "w", encoding="utf-8") as bf:
            for li, line in enumerate(raw_lines):
                obj = json.loads(line)
                
                # 翻译各个字段
                for field in fields:
                    if field not in obj or not isinstance(obj[field], str):
                        continue
                    
                    src = normalize_text(obj[field])
                    if src == "":
                        tr = ""
                    else:
                        tr = translations.get((li, field), "")

                    if args.overwrite:
                        obj[field] = tr
                    else:
                        obj[field + "_" + args.tgt_lang] = tr

                    # 检查是否有质量问题
                    key = (li, field)
                    if key in issues_map:
                        total_flagged += 1
                        bf.write(json.dumps({
                            "line_idx": li,
                            "field": field,
                            "label": obj.get("label", None),
                            "src": src,
                            "tgt": tr,
                            "issues": issues_map[key],
                        }, ensure_ascii=False) + "\n")

                # 添加元数据
                obj["mt_meta"] = mt_meta

                # 写入输出文件（保持原始结构）
                wf.write(json.dumps(obj, ensure_ascii=False) + "\n")

        print(f"[OK] {in_path} -> {out_path}")
        print(f"[QC] total_lines={len(raw_lines)} translated_fields={len(jobs)} flagged_fields={total_flagged}")
        print(f"[QC] bad cases written to: {bad_path}")


if __name__ == "__main__":
    main()