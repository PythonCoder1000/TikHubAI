# save as predict_botdet_cli.py
import os, sys, json, argparse, torch, numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def load_threshold(model_dir, cli_thr):
    if cli_thr is not None:
        return float(cli_thr)
    p = os.path.join(model_dir, "tuned_threshold.txt")
    if os.path.exists(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                return float(f.read().strip())
        except Exception:
            pass
    return 0.5

def load_max_len(model_dir, cli_len):
    if cli_len is not None:
        return int(cli_len)
    p = os.path.join(model_dir, "train_args.json")
    if os.path.exists(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                args = json.load(f)
                return int(args.get("max_len", 192))
        except Exception:
            pass
    return 192

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", default="models/botdet_roberta")
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--max_len", type=int, default=None)
    p.add_argument("--eval_bs", type=int, default=16)
    args = p.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir).to(device).eval()
    thr = load_threshold(args.model_dir, args.threshold)
    max_len = load_max_len(args.model_dir, args.max_len)

    print(f"Loaded model from {args.model_dir}")
    print(f"Threshold={thr:.3f}  max_len={max_len}  device={device}")
    print("Type a comment (or 'q' to quit).")

    buf = []
    try:
        while True:
            try:
                line = input("> ").strip()
            except EOFError:
                break
            if line.lower() in {"q","quit","exit"}:
                break
            if not line:
                continue
            buf.append(line)
            if len(buf) >= args.eval_bs:
                enc = tok(buf, truncation=True, padding=True, max_length=max_len, return_tensors="pt")
                with torch.no_grad():
                    out = model(input_ids=enc["input_ids"].to(device), attention_mask=enc["attention_mask"].to(device))
                    probs = torch.softmax(out.logits, dim=1)[:,1].detach().cpu().numpy()
                for txt, p_fake in zip(buf, probs):
                    pred = "fake" if p_fake >= thr else "real"
                    print(f"[{pred}] p_fake={p_fake:.4f}  |  {txt}")
                buf = []
        if buf:
            enc = tok(buf, truncation=True, padding=True, max_length=max_len, return_tensors="pt")
            with torch.no_grad():
                out = model(input_ids=enc["input_ids"].to(device), attention_mask=enc["attention_mask"].to(device))
                probs = torch.softmax(out.logits, dim=1)[:,1].detach().cpu().numpy()
            for txt, p_fake in zip(buf, probs):
                pred = "fake" if p_fake >= thr else "real"
                print(f"[{pred}] p_fake={p_fake:.4f}  |  {txt}")
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
