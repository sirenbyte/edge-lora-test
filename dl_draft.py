from huggingface_hub import snapshot_download, list_repo_files
import json, os
# pick an LFM2.5 ~1B-class instruct that shares the 128k tokenizer
for repo in ["LiquidAI/LFM2.5-1.2B-Instruct","LiquidAI/LFM2.5-1.2B","LiquidAI/LFM2-1.2B"]:
    try:
        fs=list_repo_files(repo)
    except Exception as e:
        print("skip",repo,type(e).__name__); continue
    print("FOUND",repo)
    p=snapshot_download(repo, local_dir=f"draft_src",
        allow_patterns=["*.safetensors","*.json","tokenizer*","*.jinja","*.txt","merges*","vocab*"],
        ignore_patterns=["*.gguf"])
    c=json.load(open(os.path.join(p,"config.json")))
    print("DRAFT",repo,"vocab_size=",c.get("vocab_size"),"hidden=",c.get("hidden_size"))
    break
