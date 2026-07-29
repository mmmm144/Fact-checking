# Sinh claim ti?ng Vi?t v?i Qwen3-14B-AWQ tr?n Kaggle T4 x2

Notebook n?y t? clone branch `data`, t?i `corpus_v1.json` t? Hugging Face, ch?y `Qwen/Qwen3-14B-AWQ` b?ng vLLM tr?n hai GPU, th? 3 b?i, x? l? to?n b? 13.572 b?i theo chunk v? l?u checkpoint l?n `aiMy144/viet-fact-checking`.

Tr??c khi ch?y: b?t Internet, ch?n GPU T4 x2 v? t?o Kaggle Secret `HF_TOKEN` c? quy?n Write.

## Cell 1 ? D?n cache Qwen3.5 t?i d?

Ch? x?a cache model c? c? th? t?i l?i; kh?ng x?a repository, corpus ho?c checkpoint k?t qu?. Kh?ng c?n ch?y l?i cell n?y sau khi ?? t?i Qwen3-14B-AWQ trong c?ng phi?n.

```python
import shutil
from pathlib import Path

cleanup_targets = [
    Path("/kaggle/working/hf_cache/hub/models--Qwen--Qwen3.5-9B"),
    Path("/kaggle/working/hf_cache/hub/models--Qwen--Qwen3.5-35B-A3B-GPTQ-Int4"),
    Path("/root/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B"),
    Path("/root/.cache/huggingface/hub/models--Qwen--Qwen3.5-35B-A3B-GPTQ-Int4"),
    Path("/kaggle/working/hf_cache/xet"),
    Path("/root/.cache/huggingface/xet"),
    Path("/kaggle/working/models/Qwen3.5-9B"),
]
allowed_roots = [
    Path("/kaggle/working/hf_cache"),
    Path("/kaggle/working/models"),
    Path("/root/.cache/huggingface"),
]
for target in cleanup_targets:
    resolved = target.resolve(strict=False)
    allowed = any(
        resolved == root.resolve(strict=False)
        or root.resolve(strict=False) in resolved.parents
        for root in allowed_roots
    )
    if not allowed:
        raise RuntimeError(f"T? ch?i x?a: {target}")
    if target.exists():
        print("?ang x?a:", target)
        shutil.rmtree(target)

for record in [
    Path("/kaggle/working/qwen3_5_9b_model_path.txt"),
    Path("/kaggle/working/qwen3_14b_awq_model_path.txt"),
]:
    if record.exists():
        record.unlink()

disk = shutil.disk_usage("/kaggle/working")
print(f"Dung l??ng tr?ng: {disk.free / 1024**3:.2f} GiB")
```

## Cell 2 ? Ki?m tra T4 x2

```python
import torch

print("PyTorch:", torch.__version__)
print("S? GPU:", torch.cuda.device_count())
for index in range(torch.cuda.device_count()):
    properties = torch.cuda.get_device_properties(index)
    print(f"GPU {index}: {properties.name} | {properties.total_memory / 1024**3:.1f} GiB")
assert torch.cuda.device_count() == 2, "V?o Settings ? Accelerator ? GPU T4 x2"
```

```python
!nvidia-smi
```

## Cell 3 ? C?i th? vi?n

```python
!pip install -q -U uv huggingface_hub requests
!uv pip install --system -U vllm --torch-backend=auto
!vllm --version
```

## Cell 4 ? N?p HF_TOKEN t? Kaggle Secret

```python
import os
from kaggle_secrets import UserSecretsClient

HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("Kh?ng t?m th?y Kaggle Secret HF_TOKEN")
os.environ["HF_TOKEN"] = HF_TOKEN
os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN
os.environ["HF_HOME"] = "/kaggle/working/hf_cache"
os.environ["HF_HUB_DISABLE_XET"] = "1"
print("?? n?p HF_TOKEN an to?n.")
```

## Cell 5 ? Clone ??ng GitHub branch data

```python
from pathlib import Path
import subprocess

GITHUB_URL = "https://github.com/mmmm144/Fact-checking.git"
GITHUB_BRANCH = "data"
REPO_DIR = Path("/kaggle/working/Fact-checking")

if not (REPO_DIR / ".git").exists():
    subprocess.run([
        "git", "clone", "--branch", GITHUB_BRANCH, "--single-branch",
        GITHUB_URL, str(REPO_DIR)
    ], check=True)
else:
    subprocess.run(["git", "-C", str(REPO_DIR), "fetch", "origin", GITHUB_BRANCH], check=True)
    subprocess.run(["git", "-C", str(REPO_DIR), "checkout", GITHUB_BRANCH], check=True)
    subprocess.run(["git", "-C", str(REPO_DIR), "pull", "--ff-only", "origin", GITHUB_BRANCH], check=True)

SCRIPT_FILE = REPO_DIR / "generation_claim" / "fact_checking_Qween.py"
assert SCRIPT_FILE.is_file(), f"Kh?ng t?m th?y script: {SCRIPT_FILE}"
script_text = SCRIPT_FILE.read_text(encoding="utf-8")
assert "Qwen/Qwen3-14B-AWQ" in script_text, (
    "Branch data ch?a c? Qwen3-14B-AWQ; h?y commit/push script t? Windows."
)
print("Repository:", REPO_DIR)
print("Branch:", GITHUB_BRANCH)
print("Script:", SCRIPT_FILE)
print("?? x?c nh?n model Qwen/Qwen3-14B-AWQ.")
```

## Cell 6 ? T?i corpus_v1.json t? Hugging Face

```python
import json
from huggingface_hub import hf_hub_download

HF_DATASET_REPO = "aiMy144/viet-fact-checking"
DATA_DIR = REPO_DIR / "data" / "vie" / "raw" / "viet-fact-checking"
DATA_DIR.mkdir(parents=True, exist_ok=True)
INPUT_FILE = DATA_DIR / "corpus_v1.json"

hf_hub_download(
    repo_id=HF_DATASET_REPO,
    repo_type="dataset",
    filename="corpus_v1.json",
    token=HF_TOKEN,
    local_dir=str(DATA_DIR),
    force_download=False,
)
assert INPUT_FILE.is_file(), f"Kh?ng t?i ???c {INPUT_FILE}"
with INPUT_FILE.open("r", encoding="utf-8") as file:
    corpus = json.load(file)
assert isinstance(corpus, list)
TOTAL_ARTICLES = len(corpus)
print("Dataset:", HF_DATASET_REPO)
print("File:", INPUT_FILE)
print("T?ng s? b?i:", TOTAL_ARTICLES)
assert TOTAL_ARTICLES == 13572
```

## Cell 7 ? Ph?c h?i checkpoint c? t? Hugging Face

```python
import shutil
from huggingface_hub import snapshot_download

REMOTE_RUN_DIR = "generated_claim/qwen3_14b_awq"
CHECKPOINT_ROOT = Path("/kaggle/working/qwen3_14b_awq_checkpoint")
OUTPUTS_DIR = CHECKPOINT_ROOT / "outputs"
FAILED_DIR = CHECKPOINT_ROOT / "failed"
DEBUG_DIR = CHECKPOINT_ROOT / "debug"
DONE_DIR = CHECKPOINT_ROOT / "done"
TEMP_DIR = CHECKPOINT_ROOT / "temp"
for directory in [OUTPUTS_DIR, FAILED_DIR, DEBUG_DIR, DONE_DIR, TEMP_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

try:
    snapshot_path = Path(snapshot_download(
        repo_id=HF_DATASET_REPO,
        repo_type="dataset",
        token=HF_TOKEN,
        allow_patterns=[f"{REMOTE_RUN_DIR}/**"],
    ))
    remote_checkpoint = snapshot_path / REMOTE_RUN_DIR
    if remote_checkpoint.is_dir():
        shutil.copytree(remote_checkpoint, CHECKPOINT_ROOT, dirs_exist_ok=True)
        print("?? ph?c h?i checkpoint t? Hugging Face.")
    else:
        print("Ch?a c? checkpoint. S? ch?y t? ??u.")
except Exception as error:
    print("Kh?ng t?i ???c checkpoint:", type(error).__name__, str(error)[:300])
    print("N?u ??y l? l?n ??u th? c? th? b? qua.")

print("Chunk ho?n t?t:", len(list(DONE_DIR.glob("*.done"))))
print("File output:", len(list(OUTPUTS_DIR.glob("*.json"))))
```

## Cell 8 ? T?i Qwen3-14B-AWQ (kho?ng 10 GB)

```python
import os
import subprocess
import sys

MODEL_ID = "Qwen/Qwen3-14B-AWQ"
HF_MODEL_CACHE = Path("/kaggle/working/hf_cache/hub")
HF_MODEL_CACHE.mkdir(parents=True, exist_ok=True)
MODEL_PATH_RECORD = Path("/kaggle/working/qwen3_14b_awq_model_path.txt")

download_code = r'''
import os
from pathlib import Path
from huggingface_hub import snapshot_download
model_path = snapshot_download(
    repo_id=os.environ["MODEL_ID_TO_DOWNLOAD"],
    repo_type="model",
    token=os.environ["HF_TOKEN"],
    cache_dir=os.environ["HF_MODEL_CACHE"],
    max_workers=1,
    force_download=False,
)
Path(os.environ["MODEL_PATH_RECORD"]).write_text(model_path, encoding="utf-8")
print("MODEL_DOWNLOAD_COMPLETE:", model_path)
'''

download_env = os.environ.copy()
download_env["MODEL_ID_TO_DOWNLOAD"] = MODEL_ID
download_env["HF_MODEL_CACHE"] = str(HF_MODEL_CACHE)
download_env["MODEL_PATH_RECORD"] = str(MODEL_PATH_RECORD)
download_env["HF_HUB_DISABLE_XET"] = "1"
print("?ang t?i Qwen3-14B-AWQ...")
subprocess.run([sys.executable, "-u", "-c", download_code], env=download_env, check=True)

MODEL_PATH = Path(MODEL_PATH_RECORD.read_text(encoding="utf-8").strip())
assert (MODEL_PATH / "config.json").is_file()
assert (MODEL_PATH / "model.safetensors.index.json").is_file()
weight_files = list(MODEL_PATH.glob("*.safetensors"))
weight_size_gib = sum(file.stat().st_size for file in weight_files) / 1024**3
print("Model path:", MODEL_PATH)
print("S? shard:", len(weight_files))
print(f"Dung l??ng weights: {weight_size_gib:.2f} GiB")
```

## Cell 9 ? Kh?i ??ng vLLM tr?n T4 x2

```python
import time
import requests

MODEL = "Qwen/Qwen3-14B-AWQ"
MODEL_PATH_RECORD = Path("/kaggle/working/qwen3_14b_awq_model_path.txt")
assert MODEL_PATH_RECORD.is_file(), "Ch?a t?i model th?nh c?ng"
MODEL_PATH = Path(MODEL_PATH_RECORD.read_text(encoding="utf-8").strip())
VLLM_LOG = Path("/kaggle/working/vllm_qwen3_14b_awq.log")

os.environ["MODEL_ID"] = MODEL
os.environ["VLLM_API_URL"] = "http://127.0.0.1:8000/v1/chat/completions"
os.environ["MAX_INPUT_CHARS"] = "16000"
os.environ["MAX_OUTPUT_TOKENS"] = "3072"
os.environ["VLLM_REQUEST_TIMEOUT"] = "1800"

server_env = os.environ.copy()
server_env["NCCL_P2P_DISABLE"] = "1"
server_env["NCCL_IB_DISABLE"] = "1"
server_env["HF_HUB_DISABLE_XET"] = "1"
server_env.pop("VLLM_USE_V2_MODEL_RUNNER", None)

command = [
    "vllm", "serve", str(MODEL_PATH),
    "--host", "127.0.0.1",
    "--port", "8000",
    "--served-model-name", MODEL,
    "--tensor-parallel-size", "2",
    "--dtype", "float16",
    "--quantization", "awq",
    "--max-model-len", "16384",
    "--max-num-seqs", "1",
    "--gpu-memory-utilization", "0.90",
    "--default-chat-template-kwargs", '{"enable_thinking": false}',
    "--disable-custom-all-reduce",
]

def get_running_model():
    try:
        response = requests.get("http://127.0.0.1:8000/v1/models", timeout=5)
        if response.ok:
            models = response.json().get("data", [])
            if models:
                return models[0].get("id")
    except requests.RequestException:
        pass
    return None

running_model = get_running_model()
if running_model:
    if running_model != MODEL:
        raise RuntimeError(f"Port 8000 ?ang ch?y model kh?c: {running_model}")
    print("Model ?? ch?y s?n:", running_model)
    vllm_server = None
    vllm_log_handle = None
else:
    vllm_log_handle = VLLM_LOG.open("w", encoding="utf-8")
    vllm_server = subprocess.Popen(
        command, stdout=vllm_log_handle, stderr=subprocess.STDOUT, env=server_env
    )
    print("?ang n?p Qwen3-14B-AWQ l?n T4 x2...")
    for waited in range(30, 1201, 30):
        time.sleep(30)
        if vllm_server.poll() is not None:
            vllm_log_handle.flush()
            log_text = VLLM_LOG.read_text(encoding="utf-8", errors="replace")
            print(log_text[-30000:])
            raise RuntimeError("vLLM kh?i ??ng th?t b?i")
        if get_running_model() == MODEL:
            print(f"Model s?n s?ng sau kho?ng {waited} gi?y.")
            break
        print(f"?? ch? {waited} gi?y...")
    else:
        print(VLLM_LOG.read_text(encoding="utf-8", errors="replace")[-30000:])
        raise TimeoutError("Model ch?a s?n s?ng sau 20 ph?t")
```

## Cell 10 ? Ki?m tra server v? VRAM

```python
response = requests.get("http://127.0.0.1:8000/v1/models", timeout=30)
print("HTTP:", response.status_code)
print(response.json())
assert response.status_code == 200
```

```python
!nvidia-smi
```

## Cell 11 ? Ch?y th? 3 b?i

```python
TEST_OUTPUT = Path("/kaggle/working/test_3_qwen3_14b_awq.json")
TEST_FAILED = Path("/kaggle/working/test_3_qwen3_14b_awq_failed.json")
TEST_DEBUG = Path("/kaggle/working/test_3_qwen3_14b_awq_debug.log")

test_command = [
    sys.executable, "-u", str(SCRIPT_FILE),
    "--input-file", str(INPUT_FILE),
    "--output-file", str(TEST_OUTPUT),
    "--failed-ids-file", str(TEST_FAILED),
    "--debug-log-file", str(TEST_DEBUG),
    "--start-index", "0",
    "--limit", "3",
    "--checkpoint-every", "1",
]
subprocess.run(test_command, check=True, env=os.environ.copy())
print("K?t qu? test:", TEST_OUTPUT)
```

## Cell 12 ? Xem k?t qu? th?

```python
with TEST_OUTPUT.open("r", encoding="utf-8") as file:
    test_results = json.load(file)
print("S? b?i th?nh c?ng:", len(test_results))
for result in test_results:
    print("=" * 80)
    print("ID:", result.get("id"))
    print(json.dumps(result.get("claims"), ensure_ascii=False, indent=2)[:8000])
```

## Cell 13 ? Ch?y full 13.572 b?i v? t? l?u checkpoint l?n HF

```python
import math
from huggingface_hub import HfApi, CommitOperationAdd

CHUNK_SIZE = 200
SYNC_INTERVAL_SECONDS = 600
api = HfApi(token=HF_TOKEN)

def create_valid_snapshot(source_file):
    if not source_file.is_file():
        return None
    snapshot_file = TEMP_DIR / f"{source_file.name}.uploading"
    for _ in range(5):
        try:
            shutil.copy2(source_file, snapshot_file)
            with snapshot_file.open("r", encoding="utf-8") as file:
                json.load(file)
            return snapshot_file
        except (json.JSONDecodeError, OSError, PermissionError):
            time.sleep(2)
    if snapshot_file.exists():
        snapshot_file.unlink()
    return None

def upload_partial_checkpoint(output_file, remote_output_path):
    snapshot_file = create_valid_snapshot(output_file)
    if snapshot_file is None:
        print("Ch?a c? checkpoint JSON h?p l? ?? upload.")
        return False
    try:
        api.upload_file(
            repo_id=HF_DATASET_REPO,
            repo_type="dataset",
            path_or_fileobj=str(snapshot_file),
            path_in_repo=remote_output_path,
            commit_message=f"Partial Qwen3-14B-AWQ {output_file.stem}",
        )
        print("?? ??ng b? checkpoint t?m:", remote_output_path)
        return True
    except Exception as error:
        print("Upload checkpoint t?m th?t b?i:", type(error).__name__, str(error)[:300])
        return False
    finally:
        if snapshot_file.exists():
            snapshot_file.unlink()

def upload_completed_chunk(output_file, failed_file, debug_file, done_file):
    files_to_upload = [
        (output_file, f"{REMOTE_RUN_DIR}/outputs/{output_file.name}"),
        (failed_file, f"{REMOTE_RUN_DIR}/failed/{failed_file.name}"),
        (debug_file, f"{REMOTE_RUN_DIR}/debug/{debug_file.name}"),
        (done_file, f"{REMOTE_RUN_DIR}/done/{done_file.name}"),
    ]
    operations = [
        CommitOperationAdd(path_in_repo=remote, path_or_fileobj=str(local))
        for local, remote in files_to_upload if local.is_file()
    ]
    api.create_commit(
        repo_id=HF_DATASET_REPO,
        repo_type="dataset",
        operations=operations,
        commit_message=f"Complete Qwen3-14B-AWQ {output_file.stem}",
    )

number_of_chunks = math.ceil(TOTAL_ARTICLES / CHUNK_SIZE)
print("T?ng b?i:", TOTAL_ARTICLES)
print("K?ch th??c chunk:", CHUNK_SIZE)
print("T?ng chunk:", number_of_chunks)

for chunk_number, start_index in enumerate(range(0, TOTAL_ARTICLES, CHUNK_SIZE), start=1):
    current_limit = min(CHUNK_SIZE, TOTAL_ARTICLES - start_index)
    end_index = start_index + current_limit - 1
    chunk_name = f"chunk_{start_index:05d}_{end_index:05d}"
    output_file = OUTPUTS_DIR / f"{chunk_name}.json"
    failed_file = FAILED_DIR / f"{chunk_name}.json"
    debug_file = DEBUG_DIR / f"{chunk_name}.log"
    done_file = DONE_DIR / f"{chunk_name}.done"

    print("\n" + "=" * 80)
    print(f"Chunk {chunk_number}/{number_of_chunks}: m?u {start_index + 1} ? {end_index + 1}")
    if done_file.is_file():
        print("Chunk ?? ho?n t?t ? b? qua.")
        continue

    run_command = [
        sys.executable, "-u", str(SCRIPT_FILE),
        "--input-file", str(INPUT_FILE),
        "--output-file", str(output_file),
        "--failed-ids-file", str(failed_file),
        "--debug-log-file", str(debug_file),
        "--start-index", str(start_index),
        "--limit", str(current_limit),
        "--checkpoint-every", "1",
    ]
    run_env = os.environ.copy()
    run_env["PYTHONUNBUFFERED"] = "1"
    claim_process = subprocess.Popen(run_command, env=run_env)
    next_sync_time = time.monotonic() + SYNC_INTERVAL_SECONDS

    while claim_process.poll() is None:
        time.sleep(30)
        if time.monotonic() >= next_sync_time:
            upload_partial_checkpoint(
                output_file, f"{REMOTE_RUN_DIR}/outputs/{output_file.name}"
            )
            next_sync_time = time.monotonic() + SYNC_INTERVAL_SECONDS

    if claim_process.returncode != 0:
        upload_partial_checkpoint(
            output_file, f"{REMOTE_RUN_DIR}/outputs/{output_file.name}"
        )
        raise RuntimeError(f"Chunk {chunk_name} l?i: {claim_process.returncode}")

    done_file.write_text(json.dumps({
        "start_index": start_index,
        "end_index": end_index,
        "limit": current_limit,
        "model": MODEL,
        "completed_at_unix": time.time(),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    upload_completed_chunk(output_file, failed_file, debug_file, done_file)
    print("?? ho?n t?t v? upload:", chunk_name)

print("\n?? X? L? XONG T?T C? CHUNK.")
```

## Cell 14 ? Gh?p k?t qu? v? upload file cu?i

```python
expected_chunks = math.ceil(TOTAL_ARTICLES / CHUNK_SIZE)
completed_chunks = len(list(DONE_DIR.glob("*.done")))
print("Chunk ho?n t?t:", completed_chunks, "/", expected_chunks)
if completed_chunks != expected_chunks:
    raise RuntimeError("Ch?a ho?n t?t to?n b? chunk; h?y ch?y l?i Cell 13")

all_results = []
seen_ids = set()
for chunk_file in sorted(OUTPUTS_DIR.glob("*.json")):
    with chunk_file.open("r", encoding="utf-8") as file:
        chunk_results = json.load(file)
    for result in chunk_results:
        article_id = str(result.get("id", ""))
        if article_id and article_id not in seen_ids:
            seen_ids.add(article_id)
            all_results.append(result)

FINAL_OUTPUT = Path("/kaggle/working/claims_corpus_v1_qwen3_14b_awq.json")
FINAL_OUTPUT.write_text(
    json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8"
)

all_failed = []
for failed_file in sorted(FAILED_DIR.glob("*.json")):
    try:
        with failed_file.open("r", encoding="utf-8") as file:
            failures = json.load(file)
        if isinstance(failures, list):
            all_failed.extend(failures)
    except json.JSONDecodeError:
        pass
FINAL_FAILED = Path("/kaggle/working/failed_ids_qwen3_14b_awq.json")
FINAL_FAILED.write_text(
    json.dumps(all_failed, ensure_ascii=False, indent=2), encoding="utf-8"
)

operations = [
    CommitOperationAdd(
        path_in_repo=f"{REMOTE_RUN_DIR}/final/{FINAL_OUTPUT.name}",
        path_or_fileobj=str(FINAL_OUTPUT),
    ),
    CommitOperationAdd(
        path_in_repo=f"{REMOTE_RUN_DIR}/final/{FINAL_FAILED.name}",
        path_or_fileobj=str(FINAL_FAILED),
    ),
]
api.create_commit(
    repo_id=HF_DATASET_REPO,
    repo_type="dataset",
    operations=operations,
    commit_message="Upload final Qwen3-14B-AWQ claim dataset",
)
print("T?ng corpus:", TOTAL_ARTICLES)
print("K?t qu? h?p l?:", len(all_results))
print("Failed:", len(all_failed))
print("Output:", FINAL_OUTPUT)
print(f"https://huggingface.co/datasets/{HF_DATASET_REPO}/tree/main/{REMOTE_RUN_DIR}/final")
```

## Cell 15 ? T?i file k?t qu? v? m?y

```python
from IPython.display import FileLink, display

display(FileLink(str(FINAL_OUTPUT)))
display(FileLink(str(FINAL_FAILED)))
```
