"""Download files from the Hugging Face Hub at a fixed revision, and nothing more than needed."""

import hashlib
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download, snapshot_download

# Model repositories often hold the same weights several times (PyTorch,
# safetensors, ONNX, OpenVINO, TensorFlow). Only one copy is downloaded.
SKIP = ["onnx/*", "openvino/*", "*.h5", "*.msgpack", "*.ot", "imgs/*", "images/*", "*.jpg", "*.png", "*.webp"]


def model_files(repo: str, revision: str) -> list[str]:
    """Return the patterns of the files a model needs: safetensors if there are any, else the .bin file."""
    names = [s.rfilename for s in HfApi().model_info(repo, revision=revision).siblings]
    has_safetensors = any(n.endswith(".safetensors") for n in names)
    skip = SKIP + (["*.bin", "*.pt"] if has_safetensors else ["*.pt"])
    return skip


def fetch_model(repo: str, revision: str) -> Path:
    """Download a model repository at `revision` and return the local folder."""
    return Path(snapshot_download(repo, revision=revision, ignore_patterns=model_files(repo, revision)))


def fetch_dataset_file(repo: str, revision: str, filename: str, sha256: str | None = None) -> Path:
    """Download one file of a dataset repository and check its SHA-256 if one is given."""
    path = Path(hf_hub_download(repo, filename, revision=revision, repo_type="dataset"))
    if sha256 is not None and file_sha256(path) != sha256:
        raise ValueError(f"{repo}/{filename} at {revision} does not have the expected SHA-256")
    return path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    import sys

    from ragbench import settings

    # python -m ragbench.fetch [key ...]: download the models with these keys, or all of them.
    models = settings.EMBEDDERS + settings.RERANKERS + settings.GUARDS
    wanted = sys.argv[1:] or [m.key for m in models]
    for key in wanted:
        model = next(m for m in models if m.key == key)
        print(model.repo, fetch_model(model.repo, model.revision), flush=True)
