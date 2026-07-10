import orjson
import torch
from ml_dtypes import bfloat16
from sentence_transformers import SentenceTransformer


class TextEmbedder:
    def __init__(self, batch_size, embedding_model, device):
        device_type = torch.device(device).type
        self.model = SentenceTransformer(
            f"sentence-transformers/{embedding_model}",
            device=device,
            model_kwargs={
                "dtype": torch.bfloat16 if device_type == "cuda" else torch.float32,
            },
        )
        self.batch_size = batch_size

    def __call__(self, text_list, device):
        if isinstance(device, list):
            # Multi-process path returns fp32 numpy regardless of flags.
            emb = self.model.encode(
                text_list,
                batch_size=self.batch_size,
                show_progress_bar=True,
                device=device,
            )
            return emb.astype(bfloat16)
        emb = self.model.encode(
            text_list,
            batch_size=self.batch_size,
            convert_to_numpy=False,
            convert_to_tensor=True,
            show_progress_bar=True,
            device=device,
        )
        # bf16 → int16 bitcast so torch .numpy() accepts it, then relabel as bf16.
        # On CPU the SBERT model loaded with fp32 (line 15), so cast first —
        # the bitcast on raw fp32 silently misinterprets 4-byte floats as 2×bf16
        # garbage and writes a .bin with NaN/inf bit patterns.
        return emb.to(torch.bfloat16).cpu().view(torch.int16).numpy().view(bfloat16)


def main(
    dataset_name,
    pre_dir: str,
    device,
    batch_size,
    embedding_model,
):
    if device is None:
        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            # Pass a string for 1 GPU. A list of len 1 routes SBERT into its
            # multi-process path, which skips length-sorted batching.
            device = [f"cuda:{i}" for i in range(n)] if n > 1 else "cuda:0"
            print(f"Using device(s): {device}")
        else:
            device = "cpu"

    init_device = device[0] if isinstance(device, list) else device

    text_path = f"{pre_dir}/{dataset_name}/text.json"
    with open(text_path) as f:
        raw = f.read()
    text_list = orjson.loads(raw)
    print(f"Loaded {len(text_list)} texts from {text_path}")

    text_embedder = TextEmbedder(batch_size, embedding_model, init_device)
    emb = text_embedder(text_list, device=device)

    emb_path = f"{pre_dir}/{dataset_name}/text_emb_{embedding_model}.bin"
    emb.tofile(emb_path)
    print(f"Wrote {emb.shape} {emb.dtype} to {emb_path}")


if __name__ == "__main__":
    # Lazy: strictfire imports the stdlib `pipes` (gone in 3.13); a top-level
    # import would cap the package at 3.12.
    import strictfire

    strictfire.StrictFire(main)
