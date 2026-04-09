"""Custom collation for variable-length text embeddings."""

import torch


def collate_fn(batch):
    """
    Pad variable-length text embeddings to the max seq_len in the batch.

    UMT5 text embeddings have different sequence lengths per caption.
    Zero-padding is safe because the DiT treats zeros as null tokens
    (same effect as CFG dropout).
    """
    keys = batch[0].keys()
    collated = {}
    for key in keys:
        vals = [sample[key] for sample in batch]
        if key == "text_embed":
            max_len = max(v.shape[0] for v in vals)
            padded = torch.zeros(len(vals), max_len, vals[0].shape[1])
            for i, v in enumerate(vals):
                padded[i, :v.shape[0]] = v
            collated[key] = padded
        elif isinstance(vals[0], torch.Tensor):
            collated[key] = torch.stack(vals)
        else:
            collated[key] = vals
    return collated
