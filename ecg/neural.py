"""A compact 1D CNN with one embedding per complete input recording."""

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from ecg.augment import augment_training
from ecg.preprocess import load_prepared
from ecg.splits import indices


class ECGEncoder(nn.Module):
    def __init__(self, classes: int, embedding_size: int = 64):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Conv1d(1, 16, kernel_size=15, stride=2, padding=7),
            nn.Conv1d(16, 32, kernel_size=11, stride=2, padding=5),
            nn.Conv1d(32, 64, kernel_size=9, stride=2, padding=4),
        ])
        self.projection = nn.Linear(128, embedding_size)
        self.classifier = nn.Linear(embedding_size, classes)

    def encode(self, signal, lengths):
        x = signal[:, None, :]
        for layer in self.layers:
            x = layer(x).relu()
            lengths = (lengths + 1) // 2
            keep = torch.arange(x.shape[-1], device=x.device)[None, :] < lengths[:, None]
            x = x * keep[:, None, :]
        average = x.sum(dim=-1) / lengths[:, None]
        maximum = x.amax(dim=-1)
        return self.projection(torch.cat([average, maximum], dim=1))

    def forward(self, signal, lengths):
        return self.classifier(self.encode(signal, lengths))


def collate(batch):
    arrays, targets = zip(*batch)
    lengths = torch.tensor([len(array) for array in arrays])
    signals = nn.utils.rnn.pad_sequence([torch.from_numpy(array) for array in arrays], batch_first=True)
    return signals, lengths, torch.tensor(targets)


def select_device(choice: str) -> torch.device:
    available = {"cpu": True, "mps": torch.backends.mps.is_available(), "cuda": torch.cuda.is_available()}
    if choice == "auto":
        choice = next(name for name in ("cuda", "mps", "cpu") if available[name])
    if not available.get(choice, False):
        raise ValueError(f"{choice} is unavailable in this PyTorch environment. Available: {available}")
    return torch.device(choice)


@torch.inference_mode()
def encode_signals(model, signals, device, batch_size):
    model.eval()
    encoding = []
    for x, lengths, _ in DataLoader([(signal, -1) for signal in signals], batch_size=batch_size,
                                    collate_fn=collate):
        encoded = model.encode(x.to(device), lengths.to(device))
        encoding.append(nn.functional.normalize(encoded, dim=1).cpu().numpy())
    return np.concatenate(encoding)


def encode_checkpoint(path, signals, metadata):
    model = ECGEncoder(len(metadata["participants"]), metadata["embedding_size"])
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return encode_signals(model, signals, torch.device("cpu"), metadata["batch_size"])


def train(path: Path, protocol: str, output: Path, choice: str, epochs: int,
          batch_size: int, seed: int, augment: bool = False) -> None:
    if epochs < 1 or batch_size < 1:
        raise ValueError("Epochs and batch size must be positive")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = select_device(choice)
    records, signals = load_prepared(path)
    train_indices = indices(path, protocol, records, "train")
    participants = sorted({records[i]["participant"] for i in train_indices})
    labels = {p: i for i, p in enumerate(participants)}
    train_examples = [(signals[i], labels[records[i]["participant"]]) for i in train_indices]
    aug_rng = np.random.default_rng(seed)

    def collate_training(batch):
        return collate([(augment_training(signal, aug_rng), label) for signal, label in batch])

    loader = DataLoader(train_examples, shuffle=True, batch_size=batch_size,
                        collate_fn=collate_training if augment else collate,
                        generator=torch.Generator().manual_seed(seed))
    model = ECGEncoder(len(labels)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    history = []
    print(f"Device: {device}; training records: {len(train_indices)}; identities: {len(labels)}", flush=True)
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        correct = 0
        for x, lengths, y in loader:
            x, lengths, y = x.to(device), lengths.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x, lengths)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach().cpu()) * len(y)
            correct += int((logits.argmax(dim=1) == y).sum().cpu())
        row = dict(epoch=epoch, train_loss=epoch_loss / len(train_examples),
                   train_accuracy=correct / len(train_examples))
        history.append(row)
        print(json.dumps(row), flush=True)

    # Always export clean embeddings for enrollment and validation.
    encoding = encode_signals(model, signals, device, batch_size)
    output.mkdir(parents=True, exist_ok=True)
    torch.save({name: value.cpu() for name, value in model.state_dict().items()}, output / "model.pt")
    np.savez_compressed(output / "embeddings.npz", record_ids=[r["record_id"] for r in records],
                        embeddings=encoding)
    metadata = dict(protocol=protocol, device=str(device), seed=seed, epochs=epochs, batch_size=batch_size,
                    augment=augment, training_records=len(train_indices), participants=participants,
                    training_record_ids=[records[i]["record_id"] for i in train_indices],
                    embedding_size=model.projection.out_features, history=history)
    (output / "training.json").write_text(json.dumps(metadata, indent=2) + "\n")
