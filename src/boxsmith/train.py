"""Train with: python -m boxsmith.train --config configs/tiny-shapes.yaml."""

import argparse
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import yaml

from boxsmith.data import GENERATOR_VERSION, SceneConfig, ShapesDataset, collate
from boxsmith.matching import assign, detection_loss
from boxsmith.metrics import evaluate
from boxsmith.model import TinyDetector
from boxsmith.predict import predict_dataset, suppress


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/tiny-shapes.yaml")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="runs/baseline")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--train-count", type=int)
    parser.add_argument("--val-count", type=int)
    parser.add_argument("--overfit", action="store_true", help="Evaluate on training scenes for a diagnostic only")
    args = parser.parse_args()
    output = Path(args.output)
    if any((output / filename).exists() for filename in ("history.jsonl", "best.pt", "latest.pt")):
        parser.error("Output already contains a run; choose a new --output directory")
    config = yaml.safe_load(Path(args.config).read_text())
    for key in ("epochs", "train_count", "val_count"):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    if min(config["epochs"], config["train_count"], config["val_count"]) < 1:
        parser.error("Epochs and dataset counts must be positive")
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(config.get("threads", 4))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    scene_config = SceneConfig(**config.get("scene", {}))
    train = ShapesDataset(config["train_count"], "train", seed, scene_config)
    validation = ShapesDataset(min(config["val_count"], len(train)) if args.overfit else config["val_count"],
                               "train" if args.overfit else "val", seed, scene_config)
    model = TinyDetector(scene_config.image_size).to(args.device)
    anchors = model.anchors.cpu().numpy()
    # Small generated training set is cached in RAM only, avoiding repeated rendering.
    examples = [train[i] for i in range(len(train))]
    assignments = [assign(anchors, target) for _, target in examples]
    loader = torch.utils.data.DataLoader(list(range(len(train))), batch_size=config["batch_size"],
                                         shuffle=True, generator=torch.Generator().manual_seed(seed))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config["epochs"], eta_min=0.0001)
    output.mkdir(parents=True, exist_ok=True)
    metadata = {"config": config, "scene_config": asdict(scene_config),
                "generator_version": GENERATOR_VERSION, "image_size": scene_config.image_size,
                "evaluation_split": validation.split,
                "parameter_count": sum(p.numel() for p in model.parameters())}
    (output / "config.json").write_text(json.dumps(metadata, indent=2))
    baseline_predictions, baseline_targets = predict_dataset(model, validation, args.device)
    baseline = evaluate(suppress(baseline_predictions), baseline_targets)["map"]
    print(json.dumps({"untrained_map50": baseline, "parameters": metadata["parameter_count"]}), flush=True)
    best, started = -1.0, time.perf_counter()
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        totals = {k: 0.0 for k in ("total", "objectness", "classification", "box")}
        for indices in loader:
            ids = indices.tolist()
            images = torch.stack([examples[i][0] for i in ids]).to(args.device)
            optimizer.zero_grad(set_to_none=True)
            losses = detection_loss(model(images), [assignments[i] for i in ids])
            if not torch.isfinite(losses["total"]):
                raise RuntimeError("Non-finite training loss")
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()
            for key in totals:
                totals[key] += float(losses[key].detach())
        scheduler.step()
        record = {"epoch": epoch, "seconds": round(time.perf_counter() - started, 2),
                  **{k: v / len(loader) for k, v in totals.items()}}
        if epoch == 1 or epoch % config.get("eval_every", 2) == 0 or epoch == config["epochs"]:
            predictions, targets = predict_dataset(model, validation, args.device)
            record["val_map50"] = evaluate(suppress(predictions), targets)["map"]
        payload = {**metadata, "model": model.state_dict(), "epoch": epoch,
                   "untrained_map50": baseline, "record": record}
        torch.save(payload, output / "latest.pt")
        if record.get("val_map50", -1) > best:
            best = record["val_map50"]
            torch.save(payload, output / "best.pt")
        with (output / "history.jsonl").open("a") as log:
            log.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
