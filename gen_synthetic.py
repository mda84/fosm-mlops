"""Synthetic data generator for fiber-optic sensing signals."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

RNG_SEED = 42


@dataclass
class EventConfig:
    name: str
    probability: float
    description: str


EVENTS: List[EventConfig] = [
    EventConfig("normal", 0.7, "Nominal flow and pump cycles"),
    EventConfig("leak", 0.1, "Short duration leak bursts"),
    EventConfig("interference", 0.1, "Third-party interference transients"),
    EventConfig("ground_movement", 0.1, "Long duration low frequency movement"),
]


def generate_event_window(
    base_time: float,
    sample_rate: float,
    duration: float,
    event: str,
    sensor_id: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a signal window for a specific event."""
    num_samples = int(duration * sample_rate)
    t = np.arange(num_samples) / sample_rate
    noise = rng.normal(0.0, 0.2, size=num_samples)

    if event == "normal":
        signal = 0.5 * np.sin(2 * math.pi * 1.5 * t) + noise
    elif event == "leak":
        burst = np.exp(-((t - duration / 2) ** 2) / (2 * 0.01))
        signal = 3.0 * burst * np.sin(2 * math.pi * 30 * t) + noise
    elif event == "interference":
        freqs = rng.uniform(5, 40, size=3)
        signal = sum(np.sin(2 * math.pi * f * t) for f in freqs)
        signal += 0.3 * rng.standard_t(df=2, size=num_samples)
    elif event == "ground_movement":
        signal = 1.5 * np.sin(2 * math.pi * 0.2 * t) + 0.2 * np.sin(2 * math.pi * 1.0 * t)
        signal += rng.normal(0.0, 0.1, size=num_samples)
    else:
        raise ValueError(f"Unknown event type: {event}")

    times = base_time + t
    return times, signal


def synthesize_dataset(
    sensors: int,
    duration: float,
    sample_rate: float,
    imbalance: Dict[str, float],
    output_dir: Path,
) -> None:
    rng = np.random.default_rng(RNG_SEED)
    total_samples = int(duration * sample_rate)

    raw_records: List[Dict[str, float]] = []
    label_records: List[Dict[str, float]] = []

    for sensor in range(sensors):
        current_time = 0.0
        while current_time < duration:
            event = rng.choice([e.name for e in EVENTS], p=[imbalance.get(e.name, e.probability) for e in EVENTS])
            window_duration = rng.uniform(2.0, 5.0) if event != "ground_movement" else rng.uniform(5.0, 10.0)
            times, signal = generate_event_window(
                current_time,
                sample_rate=sample_rate,
                duration=min(window_duration, duration - current_time),
                event=event,
                sensor_id=sensor,
                rng=rng,
            )
            for t, value in zip(times, signal, strict=False):
                raw_records.append({"time": t, "sensor_id": sensor, "value": value})
            label_records.append(
                {
                    "sensor_id": sensor,
                    "event": event,
                    "start_time": times[0],
                    "end_time": times[-1],
                }
            )
            current_time += window_duration

    raw_df = pd.DataFrame(raw_records)
    raw_df.sort_values(["sensor_id", "time"], inplace=True)
    label_df = pd.DataFrame(label_records)

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "data" / "raw"
    label_path = output_dir / "data" / "labels"
    raw_path.mkdir(parents=True, exist_ok=True)
    label_path.mkdir(parents=True, exist_ok=True)

    raw_file = raw_path / "synthetic_signals.parquet"
    label_file = label_path / "synthetic_labels.csv"
    raw_df.to_parquet(raw_file)
    label_df.to_csv(label_file, index=False)

    manifest = {
        "sensors": sensors,
        "duration": duration,
        "sample_rate": sample_rate,
        "records": len(raw_df),
        "events": label_df["event"].value_counts().to_dict(),
        "raw_path": str(raw_file),
        "label_path": str(label_file),
    }
    with (output_dir / "data" / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Generated dataset with {len(raw_df)} samples and {len(label_df)} events")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic fiber-optic sensing data")
    parser.add_argument("--sensors", type=int, default=3)
    parser.add_argument("--duration", type=float, default=300.0, help="Duration in seconds")
    parser.add_argument("--sample-rate", type=float, default=200.0, help="Samples per second")
    parser.add_argument(
        "--imbalance",
        type=str,
        default="",
        help="JSON string of event probabilities, e.g. '{\"leak\": 0.2}'",
    )
    parser.add_argument("--output", type=str, default=".")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    imbalance = {e.name: e.probability for e in EVENTS}
    if args.imbalance:
        imbalance.update(json.loads(args.imbalance))
    synthesize_dataset(
        sensors=args.sensors,
        duration=args.duration,
        sample_rate=args.sample_rate,
        imbalance=imbalance,
        output_dir=Path(args.output),
    )


if __name__ == "__main__":
    main()
