import argparse
import csv
import json
import math
import random
import textwrap
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import h5py
except ImportError:
    h5py = None
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class Config:
    data_path: str = "viscodata_3mat.mat"
    output_dir: str = "outputs"
    seed: int = 42
    device: str = "auto"
    downsample: int = 2
    train_fraction: float = 0.70
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    batch_size: int = 32
    epochs: int = 140
    sweep_epochs: int = 100
    lr: float = 3.0e-3
    weight_decay: float = 1.0e-4
    patience: int = 20
    grad_clip: float = 1.0
    hidden_dim: int = 8
    hidden_sweep: Tuple[int, ...] = (1, 2, 3, 4, 6, 8, 12, 16)
    run_hidden_sweep: bool = True
    run_baselines: bool = True
    window_size: int = 15
    num_workers: int = 0
    figure_dpi: int = 220
    save_pdf_figures: bool = True
    rno_width: int = 64
    gru_width: int = 32
    gru_layers: int = 1
    baseline_width: int = 96


DEFAULT_CONFIG = Config()


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def ensure_dirs(output_dir: Path) -> Dict[str, Path]:
    paths = {
        "root": output_dir,
        "figures": output_dir / "figures",
        "tables": output_dir / "tables",
        "logs": output_dir / "logs",
        "checkpoints": output_dir / "checkpoints",
        "report_notes": output_dir / "report_notes",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def write_text(path: Path, text: str) -> None:
    path.write_text(text.strip() + "\n", encoding="utf-8")


def save_json(path: Path, payload: Dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_csv(path: Path, rows: List[Dict[str, object]], fieldnames: Optional[List[str]] = None) -> None:
    if not rows:
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def count_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


class MatReader:
    def __init__(self, file_path: str):
        self.file_path = str(file_path)
        self.data = None
        self.old_mat = None
        self._load_file()

    def _load_file(self) -> None:
        try:
            self.data = scipy.io.loadmat(self.file_path)
            self.old_mat = True
        except NotImplementedError:
            if h5py is None:
                raise ImportError(
                    "scipy.io.loadmat could not read the MATLAB file and h5py is not installed for HDF5 fallback."
                )
            self.data = h5py.File(self.file_path, "r")
            self.old_mat = False

    def keys(self) -> List[str]:
        if self.old_mat:
            return [key for key in self.data.keys() if not key.startswith("__")]
        return list(self.data.keys())

    def read_field(self, field: str) -> np.ndarray:
        if field not in self.data:
            raise KeyError(f"Field '{field}' not found. Available keys: {self.keys()}")
        array = self.data[field]
        if not self.old_mat:
            array = array[()]
            array = np.transpose(array, axes=range(len(array.shape) - 1, -1, -1))
        return np.asarray(array, dtype=np.float32)


def infer_signal_fields(reader: MatReader) -> Tuple[str, str]:
    keys = reader.keys()
    lower_map = {key.lower(): key for key in keys}
    strain_candidates = ["epsi_tol", "epsilon", "eps", "strain", "macro_strain"]
    stress_candidates = ["sigma_tol", "sigma", "stress", "macro_stress"]
    strain_field = next((lower_map[name] for name in strain_candidates if name in lower_map), None)
    stress_field = next((lower_map[name] for name in stress_candidates if name in lower_map), None)
    if strain_field is None or stress_field is None:
        raise ValueError(f"Could not infer strain/stress fields from MATLAB keys: {keys}")
    return strain_field, stress_field


def validate_signals(strain: np.ndarray, stress: np.ndarray) -> None:
    if strain.ndim != 2 or stress.ndim != 2:
        raise ValueError(f"Expected 2D arrays. Got strain {strain.shape} and stress {stress.shape}.")
    if strain.shape != stress.shape:
        raise ValueError(f"Strain and stress shapes must match. Got {strain.shape} and {stress.shape}.")


class StandardScaler:
    def __init__(self, eps: float = 1.0e-6):
        self.eps = eps
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None

    def fit(self, data: np.ndarray) -> None:
        self.mean = data.mean(axis=0, keepdims=True)
        self.std = data.std(axis=0, keepdims=True)
        self.std = np.where(self.std < self.eps, 1.0, self.std)

    def transform(self, data: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("Scaler must be fit before transform.")
        return (data - self.mean) / self.std

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("Scaler must be fit before inverse_transform.")
        return data * self.std + self.mean

    def state_dict(self) -> Dict[str, List[float]]:
        return {
            "mean": np.asarray(self.mean).reshape(-1).tolist(),
            "std": np.asarray(self.std).reshape(-1).tolist(),
        }


def make_splits(num_samples: int, cfg: Config) -> Dict[str, np.ndarray]:
    if not math.isclose(cfg.train_fraction + cfg.val_fraction + cfg.test_fraction, 1.0, abs_tol=1.0e-8):
        raise ValueError("Train/validation/test fractions must sum to 1.")
    rng = np.random.default_rng(cfg.seed)
    indices = np.arange(num_samples)
    rng.shuffle(indices)
    n_train = int(round(cfg.train_fraction * num_samples))
    n_val = int(round(cfg.val_fraction * num_samples))
    train_idx = np.sort(indices[:n_train])
    val_idx = np.sort(indices[n_train:n_train + n_val])
    test_idx = np.sort(indices[n_train + n_val:])
    return {"train": train_idx, "val": val_idx, "test": test_idx}


def save_split_indices(path: Path, splits: Dict[str, np.ndarray]) -> None:
    rows = []
    for split_name, indices in splits.items():
        for idx in indices.tolist():
            rows.append({"split": split_name, "sample_index": int(idx)})
    save_csv(path, rows, fieldnames=["split", "sample_index"])


class MLP(nn.Module):
    def __init__(self, widths: Sequence[int], activation: nn.Module = nn.GELU):
        super().__init__()
        layers: List[nn.Module] = []
        for i in range(len(widths) - 1):
            layers.append(nn.Linear(widths[i], widths[i + 1]))
            if i < len(widths) - 2:
                layers.append(activation())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RecurrentNeuralOperator1D(nn.Module):
    """Hidden state is interpreted as learned constitutive internal variables."""

    def __init__(self, feature_dim: int, hidden_dim: int, width: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.hidden_encoder = MLP([feature_dim + 1 + hidden_dim, width, width, hidden_dim], activation=nn.GELU)
        self.output_head = MLP([feature_dim + hidden_dim, width, width, 1], activation=nn.GELU)
        self.initial_head = MLP([feature_dim + hidden_dim, width, width, 1], activation=nn.GELU)

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_dim, device=device)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch_size, time_steps, _ = features.shape
        hidden = self.init_hidden(batch_size, features.device)
        outputs: List[torch.Tensor] = []

        prev_stress = self.initial_head(torch.cat([features[:, 0, :], hidden], dim=-1))
        outputs.append(prev_stress)

        for t in range(1, time_steps):
            current = features[:, t, :]
            update_input = torch.cat([current, prev_stress, hidden], dim=-1)
            delta_h = self.hidden_encoder(update_input)
            hidden = hidden + 0.1 * torch.tanh(delta_h)
            stress = self.output_head(torch.cat([current, hidden], dim=-1))
            outputs.append(stress)
            prev_stress = stress

        return torch.cat(outputs, dim=1)


class GRUConstitutiveModel(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, num_layers: int = 1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = MLP([hidden_dim, hidden_dim, 1], activation=nn.GELU)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        latent, _ = self.gru(features)
        return self.head(latent).squeeze(-1)


class WindowMLPConstitutiveModel(nn.Module):
    def __init__(self, feature_dim: int, window_size: int, width: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.window_size = window_size
        self.head = MLP([feature_dim * window_size, width, width, 1], activation=nn.GELU)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch_size, time_steps, feature_dim = features.shape
        windows = []
        for t in range(time_steps):
            start = max(0, t - self.window_size + 1)
            window = features[:, start:t + 1, :]
            if window.shape[1] < self.window_size:
                pad = torch.zeros(batch_size, self.window_size - window.shape[1], feature_dim, device=features.device)
                window = torch.cat([pad, window], dim=1)
            windows.append(window.reshape(batch_size, -1))
        stacked = torch.stack(windows, dim=1)
        return self.head(stacked).squeeze(-1)


def combined_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = F.mse_loss(pred, target)
    rel = torch.norm(pred - target, dim=1) / (torch.norm(target, dim=1) + 1.0e-8)
    return mse + 0.1 * rel.mean()


def create_loader(features: np.ndarray, targets: np.ndarray, indices: np.ndarray, cfg: Config, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(features[indices]).float(),
        torch.from_numpy(targets[indices]).float(),
    )
    generator = torch.Generator()
    generator.manual_seed(cfg.seed)
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )


def evaluate_loss(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for features, targets in loader:
            features = features.to(device)
            targets = targets.to(device)
            predictions = model(features)
            loss = combined_loss(predictions, targets)
            total_loss += loss.item() * features.size(0)
            total_samples += features.size(0)
    return total_loss / max(total_samples, 1)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    cfg: Config,
    checkpoint_path: Path,
    epochs: int,
) -> Dict[str, object]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=max(3, cfg.patience // 3),
    )

    history = {"epoch": [], "train_loss": [], "val_loss": [], "lr": []}
    best_val = float("inf")
    best_epoch = -1
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        total_samples = 0
        for features, targets in train_loader:
            features = features.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(features)
            loss = combined_loss(predictions, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip)
            optimizer.step()
            running_loss += loss.item() * features.size(0)
            total_samples += features.size(0)

        train_loss = running_loss / max(total_samples, 1)
        val_loss = evaluate_loss(model, val_loader, device)
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(current_lr)

        if val_loss < best_val - 1.0e-6:
            best_val = val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save({"model_state": model.state_dict(), "best_val_loss": best_val, "best_epoch": best_epoch}, checkpoint_path)
        else:
            patience_counter += 1

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(
                f"[{timestamp()}] epoch={epoch:03d} train_loss={train_loss:.6f} "
                f"val_loss={val_loss:.6f} lr={current_lr:.2e}"
            )
        if patience_counter >= cfg.patience:
            print(f"[{timestamp()}] Early stopping triggered at epoch {epoch}.")
            break

    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model_state"])
    return {
        "history": history,
        "best_val_loss": float(state["best_val_loss"]),
        "best_epoch": int(state["best_epoch"]),
    }


def inverse_stress(stress_norm: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    flat = stress_norm.reshape(-1, 1)
    restored = scaler.inverse_transform(flat)
    return restored.reshape(stress_norm.shape)


def collect_predictions(model: nn.Module, features: np.ndarray, device: torch.device, batch_size: int = 64) -> np.ndarray:
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            batch = torch.from_numpy(features[start:start + batch_size]).float().to(device)
            outputs.append(model(batch).cpu().numpy())
    return np.concatenate(outputs, axis=0)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    diff = y_pred - y_true
    mse = float(np.mean(diff ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(diff)))
    rel_l2 = float(np.linalg.norm(diff) / (np.linalg.norm(y_true) + 1.0e-12))
    ss_res = float(np.sum(diff ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - ss_res / (ss_tot + 1.0e-12))
    return {"mse": mse, "rmse": rmse, "mae": mae, "rel_l2": rel_l2, "r2": r2}


def compute_per_sample_relative_error(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    numerator = np.linalg.norm(y_pred - y_true, axis=1)
    denominator = np.linalg.norm(y_true, axis=1) + 1.0e-12
    return numerator / denominator


def summarise_split_metrics(y_true_norm: np.ndarray, y_pred_norm: np.ndarray, stress_scaler: StandardScaler) -> Dict[str, float]:
    y_true_phys = inverse_stress(y_true_norm, stress_scaler)
    y_pred_phys = inverse_stress(y_pred_norm, stress_scaler)
    metrics_norm = compute_metrics(y_true_norm, y_pred_norm)
    metrics_phys = compute_metrics(y_true_phys, y_pred_phys)
    summary = {}
    for key, value in metrics_norm.items():
        summary[f"{key}_norm"] = value
    for key, value in metrics_phys.items():
        summary[f"{key}_phys"] = value
    return summary


def setup_plotting() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "lines.linewidth": 2.0,
            "figure.figsize": (7.0, 4.5),
        }
    )


def save_figure(fig: plt.Figure, path_png: Path, save_pdf: bool, dpi: int) -> None:
    fig.tight_layout()
    fig.savefig(path_png, dpi=dpi, bbox_inches="tight")
    if save_pdf:
        fig.savefig(path_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_loss_curves(history: Dict[str, List[float]], title: str, out_path: Path, cfg: Config) -> None:
    fig, ax = plt.subplots()
    ax.plot(history["epoch"], history["train_loss"], label="Train loss")
    ax.plot(history["epoch"], history["val_loss"], label="Validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend(frameon=True)
    save_figure(fig, out_path, cfg.save_pdf_figures, cfg.figure_dpi)


def plot_trajectory_examples(
    time_axis: np.ndarray,
    strain_phys: np.ndarray,
    stress_true_phys: np.ndarray,
    stress_pred_phys: np.ndarray,
    sample_indices: Sequence[int],
    labels: Sequence[str],
    prefix: str,
    fig_dir: Path,
    cfg: Config,
) -> None:
    for sample_index, label in zip(sample_indices, labels):
        fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
        axes[0].plot(time_axis, stress_true_phys[sample_index], label="True stress")
        axes[0].plot(time_axis, stress_pred_phys[sample_index], "--", label="Predicted stress")
        axes[0].set_xlabel("Normalised time")
        axes[0].set_ylabel("Macroscopic stress")
        axes[0].set_title(f"{label.capitalize()} case: stress history")
        axes[0].legend(frameon=True)
        axes[1].plot(strain_phys[sample_index], stress_true_phys[sample_index], label="True loop")
        axes[1].plot(strain_phys[sample_index], stress_pred_phys[sample_index], "--", label="Predicted loop")
        axes[1].set_xlabel("Macroscopic strain")
        axes[1].set_ylabel("Macroscopic stress")
        axes[1].set_title(f"{label.capitalize()} case: stress-strain loop")
        axes[1].legend(frameon=True)
        save_figure(fig, fig_dir / f"{prefix}_{label}_sample_{sample_index:03d}.png", cfg.save_pdf_figures, cfg.figure_dpi)


def plot_parity(y_true_phys: np.ndarray, y_pred_phys: np.ndarray, out_path: Path, cfg: Config) -> None:
    fig, ax = plt.subplots(figsize=(5.4, 5.4))
    ax.scatter(y_true_phys.reshape(-1), y_pred_phys.reshape(-1), s=8, alpha=0.25, edgecolors="none")
    lower = float(min(y_true_phys.min(), y_pred_phys.min()))
    upper = float(max(y_true_phys.max(), y_pred_phys.max()))
    ax.plot([lower, upper], [lower, upper], "k--", linewidth=1.5, label="Ideal agreement")
    ax.set_xlabel("True stress")
    ax.set_ylabel("Predicted stress")
    ax.set_title("Parity plot on the held-out test set")
    ax.legend(frameon=True)
    save_figure(fig, out_path, cfg.save_pdf_figures, cfg.figure_dpi)


def plot_hidden_sweep(rows: List[Dict[str, object]], out_path: Path, cfg: Config) -> None:
    dims = [int(row["hidden_dim"]) for row in rows]
    val_rmse = [float(row["val_rmse_phys"]) for row in rows]
    test_rmse = [float(row["test_rmse_phys"]) for row in rows]
    fig, ax = plt.subplots()
    ax.plot(dims, val_rmse, marker="o", label="Validation RMSE")
    ax.plot(dims, test_rmse, marker="s", label="Test RMSE")
    ax.set_xlabel("Hidden/internal variable dimension")
    ax.set_ylabel("RMSE in physical stress units")
    ax.set_title("Hidden-state sweep for the recurrent neural operator")
    ax.legend(frameon=True)
    save_figure(fig, out_path, cfg.save_pdf_figures, cfg.figure_dpi)


def plot_baseline_comparison(rows: List[Dict[str, object]], out_path: Path, cfg: Config) -> None:
    names = [row["model"] for row in rows]
    rel_l2 = [float(row["test_rel_l2_phys"]) for row in rows]
    rmse = [float(row["test_rmse_phys"]) for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))
    axes[0].bar(names, rel_l2, color=["#4C78A8", "#59A14F", "#E15759"])
    axes[0].set_ylabel("Relative L2 error")
    axes[0].set_title("Test relative error")
    axes[0].tick_params(axis="x", rotation=12)
    axes[1].bar(names, rmse, color=["#4C78A8", "#59A14F", "#E15759"])
    axes[1].set_ylabel("RMSE")
    axes[1].set_title("Test RMSE in physical units")
    axes[1].tick_params(axis="x", rotation=12)
    save_figure(fig, out_path, cfg.save_pdf_figures, cfg.figure_dpi)


def select_representative_samples(relative_errors: np.ndarray) -> Tuple[List[int], List[str]]:
    ordering = np.argsort(relative_errors)
    best = int(ordering[0])
    median = int(ordering[len(ordering) // 2])
    worst = int(ordering[-1])
    return [best, median, worst], ["best", "median", "worst"]


def recommend_hidden_dimension(sweep_rows: List[Dict[str, object]], tolerance: float = 0.02) -> Dict[str, object]:
    best_val = min(float(row["val_rmse_phys"]) for row in sweep_rows)
    threshold = best_val * (1.0 + tolerance)
    eligible = [row for row in sweep_rows if float(row["val_rmse_phys"]) <= threshold]
    chosen = min(eligible, key=lambda row: int(row["hidden_dim"]))
    return {
        "best_val_rmse_phys": best_val,
        "threshold_rmse_phys": threshold,
        "recommended_hidden_dim": int(chosen["hidden_dim"]),
        "criterion": (
            "Smallest hidden dimension whose validation RMSE lies within "
            f"{100.0 * tolerance:.1f}% of the best validation RMSE."
        ),
    }


def instantiate_model(model_name: str, cfg: Config, hidden_dim: Optional[int] = None) -> nn.Module:
    if model_name == "RNO":
        return RecurrentNeuralOperator1D(feature_dim=2, hidden_dim=hidden_dim or cfg.hidden_dim, width=cfg.rno_width)
    if model_name == "GRU":
        return GRUConstitutiveModel(feature_dim=2, hidden_dim=cfg.gru_width, num_layers=cfg.gru_layers)
    if model_name == "WindowMLP":
        return WindowMLPConstitutiveModel(feature_dim=2, window_size=cfg.window_size, width=cfg.baseline_width)
    raise ValueError(f"Unknown model '{model_name}'.")


def run_single_experiment(
    model_name: str,
    cfg: Config,
    device: torch.device,
    features_norm: np.ndarray,
    stress_norm: np.ndarray,
    stress_scaler: StandardScaler,
    strain_phys: np.ndarray,
    time_axis: np.ndarray,
    splits: Dict[str, np.ndarray],
    output_paths: Dict[str, Path],
    hidden_dim: Optional[int] = None,
    epochs: Optional[int] = None,
    make_figures: bool = True,
) -> Dict[str, object]:
    model = instantiate_model(model_name, cfg, hidden_dim=hidden_dim).to(device)
    experiment_tag = model_name if hidden_dim is None else f"{model_name}_h{hidden_dim}"
    train_loader = create_loader(features_norm, stress_norm, splits["train"], cfg, shuffle=True)
    val_loader = create_loader(features_norm, stress_norm, splits["val"], cfg, shuffle=False)
    checkpoint_path = output_paths["checkpoints"] / f"{experiment_tag}.pt"

    result = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        cfg=cfg,
        checkpoint_path=checkpoint_path,
        epochs=epochs or cfg.epochs,
    )

    predictions = {}
    metrics_rows = []
    for split_name, split_indices in splits.items():
        y_true_norm = stress_norm[split_indices]
        y_pred_norm = collect_predictions(model, features_norm[split_indices], device, batch_size=cfg.batch_size)
        predictions[split_name] = {"y_true_norm": y_true_norm, "y_pred_norm": y_pred_norm}
        metric_summary = summarise_split_metrics(y_true_norm, y_pred_norm, stress_scaler)
        row = {"model": experiment_tag, "split": split_name}
        row.update(metric_summary)
        metrics_rows.append(row)

    save_csv(
        output_paths["logs"] / f"{experiment_tag}_history.csv",
        [
            {
                "epoch": int(epoch),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "lr": float(lr),
            }
            for epoch, train_loss, val_loss, lr in zip(
                result["history"]["epoch"],
                result["history"]["train_loss"],
                result["history"]["val_loss"],
                result["history"]["lr"],
            )
        ],
    )
    save_csv(output_paths["tables"] / f"{experiment_tag}_metrics.csv", metrics_rows)

    if make_figures:
        plot_loss_curves(
            result["history"],
            title=f"{experiment_tag}: training history",
            out_path=output_paths["figures"] / f"{experiment_tag}_loss_curves.png",
            cfg=cfg,
        )
        test_pred_norm = predictions["test"]["y_pred_norm"]
        test_true_norm = predictions["test"]["y_true_norm"]
        test_pred_phys = inverse_stress(test_pred_norm, stress_scaler)
        test_true_phys = inverse_stress(test_true_norm, stress_scaler)
        test_errors = compute_per_sample_relative_error(test_true_phys, test_pred_phys)
        selected_local, labels = select_representative_samples(test_errors)
        plot_trajectory_examples(
            time_axis=time_axis,
            strain_phys=strain_phys[splits["test"]],
            stress_true_phys=test_true_phys,
            stress_pred_phys=test_pred_phys,
            sample_indices=selected_local,
            labels=labels,
            prefix=experiment_tag,
            fig_dir=output_paths["figures"],
            cfg=cfg,
        )
        plot_parity(
            y_true_phys=test_true_phys,
            y_pred_phys=test_pred_phys,
            out_path=output_paths["figures"] / f"{experiment_tag}_parity.png",
            cfg=cfg,
        )

    test_metrics = next(row for row in metrics_rows if row["split"] == "test")
    val_metrics = next(row for row in metrics_rows if row["split"] == "val")
    return {
        "model_name": model_name,
        "experiment_tag": experiment_tag,
        "hidden_dim": hidden_dim,
        "parameter_count": count_parameters(model),
        "best_epoch": result["best_epoch"],
        "best_val_loss": result["best_val_loss"],
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "checkpoint_path": str(checkpoint_path),
    }


def write_problem_framing_note(path: Path, dt: float, time_steps: int, downsample: int) -> None:
    text = f"""
    Problem 1(a): constitutive model input/output framing

    The learned constitutive model is treated as a causal sequence-to-sequence map at the macroscopic scale.
    At each time step t_n, the observable input is the current macroscopic strain together with a backward
    finite-difference approximation to the strain rate. The model output is the current macroscopic stress.

    A recurrent hidden state is evolved alongside the observable loading variables. In mechanics terms, this
    hidden state plays the role of learned internal variables that encode constitutive memory. The stress at the
    current step therefore depends on both the current loading state and the accumulated hidden state that stores
    history effects. This is consistent with the hereditary character expected from viscoelastic composites.

    The data were downsampled by a factor of {downsample}, giving {time_steps} time points and a normalised time
    increment dt = {dt:.6f}. Downsampling is used only to reduce computational cost while retaining enough temporal
    resolution for the observed loading histories.
    """
    write_text(path, textwrap.dedent(text))


def write_hidden_interpretation_note(path: Path, recommendation: Dict[str, object], sweep_rows: List[Dict[str, object]]) -> None:
    best_row = min(sweep_rows, key=lambda row: float(row["val_rmse_phys"]))
    text = f"""
    Problem 1(c): interpretation of the hidden-variable sweep

    In the recurrent neural operator, the hidden state is interpreted as a learned set of internal variables that
    compactly stores constitutive memory. The hidden dimension is therefore a numerical proxy for the number of
    macroscopic memory variables required by the data-driven constitutive model.

    The minimum hidden dimension is inferred empirically rather than proved analytically. The sweep compares
    multiple hidden dimensions under the same training and validation protocol. The selected recommendation is
    hidden dimension = {recommendation['recommended_hidden_dim']}, using the criterion:
    {recommendation['criterion']}

    The best validation RMSE obtained anywhere in the sweep was
    {recommendation['best_val_rmse_phys']:.6e}, achieved at hidden dimension {int(best_row['hidden_dim'])}.
    The recommended smaller dimension is preferred when it already lies on the performance plateau, so increasing
    the hidden-state size further yields only marginal gains relative to the added complexity.
    """
    write_text(path, textwrap.dedent(text))


def write_experimental_summary(
    path: Path,
    cfg: Config,
    device: torch.device,
    data_summary: Dict[str, object],
    main_result: Dict[str, object],
    baseline_rows: List[Dict[str, object]],
    recommendation: Optional[Dict[str, object]],
) -> None:
    lines = [
        "Experimental summary",
        "",
        f"Execution device: {device}",
        f"Random seed: {cfg.seed}",
        f"Dataset size: {data_summary['num_samples']} trajectories x {data_summary['num_steps']} time steps",
        f"Downsample factor: {cfg.downsample}",
        f"Time increment after downsampling: {data_summary['dt']:.6f}",
        "",
        f"Main RNO model: hidden_dim={main_result['hidden_dim']} parameters={main_result['parameter_count']}",
        f"Best validation epoch: {main_result['best_epoch']}",
        (
            "Main RNO test metrics (physical units): "
            f"RMSE={main_result['test_metrics']['rmse_phys']:.6e}, "
            f"MAE={main_result['test_metrics']['mae_phys']:.6e}, "
            f"relative L2={main_result['test_metrics']['rel_l2_phys']:.6e}, "
            f"R^2={main_result['test_metrics']['r2_phys']:.6f}"
        ),
    ]
    if baseline_rows:
        lines.extend(["", "Baseline comparison (test split, physical units):"])
        for row in baseline_rows:
            lines.append(
                f"{row['model']}: RMSE={row['test_rmse_phys']:.6e}, "
                f"relative L2={row['test_rel_l2_phys']:.6e}, R^2={row['test_r2_phys']:.6f}"
            )
    if recommendation is not None:
        lines.extend(
            [
                "",
                "Hidden-state recommendation:",
                f"Recommended minimum hidden dimension = {recommendation['recommended_hidden_dim']}",
                f"Selection rule: {recommendation['criterion']}",
            ]
        )
    write_text(path, "\n".join(lines))


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Train a 1D recurrent neural operator constitutive model.")
    parser.add_argument("--data_path", type=str, default=DEFAULT_CONFIG.data_path)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_CONFIG.output_dir)
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG.epochs)
    parser.add_argument("--sweep_epochs", type=int, default=DEFAULT_CONFIG.sweep_epochs)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_CONFIG.batch_size)
    parser.add_argument("--hidden_dim", type=int, default=DEFAULT_CONFIG.hidden_dim)
    parser.add_argument("--downsample", type=int, default=DEFAULT_CONFIG.downsample)
    parser.add_argument("--device", type=str, default=DEFAULT_CONFIG.device)
    parser.add_argument("--run_hidden_sweep", action="store_true", default=DEFAULT_CONFIG.run_hidden_sweep)
    parser.add_argument("--skip_hidden_sweep", action="store_false", dest="run_hidden_sweep")
    parser.add_argument("--run_baselines", action="store_true", default=DEFAULT_CONFIG.run_baselines)
    parser.add_argument("--skip_baselines", action="store_false", dest="run_baselines")
    args = parser.parse_args()

    cfg = Config()
    cfg.data_path = args.data_path
    cfg.output_dir = args.output_dir
    cfg.epochs = args.epochs
    cfg.sweep_epochs = args.sweep_epochs
    cfg.batch_size = args.batch_size
    cfg.hidden_dim = args.hidden_dim
    cfg.downsample = args.downsample
    cfg.device = args.device
    cfg.run_hidden_sweep = args.run_hidden_sweep
    cfg.run_baselines = args.run_baselines
    return cfg


def main() -> None:
    cfg = parse_args()
    set_global_seed(cfg.seed)
    setup_plotting()
    device = resolve_device(cfg.device)
    output_paths = ensure_dirs(Path(cfg.output_dir))

    print(f"[{timestamp()}] Device: {device}")
    if device.type == "cuda":
        print(f"[{timestamp()}] GPU: {torch.cuda.get_device_name(device)}")

    reader = MatReader(cfg.data_path)
    keys = reader.keys()
    strain_field, stress_field = infer_signal_fields(reader)
    strain = reader.read_field(strain_field)
    stress = reader.read_field(stress_field)
    validate_signals(strain, stress)

    print(f"[{timestamp()}] MATLAB keys: {keys}")
    print(f"[{timestamp()}] Using strain field '{strain_field}' with shape {strain.shape}")
    print(f"[{timestamp()}] Using stress field '{stress_field}' with shape {stress.shape}")

    if cfg.downsample < 1:
        raise ValueError("Downsample factor must be a positive integer.")

    strain = strain[:, :: cfg.downsample]
    stress = stress[:, :: cfg.downsample]
    num_samples, num_steps = strain.shape
    dt = 1.0 / (num_steps - 1)
    time_axis = np.linspace(0.0, 1.0, num_steps, dtype=np.float32)

    splits = make_splits(num_samples, cfg)
    save_split_indices(output_paths["tables"] / "data_split_indices.csv", splits)

    strain_scaler = StandardScaler()
    stress_scaler = StandardScaler()
    rate_scaler = StandardScaler()

    strain_train = strain[splits["train"]].reshape(-1, 1)
    stress_train = stress[splits["train"]].reshape(-1, 1)
    rate_train = np.zeros_like(strain[splits["train"]], dtype=np.float32)
    rate_train[:, 1:] = (strain[splits["train"], 1:] - strain[splits["train"], :-1]) / dt

    strain_scaler.fit(strain_train)
    stress_scaler.fit(stress_train)
    rate_scaler.fit(rate_train.reshape(-1, 1))

    strain_norm = strain_scaler.transform(strain.reshape(-1, 1)).reshape(strain.shape)
    rate = np.zeros_like(strain, dtype=np.float32)
    rate[:, 1:] = (strain[:, 1:] - strain[:, :-1]) / dt
    rate_norm = rate_scaler.transform(rate.reshape(-1, 1)).reshape(rate.shape)
    stress_norm = stress_scaler.transform(stress.reshape(-1, 1)).reshape(stress.shape).astype(np.float32)
    features_norm = np.stack([strain_norm, rate_norm], axis=-1).astype(np.float32)

    save_json(
        output_paths["logs"] / "normalization_stats.json",
        {
            "strain_scaler": strain_scaler.state_dict(),
            "rate_scaler": rate_scaler.state_dict(),
            "stress_scaler": stress_scaler.state_dict(),
            "dt": dt,
            "downsample": cfg.downsample,
        },
    )
    save_json(output_paths["logs"] / "config.json", asdict(cfg))

    data_summary = {
        "num_samples": num_samples,
        "num_steps": num_steps,
        "dt": dt,
        "strain_min": float(strain.min()),
        "strain_max": float(strain.max()),
        "stress_min": float(stress.min()),
        "stress_max": float(stress.max()),
    }
    save_json(output_paths["logs"] / "data_summary.json", data_summary)

    write_problem_framing_note(
        output_paths["report_notes"] / "problem1a_constitutive_io.txt",
        dt=dt,
        time_steps=num_steps,
        downsample=cfg.downsample,
    )

    print(f"[{timestamp()}] Training main RNO model with hidden_dim={cfg.hidden_dim}")
    main_result = run_single_experiment(
        model_name="RNO",
        cfg=cfg,
        device=device,
        features_norm=features_norm,
        stress_norm=stress_norm,
        stress_scaler=stress_scaler,
        strain_phys=strain,
        time_axis=time_axis,
        splits=splits,
        output_paths=output_paths,
        hidden_dim=cfg.hidden_dim,
        epochs=cfg.epochs,
        make_figures=True,
    )

    baseline_summary_rows: List[Dict[str, object]] = []
    if cfg.run_baselines:
        for baseline_name in ["WindowMLP", "GRU"]:
            print(f"[{timestamp()}] Training baseline {baseline_name}")
            result = run_single_experiment(
                model_name=baseline_name,
                cfg=cfg,
                device=device,
                features_norm=features_norm,
                stress_norm=stress_norm,
                stress_scaler=stress_scaler,
                strain_phys=strain,
                time_axis=time_axis,
                splits=splits,
                output_paths=output_paths,
                epochs=cfg.epochs,
                make_figures=False,
            )
            baseline_summary_rows.append(
                {
                    "model": result["experiment_tag"],
                    "parameters": result["parameter_count"],
                    "test_rmse_phys": result["test_metrics"]["rmse_phys"],
                    "test_rel_l2_phys": result["test_metrics"]["rel_l2_phys"],
                    "test_r2_phys": result["test_metrics"]["r2_phys"],
                }
            )

        baseline_summary_rows.insert(
            0,
            {
                "model": main_result["experiment_tag"],
                "parameters": main_result["parameter_count"],
                "test_rmse_phys": main_result["test_metrics"]["rmse_phys"],
                "test_rel_l2_phys": main_result["test_metrics"]["rel_l2_phys"],
                "test_r2_phys": main_result["test_metrics"]["r2_phys"],
            },
        )
        save_csv(output_paths["tables"] / "baseline_comparison.csv", baseline_summary_rows)
        plot_baseline_comparison(baseline_summary_rows, output_paths["figures"] / "baseline_comparison.png", cfg)

    hidden_sweep_rows: List[Dict[str, object]] = []
    recommendation = None
    if cfg.run_hidden_sweep:
        print(f"[{timestamp()}] Running hidden-state sweep: {cfg.hidden_sweep}")
        for hidden_dim in cfg.hidden_sweep:
            result = run_single_experiment(
                model_name="RNO",
                cfg=cfg,
                device=device,
                features_norm=features_norm,
                stress_norm=stress_norm,
                stress_scaler=stress_scaler,
                strain_phys=strain,
                time_axis=time_axis,
                splits=splits,
                output_paths=output_paths,
                hidden_dim=hidden_dim,
                epochs=cfg.sweep_epochs,
                make_figures=False,
            )
            hidden_sweep_rows.append(
                {
                    "hidden_dim": hidden_dim,
                    "parameters": result["parameter_count"],
                    "best_epoch": result["best_epoch"],
                    "best_val_loss": result["best_val_loss"],
                    "val_rmse_phys": result["val_metrics"]["rmse_phys"],
                    "val_rel_l2_phys": result["val_metrics"]["rel_l2_phys"],
                    "test_rmse_phys": result["test_metrics"]["rmse_phys"],
                    "test_rel_l2_phys": result["test_metrics"]["rel_l2_phys"],
                    "test_r2_phys": result["test_metrics"]["r2_phys"],
                }
            )
        save_csv(output_paths["tables"] / "hidden_state_sweep.csv", hidden_sweep_rows)
        plot_hidden_sweep(hidden_sweep_rows, output_paths["figures"] / "hidden_state_sweep.png", cfg)
        recommendation = recommend_hidden_dimension(hidden_sweep_rows)
        save_json(output_paths["tables"] / "hidden_state_recommendation.json", recommendation)
        write_hidden_interpretation_note(
            output_paths["report_notes"] / "problem1c_hidden_state_interpretation.txt",
            recommendation=recommendation,
            sweep_rows=hidden_sweep_rows,
        )

    write_experimental_summary(
        output_paths["report_notes"] / "experimental_summary.txt",
        cfg=cfg,
        device=device,
        data_summary=data_summary,
        main_result=main_result,
        baseline_rows=baseline_summary_rows,
        recommendation=recommendation,
    )

    save_csv(
        output_paths["tables"] / "main_model_summary.csv",
        [
            {
                "model": main_result["experiment_tag"],
                "parameters": main_result["parameter_count"],
                "best_epoch": main_result["best_epoch"],
                "test_rmse_phys": main_result["test_metrics"]["rmse_phys"],
                "test_mae_phys": main_result["test_metrics"]["mae_phys"],
                "test_rel_l2_phys": main_result["test_metrics"]["rel_l2_phys"],
                "test_r2_phys": main_result["test_metrics"]["r2_phys"],
            }
        ],
    )

    print("")
    print("=== Coursework summary ===")
    print(
        f"Best RNO test performance: RMSE={main_result['test_metrics']['rmse_phys']:.6e}, "
        f"MAE={main_result['test_metrics']['mae_phys']:.6e}, "
        f"relative L2={main_result['test_metrics']['rel_l2_phys']:.6e}, "
        f"R^2={main_result['test_metrics']['r2_phys']:.6f}"
    )
    if baseline_summary_rows:
        print("Baseline comparison:")
        for row in baseline_summary_rows:
            print(
                f"  {row['model']}: RMSE={row['test_rmse_phys']:.6e}, "
                f"relative L2={row['test_rel_l2_phys']:.6e}, R^2={row['test_r2_phys']:.6f}"
            )
    if recommendation is not None:
        print(
            f"Recommended minimum hidden dimension: {recommendation['recommended_hidden_dim']} "
            f"({recommendation['criterion']})"
        )
    print(f"Outputs saved to: {output_paths['root'].resolve()}")


if __name__ == "__main__":
    main()
