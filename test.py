# SPDX-License-Identifier: GPL-3.0-or-later
#
# XANESNET-lite
#
# Authors:  Thomas J. Pope, Hendrik Junkawitsch, Tom J. Penfold, C. D. Rankine, B. Li
#
"""Run inference with a trained XANESNET-lite checkpoint."""

import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path

import torch
import lightning.pytorch as pl

from e3ee_data import LightningE3EEDataModule
from e3ee_model import LightningE3EEModelModule

# ============================================================================
def parse_args(args: list[str]) -> Namespace:
    """Parse command-line arguments."""
    parser = ArgumentParser()
    parser.add_argument("--xyzs",type=str,required=True,help="Path to XYZ files")
    parser.add_argument("--spcs",type=str,required=True,help="Path to spectra files")
    parser.add_argument("--checkpoint",type=str,required=True,help="Path to the Lightning checkpoint (.ckpt)")
    parser.add_argument("--predictions",type=str,default="predictions",help="Directory in which prediction files are written")
    return parser.parse_args(args)
# ============================================================================
def main(args: list[str]) -> None:
    """Run inference."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args_namespace = parse_args(args)
    torch.set_float32_matmul_precision("high")
    prediction_dir = Path(args_namespace.predictions)
    prediction_dir.mkdir(parents=True, exist_ok=True)
    xyz_dir = Path(args_namespace.xyzs)
    spectrum_dir = Path(args_namespace.spcs)
    checkpoint = Path(args_namespace.checkpoint)
    datamodule = LightningE3EEDataModule(
        xyz_dir=xyz_dir,
        spectrum_dir=spectrum_dir,
        batch_size=64,
        num_workers=0,
        train_fraction=1.0,
        cutoff=5.0,
        attention_cutoff=5.0,
        seed=2026,
    )
    datamodule.setup("fit")
    model = LightningE3EEModelModule.load_from_checkpoint(checkpoint_path=checkpoint)
    model.eval()
    trainer = pl.Trainer(
        accelerator="cuda",
        devices=4,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
    )
    predictions = []
    targets = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(datamodule.train_dataloader()):
            batch = {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            prediction = model(batch)
            energies = batch["energies"].detach().cpu()
            prediction = prediction.detach().cpu()
            for sample_idx in range(prediction.shape[0]):
                sample_id = batch["sample_id"][sample_idx]
                if isinstance(sample_id, torch.Tensor):
                    sample_id = sample_id.item()
                sample_id = str(sample_id)
                output_file = prediction_dir / f"{sample_id}.txt"
                with output_file.open("w") as handle:
                    for energy, value in zip(energies[0],prediction[sample_idx]):
                        handle.write(f"{energy.item():.10f} {value.item():.10f}\n")
    print("Inference complete")

# ============================================================================
if __name__ == "__main__":
    main(sys.argv[1:])
# ============================================================================
