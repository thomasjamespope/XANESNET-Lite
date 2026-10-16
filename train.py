# SPDX-License-Identifier: GPL-3.0-or-later
#
# XANESNET-lite
#
# Authors:  Thomas J. Pope, Hendrik Junkawitsch, Tom J. Penfold, C. D. Rankine, B. Li
#
"""Entry point and argument parser for training runs."""
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path

import lightning.pytorch as pl

from e3ee_data  import LightningE3EEDataModule
from e3ee_model import LightningE3EEModelModule
# ============================================================================
def parse_args(args: list[str]) -> Namespace:
    """Parse command-line arguments for the training entry point."""
    parser = ArgumentParser()
    parser.add_argument("--xyzs",type=str,required=True,help="Path to XYZ files")
    parser.add_argument("--spcs",type=str,required=True,help="Path to spectra files")
    return parser.parse_args(args)
# ============================================================================
def main(args: list[str]) -> None:
    args_namespace = parse_args(args)
    datamodule = LightningE3EEDataModule(
        xyz_dir=Path(args_namespace.xyzs),
        spectrum_dir=Path(args_namespace.spcs),
        batch_size=32,
        num_workers=0,
        train_fraction=0.8,
        cutoff=5.0,
        attention_cutoff=5.0,
        seed=2026,
    )
    datamodule.setup("fit")
    model_kwargs = {
        "out_size": datamodule.out_dim,
        "max_z": 100,
        "atom_emb_dim": 64,
        "atom_hidden_dim": 128,
        "atom_layers": 3,
        "local_cutoff": 5.0,
        "energy_rbf_dim": 32,
        "energy_min": datamodule.energy_min,
        "energy_max": datamodule.energy_max,
        "attention_rbf_dim": 32,
        "attention_cutoff": 5.0,
        "attention_heads": 4,
        "latent_dim": 128,
        "head_hidden_dim": 128,
        "out_mlp_layers": 3,
    }
    model = LightningE3EEModelModule(model_kwargs=model_kwargs,learning_rate=1e-2)
    trainer = pl.Trainer(
        accelerator="cuda",
        devices=2,
        max_epochs=200,
        enable_progress_bar=True,
        log_every_n_steps=1,
        callbacks=[
            pl.callbacks.ModelCheckpoint(monitor="val_loss",mode="min",save_top_k=3),
            pl.callbacks.EarlyStopping(monitor="val_loss",patience=30,mode="min"),
            pl.callbacks.LearningRateMonitor(logging_interval="epoch"),
        ],
    )
    trainer.fit(model, datamodule=datamodule)
# ============================================================================
if __name__ == "__main__":
    main(sys.argv[1:])
# ============================================================================
