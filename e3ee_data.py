# SPDX-License-Identifier: GPL-3.0-or-later
#
# XANESNET-lite
#
# Authors:  Thomas J. Pope, Hendrik Junkawitsch, Tom J. Penfold, C. D. Rankine, B. Li
#
"""Data processing."""
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split

# ============================================================================
ELEMENTS = { 
     "H": 1,                                                                                                                               "He": 2,
    "Li": 3,"Be": 4,                                                                                 "B": 5, "C": 6, "N": 7, "O": 8, "F": 9,"Ne":10,
    "Na":11,"Mg":12,                                                                                "Al":13,"Si":14, "P":15, "S":16,"Cl":17,"Ar":18,
     "K":19,"Ca":20,"Sc":21,"Ti":22, "V":23,"Cr":24,"Mn":25,"Fe":26,"Co":27,"Ni":28,"Cu":29,"Zn":30,"Ga":31,"Ge":32,"As":33,"Se":34,"Br":35,"Kr":36,
    "Rb":37,"Sr":38, "Y":39,"Zr":40,"Nb":41,"Mo":42,"Tc":43,"Ru":44,"Rh":45,"Pd":46,"Ag":47,"Cd":48,"In":49,"Sn":50,"Sb":51,"Te":52, "I":53,"Xe":54,
    "Cs":55,"Ba":56,"Lu":71,"Hf":72,"Ta":73, "W":74,"Re":75,"Os":76,"Ir":77,"Pt":78,"Au":79,"Hg":80,"Tl":81,"Pb":82,"Bi":83,"Po":84,"At":85,"Rn":86,
                    "La":57,"Ce":58,"Pr":59,"Nd":60,"Pm":61,"Sm":62,"Eu":63,"Gd":64,"Tb":65,"Dy":66,"Ho":67,"Er":68,"Tm":69,"Yb":70,
                    }
# ============================================================================
def read_xyz(path: str | Path):
    """Read an XYZ file, where the first atom is the absorber."""
    path = Path(path)
    with path.open("r") as f:
        lines = [line.strip() for line in f if line.strip()]
    n_atoms = int(lines[0])
    atom_lines = lines[2:2 + n_atoms]
    if len(atom_lines) != n_atoms:
        raise ValueError(f"{path}: expected {n_atoms} atoms, found {len(atom_lines)}")
    symbols = []
    positions = []
    for line in atom_lines:
        fields = line.split()
        if len(fields) < 4:
            raise ValueError(f"Invalid XYZ line in {path}: {line}")
        symbols.append(fields[0])
        positions.append([float(fields[1]),float(fields[2]),float(fields[3])])
    return symbols, torch.tensor(positions, dtype=torch.float32)
# ============================================================================
def read_spectrum(path: str | Path):
    """Read a two-column spectrum file with header lines beginning with '#' ignored."""
    path = Path(path)
    rows = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) < 2:
                continue
            try:
                energy = float(fields[0])
                intensity = float(fields[1])
            except ValueError:
                continue
            rows.append((energy, intensity))
    if not rows:
        raise ValueError(f"No spectrum data found in {path}")
    array = np.asarray(rows, dtype=np.float32)
    return torch.from_numpy(array[:, 0]), torch.from_numpy(array[:, 1])
# ============================================================================
def build_graph(positions: torch.Tensor, cutoff: float = 5.0):
    """Construct a radius graph."""
    distances = torch.cdist(positions, positions)
    src, dst = torch.where((distances <= cutoff) & (distances > 0.0))
    vec = positions[dst] - positions[src]
    edge_weight = distances[src, dst]
    return src.long(),dst.long(),vec,edge_weight
# ============================================================================
def build_attention_graph(positions: torch.Tensor,absorber_index: int = 0,cutoff: float = 5.0):
    """Build an absorber-centred attention graph."""
    absorber = positions[absorber_index]
    vectors = positions - absorber
    distances = torch.linalg.norm(vectors, dim=-1)
    keep = (distances <= cutoff) & (distances > 0.0)
    att_dst = torch.where(keep)[0].long()
    att_dist = distances[keep]
    return att_dst, att_dist, vectors
# ============================================================================
class E3EEDataset(Dataset):
    """Invariant E3EE dataset."""
    def __init__(self,xyz_dir: str | Path,spectrum_dir: str | Path,cutoff: float = 5.0,attention_cutoff: float = 5.0):
        self.xyz_dir = Path(xyz_dir)
        self.spectrum_dir = Path(spectrum_dir)
        self.cutoff = cutoff
        self.attention_cutoff = attention_cutoff
        self.xyz_files = sorted(self.xyz_dir.glob("*.xyz"))
        if not self.xyz_files:
            raise RuntimeError(f"No .xyz files found in {self.xyz_dir}")
        self.samples = []
        for xyz_file in self.xyz_files:
            spectrum_file = self.spectrum_dir / f"{xyz_file.stem}.txt"
            if not spectrum_file.exists():
                raise FileNotFoundError(f"No spectrum found for {xyz_file.name}: expected {spectrum_file}")
            self.samples.append((xyz_file, spectrum_file))
                # Infer the common energy grid from the first spectrum.
        energies, _ = read_spectrum(self.samples[0][1])
        self.energy_grid = energies
        self.out_dim = energies.numel()
        self.energy_min = energies.min().item()
        self.energy_max = energies.max().item()
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, index):
        xyz_file, spectrum_file = self.samples[index]
        symbols, positions = read_xyz(xyz_file)
        energies, intensities = read_spectrum(spectrum_file)
        atomic_numbers = torch.tensor([ELEMENTS[symbol] for symbol in symbols], dtype=torch.long)
        edge_src,edge_dst,edge_vec,edge_weight, = build_graph(positions, cutoff=self.cutoff)
        att_dst,att_dist,att_vec = build_attention_graph(positions,absorber_index=0,cutoff=self.attention_cutoff)
        return {
            "x": atomic_numbers,
            "positions": positions,
            "target_site_index": torch.tensor(0, dtype=torch.long),
            "edge_src": edge_src,
            "edge_dst": edge_dst,
            "edge_vec": edge_vec,
            "edge_weight": edge_weight,
            "att_dst": att_dst,
            "att_vec": att_vec,
            "att_dist": att_dist,
            "energies": energies,
            "intensities": intensities,
            "sample_id": xyz_file.stem,
        }
# ============================================================================
def collate_e3ee(batch):
    """Collate variable-size molecular structures."""
    batch_size = len(batch)
    n_nodes = [sample["x"].shape[0] for sample in batch]
    max_nodes = max(n_nodes)
  # Node features
    x = torch.zeros(batch_size,max_nodes,dtype=torch.long)
    mask = torch.zeros(batch_size,max_nodes,dtype=torch.bool)
    target_site_index = torch.zeros(batch_size,dtype=torch.long)
    for b, sample in enumerate(batch):
        n = sample["x"].shape[0]
        x[b, :n] = sample["x"]
        mask[b, :n] = True
        target_site_index[b] = sample["target_site_index"]
  # Spectrum
    energies = torch.stack([sample["energies"] for sample in batch])
    intensities = torch.stack([sample["intensities"] for sample in batch])
  # Main graph
    edge_src, edge_dst, edge_vec, edge_weight = [], [], [], []
    offset = 0
    for sample, n in zip(batch, n_nodes):
        edge_src.append(sample["edge_src"] + offset)
        edge_dst.append(sample["edge_dst"] + offset)
        edge_vec.append(sample["edge_vec"])
        edge_weight.append(sample["edge_weight"])
        offset += n
    edge_src = torch.cat(edge_src)
    edge_dst = torch.cat(edge_dst)
    edge_vec = torch.cat(edge_vec)
    edge_weight = torch.cat(edge_weight)
  # Attention graph
    att_dst, att_vec, att_dist = [], [], []
    offset = 0
    for sample, n in zip(batch, n_nodes):
        att_dst.append(sample["att_dst"] + offset)
        att_vec.append(sample["att_vec"])
        att_dist.append(sample["att_dist"])
        offset += n
    att_dst = torch.cat(att_dst)
    att_dist = torch.cat(att_dist)
    return {
        "x": x,
        "mask": mask,
        "target_site_index": target_site_index,
        "edge_src": edge_src,
        "edge_dst": edge_dst,
        "edge_vec": edge_vec,
        "edge_weight": edge_weight,
        "att_dst": att_dst,
        "att_vec": att_vec,        
        "att_dist": att_dist,
        "energies": energies,
        "intensities": intensities,
        "sample_id": [sample["sample_id"] for sample in batch],
    }
# ============================================================================
class LightningE3EEDataModule(pl.LightningDataModule):
    """Lightning-e3ee DataModule."""
    def __init__(
        self,
        xyz_dir: str | Path,
        spectrum_dir: str | Path,
        batch_size: int = 16,
        num_workers: int = 4,
        train_fraction: float = 0.8,
        val_fraction: float = 0.1,
        cutoff: float = 5.0,
        attention_cutoff: float = 5.0,
        seed: int = 42,
    ):
        super().__init__()
        self.xyz_dir = xyz_dir
        self.spectrum_dir = spectrum_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_fraction = train_fraction
        self.cutoff = cutoff
        self.attention_cutoff = attention_cutoff
        self.seed = seed
        self.dataset = None
        self.train_dataset = None
        self.val_dataset = None
        self.out_dim = None
        self.energy_min = None
        self.energy_max = None
        self.energy_grid = None
# ============================================================================
    def setup(self, stage=None):
        if self.dataset is not None:
            return        
        self.dataset = E3EEDataset(
            xyz_dir=self.xyz_dir,
            spectrum_dir=self.spectrum_dir,
            cutoff=self.cutoff,
            attention_cutoff=self.attention_cutoff,
        )
        n = len(self.dataset)
        n_train = int(self.train_fraction * n)
        n_val = n - n_train
        generator = torch.Generator().manual_seed(self.seed)
        self.train_dataset, self.val_dataset = random_split(self.dataset, [n_train, n_val], generator=generator)
        self.out_dim = self.dataset.out_dim
        self.energy_min = self.dataset.energy_min
        self.energy_max = self.dataset.energy_max
        self.energy_grid = self.dataset.energy_grid
# ============================================================================
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            collate_fn=collate_e3ee,
        )
# ============================================================================
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            collate_fn=collate_e3ee,
        )
# ============================================================================
