# SPDX-License-Identifier: GPL-3.0-or-later
#
# XANESNET-lite
#
# Authors:  Thomas J. Pope, Hendrik Junkawitsch, Tom J. Penfold, C. D. Rankine, B. Li
"""Standalone E3EE model derived from the original XANESNET architecture."""
from __future__ import annotations

from typing import Iterable

import lightning.pytorch as pl
import torch
import torch.nn as nn
from e3nn import o3
from e3nn.nn import Gate
from e3nn.o3 import FullyConnectedTensorProduct
# ============================================================================
class GaussianRBF(nn.Module):
    """Gaussian radial basis expansion."""
    def __init__(self,start: float,stop: float,n_rbf: int,gamma: float | None = None) -> None:
        super().__init__()
        if n_rbf < 1:
            raise ValueError("n_rbf must be >= 1")
        centers = torch.linspace(start, stop, n_rbf)
        self.register_buffer("centers", centers)
        if gamma is None:
            delta = (stop - start) / max(n_rbf - 1, 1)
            gamma = 1.0 / (delta * delta + 1e-12)
        self.gamma = float(gamma)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(-self.gamma * (x.unsqueeze(-1) - self.centers) ** 2)
# ============================================================================
class CosineCutoff(nn.Module):
    """Smooth cosine cutoff envelope."""
    def __init__(self, cutoff: float) -> None:
        super().__init__()
        if cutoff <= 0:
            raise ValueError("cutoff must be > 0")
        self.cutoff = float(cutoff)
    def forward(self, r: torch.Tensor) -> torch.Tensor:
        x = r / self.cutoff
        out = 0.5 * (torch.cos(torch.pi * x) + 1.0)
        return out * (r <= self.cutoff).to(r.dtype)
# ============================================================================
class EnergyRBFEmbedding(nn.Module):
    """Gaussian RBF embedding of the energy grid."""
    def __init__(self, e_min: float, e_max: float, n_rbf: int) -> None:
        super().__init__()
        if e_max <= e_min:
            raise ValueError("e_max must be greater than e_min")
        self.rbf = GaussianRBF(e_min, e_max, n_rbf)
    def forward(self, energies: torch.Tensor) -> torch.Tensor:
        return self.rbf(energies)
# ============================================================================
class MLP(nn.Module):
    """Simple SiLU MLP."""
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        n_layers: int = 2,
        dropout: float = 0.0,
        layer_norm: bool = False,
    ) -> None:
        super().__init__()
        if n_layers < 1:
            raise ValueError("n_layers must be >= 1")
        layers: list[nn.Module] = []
        dim = in_dim
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(dim, hidden_dim))
            if layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.SiLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            dim = hidden_dim
        layers.append(nn.Linear(dim, out_dim))
        self.net = nn.Sequential(*layers)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
# ============================================================================
class IrrepNorm(nn.Module):
    """Irrep-aware normalization."""
    def __init__(self,irreps: o3.Irreps,eps: float = 1e-8,affine: bool = True) -> None:
        super().__init__()
        self.irreps = o3.Irreps(irreps)
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(self.irreps.dim))
            self.bias = nn.Parameter(torch.zeros(self.irreps.dim))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape[:-1]
        dim = x.shape[-1]
        x_flat = x.reshape(-1, dim)
        outputs: list[torch.Tensor] = []
        offset = 0
        batch_size = x_flat.shape[0]
        for mul, irrep in self.irreps:
            block_dim = mul * irrep.dim
            block = x_flat[:, offset : offset + block_dim]
            block = block.reshape(batch_size, mul, irrep.dim)
            if irrep.l == 0:
                mean = block.mean(dim=1, keepdim=True)
                var = ((block - mean) ** 2).mean(dim=1, keepdim=True)
                block = (block - mean) / torch.sqrt(var + self.eps)
            else:
                norm = torch.sqrt((block**2).mean(dim=2, keepdim=True) + self.eps)
                block = block / norm
            outputs.append(block.reshape(batch_size, block_dim))
            offset += block_dim
        out = torch.cat(outputs, dim=-1)
        if self.affine:
            out = out * self.weight + self.bias
        return out.view(*original_shape, dim)
# ============================================================================
class RadialMLP(nn.Module):
    """Radial network producing tensor-product weights."""
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
# ============================================================================
class EquivariantInteractionBlock(nn.Module):
    """e3nn interaction block."""
    def __init__(
        self,
        irreps_node: str,
        irreps_sh: str,
        irreps_message: str,
        rbf_dim: int,
        radial_hidden_dim: int,
        cutoff: float,
        residual_scale_init: float = 0.1,
    ) -> None:
        super().__init__()
        self.irreps_node = o3.Irreps(irreps_node)
        self.irreps_sh = o3.Irreps(irreps_sh)
        self.irreps_message = o3.Irreps(irreps_message)
        self.pre_norm = IrrepNorm(self.irreps_node)
        self.cutoff_fn = CosineCutoff(cutoff)
        self.tp = FullyConnectedTensorProduct(self.irreps_node,self.irreps_sh,self.irreps_message,shared_weights=False)
        self.weight_mlp = RadialMLP(rbf_dim,radial_hidden_dim,self.tp.weight_numel)
        self.edge_gate = nn.Sequential(
            nn.Linear(rbf_dim, radial_hidden_dim),
            nn.SiLU(),
            nn.Linear(radial_hidden_dim, 1),
            nn.Sigmoid(),
        )
        irreps_scalars = o3.Irreps([(mul, irrep) for mul, irrep in self.irreps_message if irrep.l == 0])
        irreps_gated = o3.Irreps([(mul, irrep) for mul, irrep in self.irreps_message if irrep.l > 0])
        irreps_gates = o3.Irreps(f"{irreps_gated.num_irreps}x0e") if irreps_gated.num_irreps > 0 else o3.Irreps("")
        self.msg_linear = o3.Linear(self.irreps_message,irreps_scalars + irreps_gates + irreps_gated)
        self.gate = Gate(
            irreps_scalars=irreps_scalars,
            act_scalars=[torch.nn.functional.silu] * len(irreps_scalars),
            irreps_gates=irreps_gates,
            act_gates=[torch.sigmoid] * len(irreps_gates),
            irreps_gated=irreps_gated,
        )

        self.update_linear = o3.Linear(self.gate.irreps_out, self.irreps_node)
        self.self_linear = o3.Linear(self.irreps_node, self.irreps_node)
        self.res_scale = nn.Parameter(torch.tensor(float(residual_scale_init), dtype=torch.float32))
    def forward(
        self,
        x: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_sh: torch.Tensor,
        edge_rbf: torch.Tensor,
        edge_len: torch.Tensor,
    ) -> torch.Tensor:
        if edge_src.numel() == 0:
            return x
        x_norm = self.pre_norm(x)
        tp_weights = self.weight_mlp(edge_rbf)
        message = self.tp(x_norm[edge_src], edge_sh, tp_weights)
        cutoff_weight = self.cutoff_fn(edge_len).unsqueeze(-1)
        gate_weight = self.edge_gate(edge_rbf)
        edge_weight = cutoff_weight * gate_weight
        message = message * edge_weight
        aggregate = torch.zeros(x.shape[0],self.irreps_message.dim,device=x.device,dtype=x.dtype)
        aggregate.index_add_(0, edge_dst, message)
        norm = torch.zeros(x.shape[0],1,device=x.device,dtype=x.dtype)
        norm.index_add_(0, edge_dst, edge_weight)
        aggregate = self.gate(self.msg_linear(aggregate / norm.clamp_min(1e-8)))
        update = self.self_linear(x_norm) + self.update_linear(aggregate)
        return x + self.res_scale.to(x.dtype) * update
# ============================================================================
class EquivariantAtomEncoder(nn.Module):
    """Original XANESNET-style equivariant atom encoder."""
    def __init__(
        self,
        max_z: int,
        cutoff: float,
        num_interactions: int,
        rbf_dim: int,
        lmax: int,
        node_attr_dim: int,
        hidden_dim: int,
        irreps_node: str,
        irreps_message: str,
        residual_scale_init: float,
    ) -> None:
        super().__init__()
        self.cutoff = float(cutoff)
        self.rbf_dim = rbf_dim
        self.irreps_node = o3.Irreps(irreps_node)
        self.irreps_message = o3.Irreps(irreps_message)
        self.dist_rbf = GaussianRBF(0.0, cutoff, rbf_dim)
        self.z_emb = nn.Embedding(max_z + 1, node_attr_dim)
        self.input_scalar_dim = node_attr_dim + 1
        self.input_lin = o3.Linear(o3.Irreps(f"{self.input_scalar_dim}x0e"),self.irreps_node)
        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax)
        self.blocks = nn.ModuleList(
            [
                EquivariantInteractionBlock(
                    irreps_node=str(self.irreps_node),
                    irreps_sh=str(self.irreps_sh),
                    irreps_message=str(self.irreps_message),
                    rbf_dim=rbf_dim,
                    radial_hidden_dim=hidden_dim,
                    cutoff=cutoff,
                    residual_scale_init=residual_scale_init,
                )
                for _ in range(num_interactions)
            ]
        )
        self.out_norm = IrrepNorm(self.irreps_node)
    def forward(
        self,
        z: torch.Tensor,
        mask: torch.Tensor,
        target_site_index: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_weight: torch.Tensor,
        edge_vec: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, n_atoms = z.shape
        device = z.device
        target_site_flag = torch.zeros(batch_size,n_atoms,dtype=torch.float32,device=device)
        batch_index = torch.arange(batch_size, device=device)
        target_site_flag[batch_index, target_site_index] = 1.0
        z_embedding = self.z_emb(z)
        scalar_input = torch.cat([z_embedding, target_site_flag.unsqueeze(-1)],dim=-1)
      # e3nn layers operate on flattened B*N node tensors.
        x = self.input_lin(scalar_input.reshape(batch_size * n_atoms, self.input_scalar_dim))
        flat_mask = mask.reshape(batch_size * n_atoms)
        x = x * flat_mask.unsqueeze(-1).to(x.dtype)
        if edge_src.numel() > 0:
            edge_len = edge_weight.to(dtype=x.dtype)
            edge_direction = edge_vec.to(dtype=x.dtype) / edge_len.unsqueeze(-1).clamp_min(1e-8)
            edge_rbf = self.dist_rbf(edge_len.clamp(max=self.cutoff))
            edge_sh = o3.spherical_harmonics(self.irreps_sh,edge_direction,normalize=True,normalization="component")
        else:
            edge_len = torch.zeros(0, device=device, dtype=x.dtype)
            edge_rbf = torch.zeros(0,self.rbf_dim,device=device,dtype=x.dtype)
            edge_sh = torch.zeros(0,self.irreps_sh.dim,device=device,dtype=x.dtype)
        for block in self.blocks:
            x = block(
                x=x,
                edge_src=edge_src,
                edge_dst=edge_dst,
                edge_sh=edge_sh,
                edge_rbf=edge_rbf,
                edge_len=edge_len,
            )
        x = self.out_norm(x)
        x = x * flat_mask.unsqueeze(-1).to(x.dtype)
        return x.reshape(batch_size, n_atoms, self.irreps_node.dim)
# ============================================================================
def invariant_feature_dim(irreps: o3.Irreps) -> int:
    """Return the number of scalar (l=0) channels in an irreps tensor."""
    return sum(mul for mul, irrep in irreps if irrep.l == 0)
# ============================================================================
def invariant_features_from_irreps(x: torch.Tensor,irreps: o3.Irreps) -> torch.Tensor:
    """Extract scalar channels from an e3nn representation."""
    parts: list[torch.Tensor] = []
    offset = 0
    for mul, irrep in irreps:
        block_dim = mul * irrep.dim
        block = x[..., offset : offset + block_dim]
        if irrep.l == 0:
            parts.append(block)
        offset += block_dim
    if not parts:
        raise ValueError("The encoder irreps contain no scalar l=0 channels")
    return torch.cat(parts, dim=-1)
# ============================================================================
class EnergyConditionedTargetSiteBranch(nn.Module):
    """Target-site invariant branch."""
    def __init__(self,atom_dim: int,e_dim: int,hidden_dim: int,out_dim: int) -> None:
        super().__init__()
        self.mlp = MLP(in_dim=atom_dim + e_dim,hidden_dim=hidden_dim,out_dim=out_dim,n_layers=3)
    def forward(self,target_features: torch.Tensor,e_feat: torch.Tensor) -> torch.Tensor:
        batch_size, atom_dim = target_features.shape
        n_energies, energy_dim = e_feat.shape
        target = target_features.unsqueeze(1).expand(batch_size,n_energies,atom_dim)
        energy = e_feat.unsqueeze(0).expand(batch_size,n_energies,energy_dim)
        return self.mlp(torch.cat([target, energy], dim=-1))
# ============================================================================
class EnergyConditionedAtomAttention(nn.Module):
    """Original XANESNET energy-conditioned invariant atom attention."""
    def __init__(
        self,
        atom_dim: int,
        e_dim: int,
        hidden_dim: int,
        latent_dim: int,
        att_cutoff: float,
        rbf_dim: int = 16,
        max_z: int = 100,
        z_emb_dim: int = 32,
        n_heads: int = 4,
    ) -> None:
        super().__init__()
        if latent_dim % n_heads != 0:
            raise ValueError("latent_dim must be divisible by n_heads")
        self.atom_dim = atom_dim
        self.e_dim = e_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.rbf_dim = rbf_dim
        self.att_cutoff = float(att_cutoff)
        self.n_heads = n_heads
        self.head_dim = latent_dim // n_heads
        self.z_emb = nn.Embedding(max_z + 1, z_emb_dim)
        self.dist_rbf = GaussianRBF(0.0, self.att_cutoff, rbf_dim)
        self.value_envelope = CosineCutoff(self.att_cutoff)
        self.query_mlp = MLP(in_dim=atom_dim + e_dim,hidden_dim=hidden_dim,out_dim=latent_dim,n_layers=3)
        atom_static_dim = atom_dim + z_emb_dim + 1 + rbf_dim
        self.key_mlp =   MLP(in_dim=atom_static_dim,hidden_dim=hidden_dim,out_dim=latent_dim,n_layers=3)
        self.value_mlp = MLP(in_dim=atom_static_dim,hidden_dim=hidden_dim,out_dim=latent_dim,n_layers=3)
        self.out_proj =  MLP(in_dim=latent_dim,     hidden_dim=hidden_dim,out_dim=latent_dim,n_layers=2)
        self.score_scale = self.head_dim**-0.5
    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(*x.shape[:-1], self.n_heads, self.head_dim)
    def forward(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor,
        e_feat: torch.Tensor,
        target_site_index: torch.Tensor,
        att_dst: torch.Tensor,
        att_dist: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, n_atoms, atom_dim = h.shape
        n_energies, energy_dim = e_feat.shape
        device = h.device
        flat_size = batch_size * n_atoms
        batch_index = torch.arange(batch_size, device=device)
        h_target = h[batch_index, target_site_index, :]
        query_input = torch.cat(
            [
                h_target.unsqueeze(1).expand(batch_size, n_energies, atom_dim),
                e_feat.unsqueeze(0).expand(batch_size, n_energies, energy_dim),
            ],
            dim=-1,
        )
        query = self.query_mlp(query_input)
        att_mask_flat = torch.zeros(flat_size,dtype=torch.bool,device=device)
        att_mask_flat[att_dst] = True
        att_dist_flat = torch.zeros(flat_size,dtype=h.dtype,device=device)
        att_dist_flat[att_dst] = att_dist.to(dtype=h.dtype)
        att_mask = att_mask_flat.view(batch_size, n_atoms) & mask
        att_dist_bn = att_dist_flat.view(batch_size, n_atoms)
        distance_rbf = self.dist_rbf(att_dist_bn)
        z_embedding = self.z_emb(z)
        is_target = torch.zeros(batch_size,n_atoms,dtype=h.dtype,device=device)
        is_target[batch_index, target_site_index] = 1.0
        atom_static = torch.cat([h,z_embedding,is_target.unsqueeze(-1),distance_rbf],dim=-1)
        key = self.key_mlp(atom_static)
        value = self.value_mlp(atom_static)
        envelope = self.value_envelope(att_dist_bn)
        value = value * envelope.unsqueeze(-1)
        query = self._split_heads(query)
        key = self._split_heads(key)
        value = self._split_heads(value)
        scores = (query.unsqueeze(2) * key.unsqueeze(1)).sum(dim=-1) * self.score_scale
        attention_mask = att_mask.unsqueeze(1).unsqueeze(-1)
        scores = scores.masked_fill(~attention_mask, -1e9)
        attention = torch.softmax(scores, dim=2)
        attention = attention * attention_mask.to(attention.dtype)
        attention = attention / attention.sum(dim=2, keepdim=True).clamp_min(1e-8)
        output = (attention.unsqueeze(-1) * value.unsqueeze(1)).sum(dim=2)
        output = output.reshape(batch_size, n_energies, self.latent_dim)
        return self.out_proj(output)
# ============================================================================
class E3EE(nn.Module):
    """Standalone XANESNET-style E3EE model."""
    def __init__(
        self,
        *,
        max_z: int,
        atom_emb_dim: int,
        atom_hidden_dim: int,
        atom_layers: int,
        local_cutoff: float,
        rbf_dim: int,
        energy_rbf_dim: int,
        latent_dim: int,
        head_hidden_dim: int,
        e3nn_irreps: str,
        e3nn_irreps_message: str,
        e3nn_lmax: int,
        out_mlp_layers: int,
        attention_heads: int,
        attention_rbf_dim: int,
        att_cutoff: float,
        energy_min: float,
        energy_max: float,
        residual_scale_init: float = 0.1,
        out_size: int | None = None,
    ) -> None:
        super().__init__()
        self.max_z = max_z
        self.atom_emb_dim = atom_emb_dim
        self.atom_hidden_dim = atom_hidden_dim
        self.atom_layers = atom_layers
        self.local_cutoff = local_cutoff
        self.rbf_dim = rbf_dim
        self.energy_rbf_dim = energy_rbf_dim
        self.latent_dim = latent_dim
        self.head_hidden_dim = head_hidden_dim
        self.e3nn_irreps = e3nn_irreps
        self.e3nn_irreps_message = e3nn_irreps_message
        self.e3nn_lmax = e3nn_lmax
        self.out_mlp_layers = out_mlp_layers
        self.attention_heads = attention_heads
        self.attention_rbf_dim = attention_rbf_dim
        self.att_cutoff = att_cutoff
        self.energy_min = float(energy_min)
        self.energy_max = float(energy_max)
        self.out_size = out_size
        self.residual_scale_init = residual_scale_init
        self.atom_encoder = EquivariantAtomEncoder(
            max_z=max_z,
            cutoff=local_cutoff,
            num_interactions=atom_layers,
            rbf_dim=rbf_dim,
            lmax=e3nn_lmax,
            node_attr_dim=atom_emb_dim,
            hidden_dim=atom_hidden_dim,
            irreps_node=e3nn_irreps,
            irreps_message=e3nn_irreps_message,
            residual_scale_init=residual_scale_init,
        )
        self.inv_dim = invariant_feature_dim(self.atom_encoder.irreps_node)
        self.energy_embedding = EnergyRBFEmbedding(
            e_min=self.energy_min,
            e_max=self.energy_max,
            n_rbf=energy_rbf_dim,
        )
        self.target_site_branch = EnergyConditionedTargetSiteBranch(
            atom_dim=self.inv_dim,
            e_dim=energy_rbf_dim,
            hidden_dim=head_hidden_dim,
            out_dim=latent_dim,
        )
        self.atom_attention = EnergyConditionedAtomAttention(
            atom_dim=self.inv_dim,
            e_dim=energy_rbf_dim,
            hidden_dim=atom_hidden_dim,
            latent_dim=latent_dim,
            att_cutoff=att_cutoff,
            rbf_dim=attention_rbf_dim,
            max_z=max_z,
            z_emb_dim=32,
            n_heads=attention_heads,
        )
        self.head = MLP(
            in_dim=2 * latent_dim,
            hidden_dim=head_hidden_dim,
            out_dim=1,
            n_layers=out_mlp_layers,
        )
    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        target_site_index: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_weight: torch.Tensor,
        edge_vec: torch.Tensor,
        att_dst: torch.Tensor,
        att_dist: torch.Tensor,
        energies: torch.Tensor,
    ) -> torch.Tensor:
        if energies.ndim == 1:
            energy_grid = energies
        elif energies.ndim == 2:
            if energies.shape[0] != x.shape[0]:
                raise ValueError("Batched energy grid has incompatible batch dimension: energies={tuple(energies.shape)}, x={tuple(x.shape)}" )
            energy_grid = energies[0]
            if not torch.allclose(
                energies,
                energy_grid.unsqueeze(0).expand_as(energies),
                rtol=1e-6,
                atol=1e-7,
            ):
                raise ValueError("Each spectrum must use the same energy grid within a batch.")
        else:
            raise ValueError(f"energies must have shape [nE] or [B, nE], got {tuple(energies.shape)}")    
        energies = energy_grid
        batch_size = x.shape[0]
        device = x.device
        e_feat = self.energy_embedding(energies.to(dtype=torch.float32))
        e_feat = e_feat.to(device=device)
        h_full = self.atom_encoder(
            z=x,
            mask=mask,
            target_site_index=target_site_index,
            edge_src=edge_src,
            edge_dst=edge_dst,
            edge_weight=edge_weight,
            edge_vec=edge_vec,
        )
        h = invariant_features_from_irreps(h_full,self.atom_encoder.irreps_node)
        batch_index = torch.arange(batch_size, device=device)
        h_target = h[batch_index, target_site_index, :]
        target_latent = self.target_site_branch(h_target,e_feat)
        attention_latent = self.atom_attention(
            h=h,
            z=x,
            mask=mask,
            e_feat=e_feat,
            target_site_index=target_site_index,
            att_dst=att_dst,
            att_dist=att_dist,
        )
        combined = torch.cat([target_latent, attention_latent],dim=-1)
        return self.head(combined).squeeze(-1)
# ============================================================================
__all__ = [
    "CosineCutoff",
    "E3EE",
    "EnergyConditionedAtomAttention",
    "EnergyConditionedTargetSiteBranch",
    "EnergyRBFEmbedding",
    "EquivariantAtomEncoder",
    "EquivariantInteractionBlock",
    "GaussianRBF",
    "IrrepNorm",
    "MLP",
    "RadialMLP",
    "invariant_feature_dim",
    "invariant_features_from_irreps",
]
# ============================================================================
class LightningE3EEModelModule(pl.LightningModule):
    """Lightning wrapper for the E3EE-style model."""
    def __init__(self,model_kwargs: dict,learning_rate: float = 1e-4):
        super().__init__()
        self.save_hyperparameters()  
        self.model = E3EE(
            max_z = model_kwargs["max_z"],
            atom_emb_dim = model_kwargs["atom_emb_dim"],
            atom_hidden_dim = model_kwargs["atom_hidden_dim"],
            atom_layers = model_kwargs["atom_layers"],
            local_cutoff = model_kwargs["local_cutoff"],
            rbf_dim = model_kwargs["energy_rbf_dim"],
            energy_rbf_dim = model_kwargs["energy_rbf_dim"],
            latent_dim = model_kwargs["latent_dim"],
            head_hidden_dim = model_kwargs["head_hidden_dim"],
            e3nn_irreps = "32x0e + 16x1o + 8x2e",
            e3nn_irreps_message = "16x0e + 8x1o + 4x2e",
            e3nn_lmax = 2,
            out_mlp_layers = model_kwargs["out_mlp_layers"],
            attention_heads = model_kwargs["attention_heads"],
            attention_rbf_dim = model_kwargs["attention_rbf_dim"],
            att_cutoff = model_kwargs["attention_cutoff"],
            energy_min = model_kwargs["energy_min"],
            energy_max = model_kwargs["energy_max"],
            residual_scale_init = 0.1 ,
            out_size = model_kwargs["out_size"],
        )
        self.learning_rate = learning_rate
    def forward(self, batch):
        return self.model(
            x=batch["x"],
            mask=batch["mask"],
            target_site_index=batch["target_site_index"],
            edge_src=batch["edge_src"],
            edge_dst=batch["edge_dst"],
            edge_vec=batch["edge_vec"],
            edge_weight=batch["edge_weight"],
            att_dst=batch["att_dst"],
            att_dist=batch["att_dist"],
            energies=batch["energies"],
        )
# ============================================================================
    def training_step(self, batch, batch_idx):
        prediction = self(batch)
        batch_size = batch["x"].shape[0]
        loss = torch.nn.functional.mse_loss(prediction,batch["intensities"])
        self.log("train_loss",loss,prog_bar=True,on_step=True,on_epoch=True,sync_dist=True,batch_size=batch_size)
        return loss
# ============================================================================
    def validation_step(self, batch, batch_idx):
        prediction = self(batch)
        batch_size = batch["x"].shape[0]
        loss = torch.nn.functional.mse_loss(prediction,batch["intensities"])
        self.log("val_loss",loss,prog_bar=True,on_step=True,on_epoch=True,sync_dist=True,batch_size=batch_size)
        return loss
# ============================================================================
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
        return [optimizer], [scheduler]
# ============================================================================
