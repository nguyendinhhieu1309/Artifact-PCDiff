"""Fracture-aware components proposed by Artifact-PCDiff."""

import math

import torch
import torch.nn as nn


class ReliabilityGatedFractureRefinement(nn.Module):
    """Refine repair noise using dynamically selected fracture-side features.

    Tensor layouts follow the surrounding PVCNN code: coordinates and point
    features are channel-first, with shapes [B, 3, P] and [B, C, P].
    """

    def __init__(self, feature_dim, time_dim, attention_dim=100, fracture_points=10):
        super().__init__()
        self.attention_dim = attention_dim
        self.fracture_points = fracture_points

        self.query = nn.Linear(feature_dim, attention_dim)
        self.key = nn.Linear(feature_dim, attention_dim)
        self.value = nn.Linear(feature_dim, attention_dim)
        self.reliability = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
        self.refinement = nn.Sequential(
            nn.Linear(feature_dim + attention_dim + time_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 3),
        )

    def forward(self, fragment_xyz, noisy_repair, base_noise,
                fragment_features, repair_features, timestep_features,
                alpha_bar, timesteps, total_timesteps):
        """Return base epsilon plus the gated fracture-aware residual.

        Args use channel-first layouts except ``alpha_bar`` [B, 1, 1] and
        ``timesteps`` [B]. Timesteps are zero-indexed in the legacy DDPM code;
        adding one reproduces the paper's t in {1, ..., T} gate definition.
        """
        batch_size, _, repair_count = noisy_repair.shape
        fragment_count = fragment_xyz.shape[-1]
        if fragment_count == 0 or repair_count == 0:
            return base_noise

        alpha_bar = alpha_bar.to(device=noisy_repair.device, dtype=noisy_repair.dtype)
        alpha_bar = alpha_bar.clamp_min(torch.finfo(noisy_repair.dtype).eps)
        clean_estimate = (noisy_repair - torch.sqrt(1.0 - alpha_bar) * base_noise) / torch.sqrt(alpha_bar)

        # Top-K is used only to choose the fracture neighborhood. Detaching this
        # search avoids retaining a large B x N x M distance graph per step.
        with torch.no_grad():
            fragment_points = fragment_xyz.detach().transpose(1, 2)
            repair_points = clean_estimate.detach().transpose(1, 2)
            # Chunk over observed points so large inference batches do not
            # allocate a full B x N_fragment x M_repair distance tensor.
            nearest_chunks = []
            for start in range(0, fragment_count, 2048):
                fragment_chunk = fragment_points[:, start:start + 2048]
                nearest_chunks.append(torch.cdist(fragment_chunk, repair_points).min(dim=-1).values)
            nearest_repair_distance = torch.cat(nearest_chunks, dim=1)
            k = min(self.fracture_points, fragment_count)
            boundary_indices = nearest_repair_distance.topk(k, dim=1, largest=False).indices

        fragment_point_features = fragment_features.transpose(1, 2)
        selected_fragment_features = fragment_point_features.gather(
            1, boundary_indices.unsqueeze(-1).expand(-1, -1, fragment_point_features.shape[-1])
        )
        repair_point_features = repair_features.transpose(1, 2)
        timestep_point_features = timestep_features.transpose(1, 2)

        queries = self.query(repair_point_features)
        keys = self.key(selected_fragment_features)
        values = self.value(selected_fragment_features)
        reliability = self.reliability(selected_fragment_features)

        attention = torch.softmax(
            torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(self.attention_dim),
            dim=-1,
        )
        attended = torch.bmm(attention, reliability * values)
        residual = self.refinement(
            torch.cat((repair_point_features, attended, timestep_point_features), dim=-1)
        ).transpose(1, 2)

        # The diffusion implementation uses t=0,...,T-1. The paper uses 1,...,T.
        gate = 1.0 - (timesteps.to(noisy_repair.dtype) + 1.0) / float(total_timesteps)
        gate = gate.clamp_(0.0, 1.0).view(batch_size, 1, 1)
        return base_noise + gate * residual


def interface_chamfer_loss(predicted_repair, target_repair, fragment_xyz, point_count=64):
    """Symmetric squared Chamfer distance between repair-side interface bands."""
    predicted_points = predicted_repair.transpose(1, 2)
    target_points = target_repair.transpose(1, 2)
    fragment_points = fragment_xyz.transpose(1, 2)
    k = min(point_count, predicted_points.shape[1], target_points.shape[1])
    if k == 0:
        return predicted_repair.new_zeros((predicted_repair.shape[0],))

    # The ground-truth band is fixed; only the predicted band coordinates need
    # gradients. Top-K indices are discrete, matching the paper's definition.
    with torch.no_grad():
        target_to_fragment = torch.cdist(target_points, fragment_points).min(dim=-1).values
        target_indices = target_to_fragment.topk(k, dim=1, largest=False).indices
    target_band = target_points.gather(1, target_indices.unsqueeze(-1).expand(-1, -1, 3))

    # The nearest-fragment search only chooses a discrete band. Keeping its
    # distance graph would waste memory; gradients still flow through the
    # selected coordinates below.
    with torch.no_grad():
        predicted_to_fragment = torch.cdist(
            predicted_points.detach(), fragment_points
        ).min(dim=-1).values
        predicted_indices = predicted_to_fragment.topk(k, dim=1, largest=False).indices
    predicted_band = predicted_points.gather(
        1, predicted_indices.unsqueeze(-1).expand(-1, -1, 3)
    )

    band_distances = torch.cdist(predicted_band, target_band).square()
    return band_distances.min(dim=-1).values.mean(dim=-1) + band_distances.min(dim=-2).values.mean(dim=-1)


def method_from_checkpoint(checkpoint):
    """Identify new Artifact-PCDiff checkpoints while keeping legacy loads safe."""
    method = checkpoint.get("method") if isinstance(checkpoint, dict) else None
    if method in {"pcdiff", "artifact_pcdiff"}:
        return method

    state = checkpoint.get("model_state", {}) if isinstance(checkpoint, dict) else {}
    if any("fracture_refinement." in key for key in state):
        return "artifact_pcdiff"
    return "pcdiff"
