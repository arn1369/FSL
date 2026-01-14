import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict
import numpy as np

class SoftOrthogonalRestriction(nn.Module):
    def __init__(self, dim_source: int, dim_target: int, prior_weight=None):
        super().__init__()
        self.dim_source = dim_source
        self.dim_target = dim_target
        
        if prior_weight is not None:
            self.weight = nn.Parameter(prior_weight.clone() + 0.02 * torch.randn(dim_target, dim_source, device=prior_weight.device))
        else:
            weight = torch.empty(dim_target, dim_source)
            nn.init.orthogonal_(weight)
            self.weight = nn.Parameter(weight + 0.02 * torch.randn_like(weight))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight)

    def orthogonality_loss(self) -> torch.Tensor:
        if self.dim_target >= self.dim_source:
            gram = self.weight.t() @ self.weight
            eye = torch.eye(self.dim_source, device=self.weight.device)
        else:
            gram = self.weight @ self.weight.t()
            eye = torch.eye(self.dim_target, device=self.weight.device)
        return F.mse_loss(gram, eye)

class DynamicSheafLaplacian(nn.Module):
    def __init__(self, n_contexts: int, context_dims: List[int], attention_dim: int = 64, prior=None):
        super().__init__()
        self.n_contexts = n_contexts
        self.context_dims = context_dims
        self.alpha_logit = nn.Parameter(torch.tensor(0.0))
        
        self.query_projs = nn.ModuleList([nn.Sequential(nn.Linear(d, attention_dim), nn.LayerNorm(attention_dim)) for d in context_dims])
        self.key_projs = nn.ModuleList([nn.Sequential(nn.Linear(d, attention_dim), nn.LayerNorm(attention_dim)) for d in context_dims])
        
        self.restrictions = nn.ModuleDict()
        for i in range(n_contexts):
            for j in range(n_contexts):
                if i != j:
                    p_weight = None
                    if prior is not None:
                        device = prior.device 
                        p_weight = torch.eye(context_dims[j], context_dims[i], device=device) * prior[i, j]
                    
                    self.restrictions[f"{j}_{i}"] = SoftOrthogonalRestriction(
                        context_dims[j], context_dims[i], prior_weight=p_weight
                    )

    def compute_adjacency(self, sections: List[torch.Tensor]) -> torch.Tensor:
        batch_size = sections[0].shape[0]
        device = sections[0].device
        pooled = [s.mean(dim=1) if s.dim() == 3 else s for s in sections]

        Q = torch.stack([self.query_projs[i](pooled[i]) for i in range(self.n_contexts)], dim=1)
        K = torch.stack([self.key_projs[i](pooled[i]) for i in range(self.n_contexts)], dim=1)
        
        scores = torch.bmm(Q, K.transpose(1, 2)) / np.sqrt(Q.shape[-1])
        mask = torch.eye(self.n_contexts, device=device).bool().unsqueeze(0)
        scores = scores.masked_fill(mask, -1e9)
        return F.softmax(scores, dim=-1)

    def forward(self, sections: List[torch.Tensor], diffusion_scale: float = 1.0) -> Tuple[List[torch.Tensor], torch.Tensor]:
        batch_size = sections[0].shape[0]
        adj = self.compute_adjacency(sections)
        alpha = torch.sigmoid(self.alpha_logit) * diffusion_scale
        
        new_sections = []
        for i in range(self.n_contexts):
            diff_term = torch.zeros_like(sections[i])
            for j in range(self.n_contexts):
                if i == j: continue
                transported = self.restrictions[f"{j}_{i}"](sections[j])
                weight = adj[:, i, j].view(batch_size, *([1] * (sections[i].dim() - 1)))
                diff_term += weight * transported
            new_sections.append((1 - alpha) * sections[i] + alpha * diff_term)
        return new_sections, adj

class HierarchicalFSL(nn.Module):
    def __init__(self, scales=[16, 4, 1], context_dim=32, attention_dim=64, diffusion_steps=[3, 5, 5], prior=None):
        super().__init__()
        self.scales = scales
        self.diffusion_steps = diffusion_steps
        
        self.encoders = nn.ModuleList([nn.Linear(context_dim, context_dim) for _ in range(scales[0])])
        self.decoders = nn.ModuleList([nn.Linear(context_dim, context_dim) for _ in range(scales[0])])
        
        self.sheaves = nn.ModuleList([
            DynamicSheafLaplacian(n, [context_dim]*n, attention_dim, prior=(prior if i==0 else None))
            for i, n in enumerate(scales)
        ])
        
        #ANCHOR: transitions are simple :((
        self.pool = nn.AvgPool1d(kernel_size=scales[0]//scales[1]) if len(scales)>1 else None

    def forward(self, inputs: List[torch.Tensor]) -> Dict:
        # Gatin on volatility
        mag = torch.mean(torch.abs(torch.stack(inputs)))
        diff_scale = 1.0 if mag < 0.8 else 0.1
        
        # Encode
        sections = [enc(x) for enc, x in zip(self.encoders, inputs)]
        
        # Diffusion at fine scale
        for _ in range(self.diffusion_steps[0]):
            sections, adj = self.sheaves[0](sections, diffusion_scale=diff_scale)
        
        # Calculate H1 (with contrast)
        h1_score, individual_h1 = self.compute_h1_with_contrast(sections, adj, self.sheaves[0])
        
        return {
            'outputs': [dec(s) for dec, s in zip(self.decoders, sections)],
            'sections': sections,
            'h1_score': h1_score,
            'individual_h1': individual_h1,
            'ortho_loss': sum(sheaf.restrictions[k].orthogonality_loss() 
                             for sheaf in self.sheaves for k in sheaf.restrictions) / (len(self.sheaves)*10)
        }

    def compute_h1_with_contrast(self, sections, adj, sheaf):
        h1_global = 0.0
        n = len(sections)
        individual_errors = torch.zeros(n, device=sections[0].device)
        for i in range(n):
            asset_error = 0.0
            count = 0
            for j in range(n):
                if i == j: continue
                # Cycle: i -> j -> i
                rho_ij = sheaf.restrictions[f"{i}_{j}"]
                rho_ji = sheaf.restrictions[f"{j}_{i}"]
                
                z_i = sections[i]
                cycle = rho_ji(rho_ij(z_i))
                
                # Contrast: Error weighted by dissimilarity
                sim = F.cosine_similarity(z_i, sections[j], dim=-1).mean()
                contrast = (1.5 - sim.clamp(min=-0.5, max=0.8))
                
                err = adj[:, i, j].mean() * contrast * F.mse_loss(cycle, z_i)
                
                asset_error += err
                count += 1
                
            individual_errors[i] = asset_error / (count + 1e-8)
            h1_global += asset_error
            
        return h1_global / (n * (n-1) + 1e-8), individual_errors

class TopologicalTripletLoss(nn.Module):
    def __init__(self, margin=0.5):
        super().__init__()
        self.margin = margin
    def forward(self, h1_anchor, h1_pos, h1_neg):
        # On multiplie par 100 pour la visibilité numérique
        d_pos = h1_anchor + h1_pos
        d_neg = h1_neg
        return torch.relu(self.margin + d_pos - d_neg)