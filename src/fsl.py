
"""
FSL Implementation
@author : Arnaud Ullens
@created :  8th dec.2025
last modification : 11th jan.2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict
import numpy as np


class SoftOrthogonalRestriction(nn.Module):
    """
    Learns a linear mapping between dimensions while enforcing soft orthogonality.
    This helps preserve structural properties during dimension reduction/expansion.
    """
    def __init__(self, dim_source: int, dim_target: int):
        super().__init__()
        self.dim_source = dim_source
        self.dim_target = dim_target
        
        # SVD-based initialization for better convergence
        weight = torch.empty(dim_target, dim_source)
        nn.init.orthogonal_(weight)
        
        # Add noise to prevent collapse
        weight = weight + 0.01 * torch.randn_like(weight)
        self.weight = nn.Parameter(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight)

    def orthogonality_loss(self) -> torch.Tensor:
        """
        Calculates loss to maintain orthogonality (Gram matrix ≈ Identity).
        Includes spectral regularization to penalize small singular values.
        """
        if self.dim_target >= self.dim_source:
            gram = self.weight.t() @ self.weight
            eye = torch.eye(self.dim_source, device=self.weight.device)
        else:
            gram = self.weight @ self.weight.t()
            eye = torch.eye(self.dim_target, device=self.weight.device)
        
        # MSE between Gram matrix and Identity
        ortho_loss = F.mse_loss(gram, eye)
        
        # Spectral regularization: prevents the matrix from becoming rank-deficient
        s = torch.linalg.svdvals(self.weight)
        spectral_reg = torch.mean(torch.relu(0.1 - s))
        
        return ortho_loss + 0.1 * spectral_reg

class MultiHeadSheafLaplacian(nn.Module):
    """
    Multi-Head FSL & Non-linear projectors (MLPs)
    Complexity: O(N * d^2) (Linear in N).
    """
    def __init__(self, n_contexts: int, context_dim: int, latent_dim: int = 16, 
                 attention_dim: int = 32, n_heads: int = 4):
        super().__init__()
        self.n_contexts = n_contexts
        self.context_dim = context_dim
        self.total_latent_dim = latent_dim
        self.n_heads = n_heads
        self.head_dim = latent_dim // n_heads
        self.attention_dim = attention_dim
        
        assert latent_dim % n_heads == 0, "latent_dim must be divisible by n_heads"

        # 1. Local Projectors (Asset-Specific)
        self.projectors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(context_dim, latent_dim),
                nn.LayerNorm(latent_dim),
                nn.GELU(),
                nn.Linear(latent_dim, latent_dim)
            ) for _ in range(n_contexts)
        ])
        
        # 2. Local Deprojectors (Asset-Specific)
        # z_global -> x_i_new
        self.deprojectors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim, latent_dim),
                nn.GELU(),
                nn.Linear(latent_dim, context_dim)
            ) for _ in range(n_contexts)
        ])
        
        # Orthogonal Initialization (to prevent initial collapse)
        for m in self.projectors: 
            if isinstance(m, nn.Linear): nn.init.orthogonal_(m.weight)
        for m in self.deprojectors:
            if isinstance(m, nn.Linear): nn.init.orthogonal_(m.weight)

        # 3. Multi-Head Linear Attention (Shared Global Dynamics)
        # Project input to n_heads * attention_dim
        self.query_proj = nn.Linear(context_dim, n_heads * attention_dim)
        self.key_proj = nn.Linear(context_dim, n_heads * attention_dim)
        
        # Value projection: Transform the latent Z before diffusion
        self.value_proj = nn.Linear(latent_dim, latent_dim)
        
        # Output projection after concatenating heads
        self.out_proj = nn.Linear(latent_dim, latent_dim)
        
        self.alpha_logit = nn.Parameter(torch.tensor(0.0))

    def forward(self, sections: List[torch.Tensor], diffusion_scale: float = 1.0) -> Tuple[List[torch.Tensor], torch.Tensor]:
        device = sections[0].device
        batch_size = sections[0].shape[0]
        
        # Stack inputs: (Batch, N, Context_Dim)
        x_stack = torch.stack(sections, dim=1) 
        
        # Projection into the Latent Space
        # optimizing with Conv1D ?
        z_list = [proj(sections[i]) for i, proj in enumerate(self.projectors)]
        z_stack = torch.stack(z_list, dim=1)
        
        # Multi-Head Attention Preparation
        Q = self.query_proj(x_stack).view(batch_size, self.n_contexts, self.n_heads, self.attention_dim)
        K = self.key_proj(x_stack).view(batch_size, self.n_contexts, self.n_heads, self.attention_dim)
        
        # V: (Batch, N, Heads, Head_Dim) -> The latent Z serves as Value
        # Transform Z to mix before diffusion
        V = self.value_proj(z_stack).view(batch_size, self.n_contexts, self.n_heads, self.head_dim)
        
        # Feature map phi(x) = elu(x) + 1 for Linear Attention
        Q_prime = F.elu(Q) + 1.0
        K_prime = F.elu(K) + 1.0
        
        # Linear Multi-Head Diffusion
        # Linear Attention formula: (Q @ (K.T @ V)) / (Q @ K.T @ 1)
        
        # Numerator: KV_sum = Sum_j (K_j * V_j) -> (Batch, Heads, Attn_Dim, Head_Dim)
        KV_sum = torch.einsum('bnhd,bnhv->bhdv', K_prime, V) # einsum for my GPU :)
        
        # Z_num = Q_i * KV_sum -> (Batch, N, Heads, Head_Dim)
        Z_num = torch.einsum('bnhd,bhdv->bnhv', Q_prime, KV_sum)
        
        # Denominator: K_sum = Sum_j (K_j) -> (Batch, Heads, Attn_Dim)
        K_sum = K_prime.sum(dim=1) 
        # Z_denom = Q_i * K_sum -> (Batch, N, Heads)
        Z_denom = torch.einsum('bnhd,bhd->bnh', Q_prime, K_sum).unsqueeze(-1)
        
        # Result per head
        z_diffused_heads = Z_num / (Z_denom + 1e-6)
        
        # Concatenation of heads (Flatten) -> (Batch, N, Total_Latent_Dim)
        z_diffused = z_diffused_heads.reshape(batch_size, self.n_contexts, self.total_latent_dim)
        z_diffused = self.out_proj(z_diffused) # Final mixing between heads

        # Deprojection and Update
        alpha = torch.sigmoid(self.alpha_logit) * diffusion_scale
        new_sections = []
        
        # Apply specific deprojectors for each asset
        for i, deproj in enumerate(self.deprojectors):
            term_diffused = deproj(z_diffused[:, i, :])
            # Residual update: X_new = (1-a)X + a*Diffused
            new_sections.append((1 - alpha) * sections[i] + alpha * term_diffused)
            
        return new_sections, None # No explicit adjacency (compatibility)

class ScaleTransition(nn.Module):
    """
    Manages the data flow between hierarchy levels (Fine <-> Coarse).
    Uses Attention for pooling (down) and Projections for unpooling (up).
    """
    def __init__(self, n_contexts_fine: int, n_contexts_coarse: int, context_dim: int):
        super().__init__()
        self.n_fine = n_contexts_fine
        self.n_coarse = n_contexts_coarse
        self.context_dim = context_dim
        self.pool_ratio = n_contexts_fine // n_contexts_coarse
        
        assert n_contexts_fine % n_contexts_coarse == 0, "Fine must be divisible by coarse"
        
        # Components for attention-based pooling
        self.pool_query = nn.Linear(context_dim, context_dim)
        self.pool_key = nn.Linear(context_dim, context_dim)
        self.pool_value = nn.Linear(context_dim, context_dim)
        
        # Components for unpooling (upsampling)
        self.unpool_proj = nn.Sequential(
            nn.Linear(context_dim, context_dim * self.pool_ratio),
            nn.LayerNorm(context_dim * self.pool_ratio),
            nn.GELU(),
            nn.Linear(context_dim * self.pool_ratio, context_dim * self.pool_ratio)
        )
        
    def pool_up(self, sections_fine: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Aggregates fine-grained sections into coarse sections using attention.
        """
        sections_coarse = []
        
        for i in range(self.n_coarse):
            start_idx = i * self.pool_ratio
            end_idx = start_idx + self.pool_ratio
            
            # Stack relevant fine sections
            group = torch.stack(sections_fine[start_idx:end_idx], dim=1)
            group = group.flatten(start_dim=2)
            
            # Compute attention weights within the group
            Q = self.pool_query(group.mean(dim=1, keepdim=True))
            K = self.pool_key(group)
            V = self.pool_value(group)
            
            scores = torch.bmm(Q, K.transpose(1, 2)) / np.sqrt(self.context_dim)
            attn_weights = F.softmax(scores, dim=-1)
            
            # Weighted aggregation
            pooled = torch.bmm(attn_weights, V).squeeze(1)
            sections_coarse.append(pooled)
        
        return sections_coarse
    
    def unpool_down(self, sections_coarse: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Distributes coarse information back to fine sections via projection.
        """
        sections_fine = []
        
        for coarse_section in sections_coarse:
            # Expand dimensions
            unpooled = self.unpool_proj(coarse_section)  # (B, D * pool_ratio)
            unpooled = unpooled.view(-1, self.pool_ratio, self.context_dim)  # (B, pool_ratio, D)
            
            # Split into individual fine sections
            for k in range(self.pool_ratio):
                sections_fine.append(unpooled[:, k, :])
        
        return sections_fine


class HierarchicalFSL(nn.Module):
    """
    Main Hierarchical Folded Sheaf Laplacian Model.
    Structure acts like a Graph U-Net:
    1. Bottom-up: Encodes and coarsens data (extracts global features).
    2. Top-down: Refines features using residuals and diffusion.
    """
    def __init__(self, 
                 scales: List[int] = [16, 8, 4],
                 context_dim: int = 32,
                 attention_dim: int = 64,
                 latent_dim: int = 16,
                 diffusion_steps: List[int] = [2, 3, 4]):
        super().__init__()
        self.scales = scales
        self.context_dim = context_dim
        self.n_scales = len(scales)
        self.diffusion_steps = diffusion_steps
        
        assert len(diffusion_steps) == len(scales), "Need diffusion steps for each scale"
        
        # Encoders (applied only at the finest scale)
        self.encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(context_dim, context_dim),
                nn.LayerNorm(context_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(context_dim, context_dim),
                nn.LayerNorm(context_dim)
            )
            for _ in range(scales[0])
        ])
        
        # Diffusion layers for each scale
        self.sheaves = nn.ModuleList([
            MultiHeadSheafLaplacian(
                n_contexts=n,
                context_dim=context_dim,
                latent_dim=latent_dim,
                attention_dim=attention_dim,
                n_heads=4
            )
            for n in scales
        ])
        
        # Transition layers (Pooling/Unpooling)
        self.transitions = nn.ModuleList([
            ScaleTransition(
                n_contexts_fine=scales[i],
                n_contexts_coarse=scales[i+1],
                context_dim=context_dim
            )
            for i in range(len(scales) - 1)
        ])
        
        # Decoders (applied only at the finest scale)
        self.decoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(context_dim, context_dim),
                nn.GELU(),
                nn.Linear(context_dim, context_dim)
            )
            for _ in range(scales[0])
        ])
    
    def forward(self, inputs: List[torch.Tensor]) -> Dict:
        """
        Forward pass with automatic volatility gating.
        inputs: List of tensors corresponding to the finest scale.
        """
        
        # --- Gating Mechanism ---
        # Detects high volatility/magnitude. If data is too volatile,
        # we reduce diffusion to prevent over-smoothing.
        stacked_inputs = torch.stack(inputs)
        avg_magnitude = torch.mean(torch.abs(stacked_inputs))
        
        diffusion_scale = 1.0 # This will change in the future !!
        if avg_magnitude > 0.9: 
            diffusion_scale = 0.1 # Reduce diffusion by 90%
        
        # Initial Encoding
        sections_fine = [enc(x) for enc, x in zip(self.encoders, inputs)]
        
        # --- BOTTOM-UP: Building the Pyramid ---
        pyramid = [sections_fine]
        adjacency_pyramid = []
        
        for i, (transition, sheaf) in enumerate(zip(self.transitions, self.sheaves[1:])):
            # Pool to coarser scale
            sections_coarse = transition.pool_up(pyramid[-1])
            
            # Diffuse information at this coarse level
            for _ in range(self.diffusion_steps[i+1]):
                sections_coarse, adj_coarse = sheaf(sections_coarse, diffusion_scale=diffusion_scale)
            
            pyramid.append(sections_coarse)
            adjacency_pyramid.append(adj_coarse)
        
        # --- TOP-DOWN: Refinement Cascade ---
        for i in reversed(range(len(self.transitions))):
            # Unpool from coarse to fine
            sections_from_coarse = self.transitions[i].unpool_down(pyramid[i+1])
            
            # Residual connection: Mix original fine features with upsampled coarse features
            residual_weight = 0.3
            for j in range(len(pyramid[i])):
                pyramid[i][j] = pyramid[i][j] + residual_weight * sections_from_coarse[j]
            
            # Re-diffuse at fine scale with enriched information
            for _ in range(self.diffusion_steps[i]):
                pyramid[i], adj_fine = self.sheaves[i](pyramid[i], diffusion_scale=diffusion_scale)
            
            # Update adjacency for return
            if i < len(adjacency_pyramid):
                adjacency_pyramid[i] = adj_fine
        
        # Final Reconstruction
        final_sections = pyramid[0]
        outputs = [dec(s) for dec, s in zip(self.decoders, final_sections)]
        
        # Compute Structural Losses
        h1_loss = self.compute_h1_multiscale(pyramid, adjacency_pyramid)
        ortho_loss = self.compute_ortho_loss()
        
        return {
            'outputs': outputs,
            'sections': final_sections,
            'pyramid': pyramid,
            'adjacency_pyramid': adjacency_pyramid,
            'h1_score': h1_loss,
            'ortho_loss': ortho_loss,
            'diffusion_scale': diffusion_scale
        }
    
    def compute_h1_multiscale(self, pyramid: List[List[torch.Tensor]], 
                             adjacency_pyramid: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute cohomology H1 across scales.
        """
        total_h1 = 0.0
        # Stronger weights for coarser scales (global structures)
        scale_weights = [1.0, 1.5, 2.0] 
        
        # Iterate over pyramid levels (except level 0 which is the raw input)
        for scale_idx, sections in enumerate(pyramid[1:]):
            # Retrieve the Sheaf corresponding to this scale
            # Note: sheaves[0] is for the fine scale, sheaves[1] for the next, etc.
            sheaf = self.sheaves[scale_idx + 1]
            
            n_contexts = len(sections)
            scale_h1 = 0.0
            count = 0
            
            # --- Stochastic sampling strategy ---
            # Instead of doing N^2 pairs, we take K random pairs per node.
            # This keeps the complexity linear O(N * K).
            n_samples = 5 
            
            # If we have an adjacency (non-factorized case), we use it to guide.
            # Otherwise (factorized case), we sample uniformly or via cosine similarity.
            adj = adjacency_pyramid[scale_idx] if scale_idx < len(adjacency_pyramid) else None
            
            for i in range(n_contexts):
                # Select neighbors 'j' to compare
                if adj is not None:
                    # We take the strongest neighbors based on adjacency
                    weights = adj[:, i, :].mean(dim=0) # Average over the batch
                    indices = torch.topk(weights, k=min(n_samples, n_contexts), largest=True).indices
                else:
                    # Random sampling (Monte Carlo H1)
                    indices = torch.randint(0, n_contexts, (n_samples,), device=sections[0].device)
                
                for j in indices:
                    if i == j: continue
                    
                    # Factorized Case (O(N)): Generate Rho on the fly
                    # Rho_ij = Deproj_j @ Proj_i
                    # Rho_ji = Deproj_i @ Proj_j
                    
                    #ANCHOR: For efficiency, we apply Proj then Deproj directly on tensors
                    # without constructing the dense W matrix. This will maybe change in the future
                                            
                    # Cycle: x_i -> Proj_i -> z -> Deproj_j (x_j_hat) -> Proj_j -> z' -> Deproj_i (cycle)
                    
                    # x_i projected into the latent space of i
                    z_i = sheaf.projectors[i](sections[i]) 
                    
                    # Transport to j (in the latent space, we assume trivial transport or implicit adjacency)
                    # For standard H1: we compare Rho_ji( Rho_ij(x_i) ) to x_i
                    
                    # Step 1: Rho_ij(x_i) : Transport from i to j
                    # In the factorized model: x_i -> z -> x_j (via deproj_j)
                    x_j_simulated = sheaf.deprojectors[j](z_i)
                    
                    # Step 2: Rho_ji(...) : Return from j to i
                    z_j = sheaf.projectors[j](x_j_simulated)
                    cycle = sheaf.deprojectors[i](z_j)
                    
                    # Implicit weight (assumed 1.0 or based on latent similarity)
                    weight = 1.0

                    # 2. Calculation of the consistency error (H1)
                    if weight > 0.05:
                        diff = (cycle - sections[i]).pow(2).mean()
                        scale_h1 += weight * diff
                        count += 1
            
            if count > 0:
                w = scale_weights[min(scale_idx, len(scale_weights)-1)]
                total_h1 += w * (scale_h1 / count)
        
        return total_h1
    
    def compute_ortho_loss(self) -> torch.Tensor:
        loss = 0.0
        count = 0
        
        for sheaf in self.sheaves:
            # Iterate over all submodules to find Linear layers
            # Projectors
            for proj in sheaf.projectors:
                for m in proj.modules():
                    if isinstance(m, nn.Linear):
                        w = m.weight
                        # Tall matrix
                        if w.shape[0] > w.shape[1]:
                            gram = w.t() @ w
                            eye = torch.eye(w.shape[1], device=w.device)
                        # Wide matrix
                        else: 
                            gram = w @ w.t()
                            eye = torch.eye(w.shape[0], device=w.device)
                        loss += F.mse_loss(gram, eye)
                        count += 1
            
            # Deprojectors
            for deproj in sheaf.deprojectors:
                for m in deproj.modules():
                    if isinstance(m, nn.Linear):
                        w = m.weight
                        if w.shape[0] > w.shape[1]:
                            gram = w.t() @ w
                            eye = torch.eye(w.shape[1], device=w.device)
                        else:
                            gram = w @ w.t()
                            eye = torch.eye(w.shape[0], device=w.device)
                        loss += F.mse_loss(gram, eye)
                        count += 1
                        
        return loss / (count + 1e-8)

class TopologicalContrastiveLoss(nn.Module):
    """
    Simple contrastive loss on H1 scores.
    """
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin
    
    def forward(self, h1_coherent: torch.Tensor, h1_incoherent: torch.Tensor) -> torch.Tensor:
        # Coherent data should have low H1, Incoherent data should have high H1
        loss_coherent = h1_coherent
        loss_incoherent = F.relu(self.margin - h1_incoherent)
        
        return loss_coherent + loss_incoherent
    
class TopologicalTripletLoss(nn.Module):
    """
    Advanced Triplet Loss optimized for Sheaf Cohomology (H1).
    Objectives:
    1. Structural Consistency: Real data should have low H1.
    2. Temporal Stability: Anchor and Positive should have similar H1.
    3. Discrimination: Anchor and Negative (noise) should have very different H1.
    """
    def __init__(self, margin: float = 0.5, structural_weight: float = 1.0):
        super().__init__()
        self.margin = margin
        self.structural_weight = structural_weight
    
    def forward(self, 
                h1_anchor: torch.Tensor, 
                h1_positive: torch.Tensor, 
                h1_negative: torch.Tensor,
                anchor_embedding: torch.Tensor = None,
                positive_embedding: torch.Tensor = None) -> torch.Tensor:
        """
        h1_anchor : Score H1 of actual window (should be low)
        h1_positive : Score H1 of a nearby or augmented window (should be low and close to anchor)
        h1_negative : Score H1 of a noisy/shuffled window (should be high)
        """
        
        # Structural Term: Minimize H1 for valid data (Anchor & Positive)
        structural_loss = h1_anchor + h1_positive
        
        # Triplet Term: Ensure Anchor is topologically closer to Positive than Negative
        d_pos = torch.abs(h1_anchor - h1_positive)
        d_neg = torch.abs(h1_anchor - h1_negative)
        
        triplet_loss = torch.relu(d_pos - d_neg + self.margin)
        
        # Standard embedding distance regularization
        embedding_loss = 0.0
        if anchor_embedding is not None and positive_embedding is not None:
            embedding_loss = F.mse_loss(anchor_embedding, positive_embedding)
            
        return self.structural_weight * structural_loss + triplet_loss + 0.1 * embedding_loss