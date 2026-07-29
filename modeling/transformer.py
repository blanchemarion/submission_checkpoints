"""
Self-Feedback Autoregressive Decoder-Only Transformer for Multi-Step Time Series Prediction
KV cache custom implementation to speed up the inference process.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
import time 


class _CausalSelfAttentionKV(nn.Module):
    """
    Causal self-attention with optional KV cache, using PyTorch SDPA (differentiable).
    - x: (B, T, d_model)
    - past_kv: (k, v) where k,v: (B, n_heads, T_past, head_dim)
    Returns:
      y: (B, T, d_model)
      present_kv (if use_cache): (k_all, v_all)
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.proj = nn.Linear(d_model, d_model, bias=True)

        self.dropout = dropout

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # (B, T, d) -> (B, H, T, Dh)
        B, T, D = x.shape
        return x.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        # (B, H, T, Dh) -> (B, T, d)
        B, H, T, Dh = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, H * Dh)

    def forward(
        self,
        x: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        *,
        attn_mask: Optional[torch.Tensor] = None,   # additive mask: (T_q, T_k)
        is_causal: bool = False,
        use_cache: bool = False,
        window_len: Optional[int] = None,          # keep last window_len keys/values if not None
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, T, _ = x.shape

        qkv = self.qkv(x)  # (B, T, 3d)
        q, k, v = qkv.chunk(3, dim=-1)

        q = self._split_heads(q)  # (B, H, T, Dh)
        k = self._split_heads(k)
        v = self._split_heads(v)

        if past_kv is not None:
            k_past, v_past = past_kv
            k_all = torch.cat([k_past, k], dim=2)
            v_all = torch.cat([v_past, v], dim=2)
        else:
            k_all, v_all = k, v

        if window_len is not None and k_all.size(2) > window_len:
            k_all = k_all[:, :, -window_len:, :]
            v_all = v_all[:, :, -window_len:, :]

        # SDPA: attn_mask should broadcast to (B, H, T_q, T_k)
        dropout_p = self.dropout if self.training else 0.0

        if attn_mask is not None:
            # attn_mask: (T_q, T_k) -> (1,1,T_q,T_k)
            attn_mask_4d = attn_mask.unsqueeze(0).unsqueeze(0)
        else:
            attn_mask_4d = None

        y = F.scaled_dot_product_attention(
            q, k_all, v_all,
            attn_mask=attn_mask_4d,
            dropout_p=dropout_p,
            is_causal=is_causal
        )  # (B, H, T, Dh)

        y = self._merge_heads(y)  # (B, T, d)
        y = self.proj(y)

        present_kv = (k_all, v_all) if use_cache else None
        return y, present_kv


class _DecoderBlockKV(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = _CausalSelfAttentionKV(d_model, n_heads, dropout=dropout)
        self.ln2 = nn.LayerNorm(d_model)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        *,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
        use_cache: bool = False,
        window_len: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        h = self.ln1(x)
        a, present_kv = self.attn(
            h,
            past_kv=past_kv,
            attn_mask=attn_mask,
            is_causal=is_causal,
            use_cache=use_cache,
            window_len=window_len,
        )
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x, present_kv


class TransformerModel(nn.Module):
    def __init__(
        self,
        n_vars=16,
        d_model=64,
        n_heads=4,
        n_layers=4,
        d_ff=None,
        dropout=0.1,
        T_in=70,
        T_out=21,
    ):
        super().__init__()

        self.n_vars = n_vars
        self.T_in = T_in
        self.T_out = T_out
        self.d_model = d_model
        d_ff = d_ff or 4 * d_model

        # Total token positions per (history + future) block
        self.total_tokens = T_in + T_out

        # Plain token projections
        self.in_proj = nn.Linear(n_vars, d_model)
        self.out_proj = nn.Linear(d_model, n_vars)

        # Positional embeddings (token-level)
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.total_tokens, d_model))
        nn.init.normal_(self.pos_embedding, std=0.02)
        self.dropout = nn.Dropout(dropout)

        # Replace nn.TransformerEncoder with explicit decoder-only blocks (KV-cache friendly)
        self.blocks = nn.ModuleList([
            _DecoderBlockKV(d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)

        # Learned start token (optional; you can still use zeros to match old behavior)
        self.future_start = nn.Parameter(torch.zeros(1, 1, self.d_model))
        nn.init.normal_(self.future_start, std=0.02)

    def _positions_block_relative(self, L: int, start_in_block: int, device):
        pos = torch.arange(L, device=device) + start_in_block
        if torch.any(pos >= self.total_tokens):
            raise ValueError(
                f"Pos index overflow: max={int(pos.max())} but total_tokens={self.total_tokens}. "
                "Increase total_tokens or reset block/offset earlier."
            )
        return pos

    def _run_blocks(
        self,
        x: torch.Tensor,
        *,
        attn_mask: Optional[torch.Tensor] = None,  # (Tq, Tk) additive
        is_causal: bool = False,
        use_cache: bool = False,
        cache: Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]] = None,
        window_len: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        """
        Runs the stack.
        If use_cache=True, returns updated per-layer cache list.
        """
        new_cache = [] if use_cache else None
        for li, blk in enumerate(self.blocks):
            past_kv = None if cache is None else cache[li]
            x, present_kv = blk(
                x,
                past_kv=past_kv,
                attn_mask=attn_mask,
                is_causal=is_causal,
                use_cache=use_cache,
                window_len=window_len,
            )
            if use_cache:
                new_cache.append(present_kv)  # type: ignore[arg-type]
        x = self.ln_f(x)
        return x, new_cache

    def forward_teacher_forcing(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        block_offset: int = 0,
    ) -> torch.Tensor:
        """
        Teacher-forced AR training (GPT-style), but for continuous values.

        At future step t, the model sees ground-truth y[0],...,y[t-1] (via shifted inputs) and
        predicts y[t]. Aligns with ``forward_autoregressive_kvcache`` at inference: same
        embeddings, positions, causal stack, and ``future_start`` as the first future input token.

        Args:
            x: (B, T_in, n_vars)  ground-truth context / history
            y: (B, T_out, n_vars) ground-truth future targets
            block_offset: absolute position index of the first history token within the block
                (same convention as ``forward_autoregressive_kvcache``).

        Returns:
            pred: (B, T_out, n_vars) predictions for each future step (aligned with y).
        """
        B, T_in, V = x.shape
        B2, T_out, V2 = y.shape
        if B != B2 or V != V2:
            raise ValueError("x and y must match batch and n_vars")
        if T_in != self.T_in or T_out != self.T_out:
            raise ValueError("x/y lengths must match model T_in/T_out")

        device = x.device

        hist_tokens = self.in_proj(x)
        bos = self.future_start.expand(B, 1, self.d_model).to(device=device, dtype=hist_tokens.dtype)

        if T_out > 1:
            y_shift_tokens = self.in_proj(y[:, :-1, :])
            fut_in_tokens = torch.cat([bos, y_shift_tokens], dim=1)
        else:
            fut_in_tokens = bos

        all_tokens = torch.cat([hist_tokens, fut_in_tokens], dim=1)
        pos = self._positions_block_relative(all_tokens.size(1), block_offset, device=device)
        all_tokens = all_tokens + self.pos_embedding[:, pos, :].to(dtype=all_tokens.dtype)
        all_tokens = self.dropout(all_tokens)

        out, _ = self._run_blocks(
            all_tokens,
            attn_mask=None,
            is_causal=True,
            use_cache=False,
            cache=None,
            window_len=None,
        )
        future_tokens = out[:, self.T_in :, :]
        pred = self.out_proj(future_tokens)
        return pred

    # ---------- TRAINING forward (KV-CACHED) ----------
    def forward_autoregressive_kvcache(self, x, block_offset=0, *, use_learned_start: bool = True):
        """
        KV-cached closed-loop rollout for TRAINING:
          - Prefill once on (context + start_token)
          - Decode T_out steps with 1-token passes, updating cache each time
        Returns: (B, T_out, n_vars)

        IMPORTANT:
        This is not exactly identical to forward_autoregressive_old, because cached mode
        doesn't recompute past-token hidden states under the shifted local mask.
        """
        B = x.size(0)
        device = x.device

        # Embed context
        """ctx = self.in_proj(x)      # (B, T_in, d)
        ctx = self.dropout(ctx)

        # Start token (to predict y0)
        if use_learned_start:
            start_tok = self.future_start.expand(B, -1, -1).to(device=device, dtype=ctx.dtype)
        else:
            start_tok = torch.zeros(B, 1, self.d_model, device=device, dtype=ctx.dtype)

        # Prefill tokens: [x_0 .. x_{T_in-1}, start]
        tokens0 = torch.cat([ctx, start_tok], dim=1)  # (B, T_in+1, d)

        # Positions for prefill
        prefill_start_pos = block_offset  # first token position inside block
        pos0 = self._positions_block_relative(tokens0.size(1), prefill_start_pos, device=device)
        tokens0 = tokens0 + self.pos_embedding[:, pos0, :].to(dtype=tokens0.dtype)

        tokens0 = self.dropout(tokens0)"""

        ctx = self.in_proj(x)
        start_tok = self.future_start.expand(B, -1, -1).to(device=device, dtype=ctx.dtype)

        tokens0 = torch.cat([ctx, start_tok], dim=1)
        pos0 = self._positions_block_relative(tokens0.size(1), block_offset, device=device)
        tokens0 = tokens0 + self.pos_embedding[:, pos0, :].to(dtype=tokens0.dtype)
        tokens0 = self.dropout(tokens0)

        # Build cache list (one entry per layer)
        cache: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [None] * len(self.blocks)

        # Prefill: full causal attention (sequence length is only T_in+1 so no need for window mask here)
        h0, cache = self._run_blocks(
            tokens0,
            attn_mask=None,
            is_causal=True,     # standard causal for prefill
            use_cache=True,
            cache=cache,
            window_len=None,    # no truncation during prefill
        )

        # First prediction y0 from last token
        last_h = h0[:, -1:, :]
        pred = self.out_proj(last_h)
        preds = [pred]

        # Keep KV cache window size ~ (T_in + 1) to mimic your sliding context size
        #kv_window = self.T_in + 1

        # Decode steps: each step processes ONLY the new token
        for i in range(1, self.T_out):
            # Next input token is in_proj(pred_{i-1})
            """next_tok = self.dropout(self.in_proj(pred))  # (B,1,d)
            # Add positional embedding at the absolute position
            pos_idx = block_offset + self.T_in + i  # same convention as old code
            pos = self._positions_block_relative(1, pos_idx, device=device)
            next_tok = next_tok + self.pos_embedding[:, pos, :].to(dtype=next_tok.dtype)"""

            next_tok = self.in_proj(pred)
            pos_idx = block_offset + self.T_in + i
            pos = self._positions_block_relative(1, pos_idx, device=device)
            next_tok = next_tok + self.pos_embedding[:, pos, :].to(dtype=next_tok.dtype)
            next_tok = self.dropout(next_tok)

            # Run 1-token pass with cache, truncating KV to window
            h_new, cache = self._run_blocks(
                next_tok,
                attn_mask=None,
                is_causal=False,     # q_len=1 and KV are only past+current => no future leakage
                use_cache=True,
                cache=cache,
                window_len= None #kv_window,
            )

            pred = self.out_proj(h_new)  # (B,1,n_vars)
            preds.append(pred)

        return torch.cat(preds, dim=1)

    
    def count_parameters(self):
        """Count the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)




def create_model_cached(n_vars=16, d_model=64, n_heads=4, n_layers=4, d_ff=None, dropout=0.1,
                 T_in=70, T_out=21, device='cpu'):

    model = TransformerModel(
        n_vars=n_vars,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        dropout=dropout,
        T_in=T_in,
        T_out=T_out,
    ).to(device)

    return model



if __name__ == "__main__":
    # Test the model
    print("Testing TransformerModel model...")
    
    # Model parameters
    n_vars = 16
    T_in = 70
    T_out = 21
    batch_size = 4
    
    # Create model
    model = create_model_cached(
        n_vars=n_vars,
        d_model=64,
        n_heads=4,
        n_layers=4,
        T_in=T_in,
        T_out=T_out,
    )
    
    print(f"\nModel created:")
    print(f"  Parameters: {model.count_parameters():,}")
    print(f"  Input shape: (batch, {T_in}, {n_vars})")
    print(f"  Output shape: (batch, {T_out}, {n_vars})")
    
    # Test input
    x = torch.randn(batch_size, T_in, n_vars)

    # Test autoregressive forward pass
    print(f"\n{'='*60}")
    print("Testing AUTOREGRESSIVE inference")
    print('='*60)
    print(f"Input shape: {x.shape}")
    
    with torch.no_grad():
        start = time.time()
        output_autoreg = model.forward_autoregressive_kvcache(x)
        elapsed_autoreg = time.time() - start
        
        print(f"  Output shape: {output_autoreg.shape}")
        print(f"  Expected: ({batch_size}, {T_out}, {n_vars})")
        print(f"  Time: {elapsed_autoreg*1000:.2f} ms")
        
        if output_autoreg.shape == (batch_size, T_out, n_vars):
            print("  [OK] Autoregressive inference passed!")
        else:
            print("  [ERROR] Shape mismatch!")

    # Teacher-forcing forward + backward smoke test
    print(f"\n{'='*60}")
    print("Testing TEACHER FORCING (train regime)")
    print('='*60)
    y_tgt = torch.randn(batch_size, T_out, n_vars)
    pred_tf = model.forward_teacher_forcing(x, y_tgt)
    print(f"  pred shape: {pred_tf.shape}")
    if pred_tf.shape == (batch_size, T_out, n_vars):
        print("  [OK] Teacher forcing shape passed!")
    else:
        print("  [ERROR] Shape mismatch!")
    loss = F.mse_loss(pred_tf, y_tgt)
    loss.backward()
    print(f"  [OK] Backward pass (loss={loss.item():.6f})")
    
    # Compare outputs
    print(f"\n{'='*60}")
    print(f"  Autoregressive output stats: mean={output_autoreg.mean():.4f}, std={output_autoreg.std():.4f}")
    
    print(f"\n{'='*60}")
    print("[OK] All tests passed!")
    print('='*60)