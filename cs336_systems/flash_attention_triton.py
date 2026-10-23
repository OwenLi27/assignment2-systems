import torch
import triton
import triton.language as tl
from cs336_systems.flash_attention_pytorch import my_backward

# 每个program处理一个Q tile
@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr
):
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)
    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    Q_tile = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero") # shape: (Q_TILE_SIZE, D)

    m = tl.full((Q_TILE_SIZE,), float('-inf'), dtype=tl.float32) # shape: (Q_TILE_SIZE,)
    l = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32) # shape: (Q_TILE_SIZE,)
    O_tile = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32) # shape: (Q_TILE_SIZE, D)

    for j in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        K_tile = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero") # shape: (K_TILE_SIZE, D)
        V_tile = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero") # shape: (K_TILE_SIZE, D)

        # Compute attention scores
        # Keep FP32 inputs at IEEE precision rather than the default TF32.
        scores = tl.dot(Q_tile, tl.trans(K_tile), out_dtype=tl.float32, input_precision="ieee") * scale # shape: (Q_TILE_SIZE, K_TILE_SIZE)

        # Apply causal masking if needed
        if is_causal:
            query_indices = tl.arange(0, Q_TILE_SIZE) + query_tile_index * Q_TILE_SIZE
            key_indices = tl.arange(0, K_TILE_SIZE) + j * K_TILE_SIZE
            mask = query_indices[:, None] < key_indices[None, :]
            scores = tl.where(mask, -1e6, scores)
        
        # Apply softmax
        m_new = tl.maximum(m, tl.max(scores, axis=1))
        p = tl.exp(scores - m_new[:, None])
        l_new = l * tl.exp(m - m_new) + tl.sum(p, axis=1)
        # Compute the normalizer in FP32 before casting probabilities for PV.
        p_casted = p.to(V_tile.dtype)
        O_tile = O_tile * tl.exp(m - m_new)[:, None]
        O_tile = tl.dot(p_casted, V_tile, acc=O_tile, out_dtype=tl.float32, input_precision="ieee")
        m = m_new
        l = l_new

        K_block_ptr = tl.advance(K_block_ptr, (K_TILE_SIZE, 0))
        V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))

    O_tile = O_tile / l[:, None]
    l = m + tl.log(l)
    # Round only the final normalized output to the destination dtype.
    tl.store(O_block_ptr, O_tile.to(O_block_ptr.type.element_ty), boundary_check=(0, 1))
    tl.store(L_block_ptr, l, boundary_check=(0,))


class FlashForwardTriton(torch.autograd.Function):

    Q_tile_size = 32
    K_tile_size = 32


    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):

        batch = Q.shape[0]
        d = Q.shape[-1]
        N_q = Q.shape[-2]
        N_k = K.shape[-2]

        output = torch.zeros_like(Q, device=Q.device, dtype=Q.dtype)
        # Preserve normalization statistics for backward even with FP16/BF16 inputs.
        logsumexp = torch.zeros(batch, N_q, device=Q.device, dtype=torch.float32)

        flash_fwd_kernel[(triton.cdiv(N_q, FlashForwardTriton.Q_tile_size), batch)](
            Q, K, V, output, logsumexp,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            output.stride(0), output.stride(1), output.stride(2),
            logsumexp.stride(0), logsumexp.stride(1),
            N_q, N_k,
            1/d**0.5,
            d,
            FlashForwardTriton.Q_tile_size,
            FlashForwardTriton.K_tile_size,
            is_causal
        )

        ctx.save_for_backward(Q, K, V, output, logsumexp)
        ctx.is_causal = is_causal
        return output

    @staticmethod
    def backward(ctx, grad_output):
        Q, K, V, output, logsumexp = ctx.saved_tensors
        is_causal = ctx.is_causal
        return my_backward(Q, K, V, output, logsumexp, grad_output, is_causal)
