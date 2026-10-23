import torch
from einops import rearrange, einsum

class FlashForwardFunc(torch.autograd.Function):

    B_q = 32
    B_k = 32

    @staticmethod
    def forward(ctx, Q, K, V, is_causal = False):
        batch = Q.shape[0]

        d = Q.shape[-1]
        N_q = Q.shape[-2]

        N_k = K.shape[-2]
        T_q = N_q // FlashForwardFunc.B_q
        T_k = N_k // FlashForwardFunc.B_k

        # split Q into blocks of size B_q
        Q = rearrange(Q, '... (t b) d -> ... t b d', b=FlashForwardFunc.B_q)
        # split K into blocks of size B_k
        K = rearrange(K, '... (t b) d -> ... t b d', b=FlashForwardFunc.B_k)
        # split V into blocks of size B_k
        V = rearrange(V, '... (t b) d -> ... t b d', b=FlashForwardFunc.B_k)

        # initialize output tensor
        output = torch.zeros(batch, T_q, FlashForwardFunc.B_q, d, device=Q.device, dtype=Q.dtype)
        logsumexp = torch.zeros(batch, T_q, FlashForwardFunc.B_q, device=Q.device, dtype=Q.dtype)
        Q_causal_row_index = torch.arange(FlashForwardFunc.B_q, device=Q.device)
        K_causal_col_index = torch.arange(FlashForwardFunc.B_k, device=Q.device)

        # outer loop over Q blocks
        for i in range(T_q):
            Q_i = Q[:, i]
            # initialize output block
            O_i = torch.zeros_like(Q_i)
            l_i = torch.zeros(batch, FlashForwardFunc.B_q, device=Q.device, dtype=Q.dtype)
            m_i = torch.full((batch, FlashForwardFunc.B_q), float('-inf'), device=Q.device, dtype=Q.dtype)
            # process each Q block with all K blocks and V blocks
            for j in range(T_k):
                K_j = K[:, j]
                V_j = V[:, j]
                # compute tile of pre-softmax attention scores
                S_ij = einsum(Q_i, K_j, '... i d, ... j d -> ... i j').divide(d**0.5)
                # apply causal masking if needed
                if is_causal:
                    mask = (Q_causal_row_index[:, None] + i * FlashForwardFunc.B_q) < (K_causal_col_index[None, :] + j * FlashForwardFunc.B_k)
                    S_ij = S_ij.masked_fill(mask, -1e6)
                # softmax
                m_ij = torch.max(m_i, torch.max(S_ij, dim=-1).values)
                P_ij = torch.exp(S_ij - m_ij.unsqueeze(-1))
                l_ij = torch.exp(m_i - m_ij)*l_i+torch.sum(P_ij, dim=-1)
                O_i = torch.exp(m_i - m_ij).unsqueeze(-1)*O_i + torch.matmul(P_ij, V_j)
                m_i = m_ij
                l_i = l_ij
            O_i = O_i.divide(l_i.unsqueeze(-1))
            output[:, i] = O_i
            logsumexp[:, i] = m_i + torch.log(l_i)

        Q = rearrange(Q, '... t b d -> ... (t b) d')
        K = rearrange(K, '... t b d -> ... (t b) d')
        V = rearrange(V, '... t b d -> ... (t b) d')
        logsumexp = rearrange(logsumexp, '... t b -> ... (t b)')
        output = rearrange(output, '... t b d -> ... (t b) d')
        ctx.save_for_backward(Q, K, V, output, logsumexp)
        ctx.is_causal = is_causal

        return output

    @staticmethod
    def backward(ctx, grad_output):
        Q, K, V, output, logsumexp = ctx.saved_tensors
        is_causal = ctx.is_causal
        return my_backward(Q, K, V, output, logsumexp, grad_output, is_causal)

@torch.compile
def my_backward(Q, K, V, output, logsumexp, grad_output, is_causal):
    d = Q.shape[-1]
    D = torch.mul(output, grad_output).sum(dim=-1)
    S = einsum(Q, K, '... i d, ... j d -> ... i j').divide(d**0.5)
    # apply causal masking if needed
    if is_causal:
        mask = torch.triu(torch.ones(S.shape[-2], S.shape[-1], device=S.device, dtype=torch.bool), diagonal=1)
        S = S.masked_fill(mask, -1e6)
    P = torch.exp(S - logsumexp.unsqueeze(-1))
    grad_V = einsum(P, grad_output, '... n_q n_k, ... n_q d -> ... n_k d')
    grad_P = einsum(grad_output, V, '... n_q d, ... n_k d -> ... n_q n_k')
    grad_S = P * (grad_P - D.unsqueeze(-1))
    grad_Q = einsum(grad_S, K, '... n_q n_k, ... n_k d -> ... n_q d').divide(d**0.5)
    grad_K = einsum(grad_S, Q, '... n_q n_k, ... n_q d -> ... n_k d').divide(d**0.5)

    return grad_Q, grad_K, grad_V, None
