# Gradient Checkpointing：激活内存与重算成本

## (a) Nested checkpointing

What checkpointing strategy minimizes peak activation memory, ignoring the compute cost?
Describe how you would arrange the checkpoint calls (a code sketch is fine), and give the
asymptotic peak activation memory and compute of your strategy as a function of 𝑁. Assume
the residuals saved by a single block dominate any per-checkpoint bookkeeping.

1. checkpoint strategy: only keep the residuals of the outlast final block and wrap prefix of the first N-1 blocks into one checkpoint. In this checkpoint, use the same method, keep the last residuals of last block and wrap prefix blocks into inner checkpoint. 
2. the peak memory should be $O(1)$, which is memory for residuals of one block and is not related to N. When current block backward finishes, the residuals are released and it continue to the previous checkpoint. 
3. The compute complexity is sum N to 1 total forward pass:

   $$
   \sum_{i=1}^{N}i = \frac{N(N+1)}{2} = O(N^2)
   $$

### Code sketch

```python
def recursive_checkpoints(blocks: list[Block], x: tensor):
    if len(blocks)==1:
        block = blocks[0]
        x = block(x)
    else:
        x = checkpoint(recursive_checkpoints, blocks[0:-1], x, use_reentrant=False)
        x = blocks[-1](x)
    return x
```

## (b) Non-nested checkpointing

Nested checkpoint calls are not allowed. The configuration is:

- N = 32 total number of blocks
- k: the number of blocks of each region
- C = 80MB, memory cost of one checkpoint input
- R = 3651.31 MB memory cost of residuals of one block

checkpoint region number: ceil(N/k)

long term checkpoint memory: ceil(N/k)*C

ongoing calculation memory of current region k*R

sum up:
M(k) = ceil(N/k)*C + k*R
k increase, long term memory cost ceil(N/k)*C decrease but current residuals cost increase. 

| k | checkpoint memory | recomputation memory | total |
|---|---|---|---|
| 1 | 32\*80 = 2560MB | 1\*3651.31MB | 6211.31MB |
| 2 | 16\*80 = 1280MB | 2\*3651.31MB | 8582.62MB |
| 4 | 8\*80 = 640MB | 4\*3651.31 = 14605.24MB | 15245.24MB |

Since R is 45x of C, N\*C is about 32x of c, M(k) = 32C/k + 45C\*k. k=1 will incur smallest M(k).
