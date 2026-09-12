import os

import pytest
import torch
from torch.nn.attention.flex_attention import BlockMask, create_block_mask

from wam_h3.refined.backbone import _compile_attention, _fixed_tile_attention


@pytest.fixture(autouse=True)
def isolated_dynamo():
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


def inputs(text, frames, device="cpu"):
    frame_tokens = 4
    n = text + frames * frame_tokens

    def allowed(b, head, q, k):
        return (k < text) | ((q >= text) & (k >= text) & ((k - text) // frame_tokens <= (q - text) // frame_tokens))

    raw = create_block_mask(allowed, 1, 1, n, n, device=device)
    blocks = raw.to_dense().bool()
    counts = blocks.sum(-1).to(torch.int32)
    indices = blocks.to(torch.int32).argsort(dim=-1, descending=True, stable=True).to(torch.int32)
    mask = BlockMask.from_kv_blocks(counts, indices, BLOCK_SIZE=raw.BLOCK_SIZE,
                                   mask_mod=allowed, seq_lengths=(n, n))
    q, k, v = [torch.randn(1, n, 2, 16, device=device).transpose(1, 2) for _ in range(3)]
    return q, k, v, mask


def test_more_than_eight_signatures_remain_compiled():
    compiled_graphs = []
    executions = []

    def backend(graph, example_inputs):
        compiled_graphs.append(graph)
        nodes = [node for node in graph.graph.nodes if node.target == torch.ops.higher_order.flex_attention]
        assert len(nodes) == 1
        options = nodes[0].args[6]
        assert options["BLOCK_M"] == options["BLOCK_N"] == 64
        assert options["BACKEND"] == "TRITON"

        def execute(*args):
            executions.append(1)
            return graph.forward(*args)

        return execute

    attend = _compile_attention(backend=backend)
    signatures = [(67 + i * 7, frames) for i, frames in enumerate((1, 4, 8, 16, 24, 32, 48, 64, 80, 96, 128, 152))]
    with torch.no_grad():
        for text, frames in signatures:
            q, k, v, mask = inputs(text, frames)
            output = attend(q, k, v, mask)
            positions = torch.arange(q.shape[-2])
            allowed = mask.mask_mod(0, 0, positions[:, None], positions[None, :])
            expected = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=allowed)
            torch.testing.assert_close(output, expected, atol=1e-6, rtol=1e-5)
    assert len(executions) == len(signatures)
    assert len(compiled_graphs) <= 8


def test_cache_exhaustion_fails_closed():
    executions = []

    def backend(graph, example_inputs):
        def execute(*args):
            executions.append(1)
            return graph.forward(*args)
        return execute

    attend = _compile_attention(backend=backend)
    args = inputs(67, 4)
    with torch._dynamo.config.patch(recompile_limit=1), torch.no_grad():
        attend(*args)
        changed = tuple(t.double() for t in args[:3]) + args[3:]
        with pytest.raises(torch._dynamo.exc.FailOnRecompileLimitHit, match="fullgraph=True"):
            attend(*changed)
        with pytest.raises(torch._dynamo.exc.FailOnRecompileLimitHit):
            attend(*changed)
    assert len(executions) == 1


@pytest.mark.skipif(os.environ.get("REFINED_ATTENTION_CUDA_TEST") != "1", reason="opt-in tiny CUDA kernel test")
def test_cuda_signatures_and_exact_prefix():
    attend = _compile_attention()
    device = "cuda:0"
    with torch.no_grad():
        for i, frames in enumerate((1, 4, 8, 16, 24, 32, 48, 64, 80, 96, 128, 152)):
            q, k, v, mask = inputs(67 + i * 7, frames, device)
            result = attend(q.bfloat16(), k.bfloat16(), v.bfloat16(), mask)
            assert torch.isfinite(result).all()
        q, k, v, mask = inputs(67, 152, device)
        q, k, v = (x.bfloat16() for x in (q, k, v))
        full = attend(q, k, v, mask)
        for frames in (1, 4, 16, 64, 128):
            prefix_mask = inputs(67, frames, device)[3]
            n = 67 + 4 * frames
            prefix = attend(q[:, :, :n], k[:, :, :n], v[:, :, :n], prefix_mask)
            assert torch.equal(prefix, full[:, :, :n])


def test_eager_execution_is_rejected():
    with pytest.raises(RuntimeError, match="requires compiled fixed-tile"):
        _fixed_tile_attention(*inputs(67, 4))
    with torch.compiler.set_stance("force_eager"):
        with pytest.raises(RuntimeError, match="requires compiled fixed-tile"):
            _compile_attention()(*inputs(67, 4))
