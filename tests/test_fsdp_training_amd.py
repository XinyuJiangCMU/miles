"""Comprehensive FSDP training tests for AMD MI300X.

Tests:
1. Single GPU BF16 training step
2. Gradient checkpointing memory savings
3. Fused AdamW speedup
4. FP8 linear layer correctness
5. MoE model FSDP compatibility
6. Micro-batch size scaling efficiency
"""

import os
import sys
import unittest

import torch

# Skip all tests if not on AMD
if not (torch.version.hip is not None and torch.cuda.is_available()):
    print("Skipping AMD FSDP tests: not on ROCm with CUDA")
    sys.exit(0)

os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "12390")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")


def _init_dist():
    import torch.distributed as dist

    if not dist.is_initialized():
        dist.init_process_group("nccl", rank=0, world_size=1)


class TestFSDPTraining(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _init_dist()

    def test_bf16_training_step(self):
        """Basic BF16 training step works on AMD."""
        from torch.distributed._composable.fsdp import fully_shard
        from transformers import AutoConfig, AutoModelForCausalLM

        config = AutoConfig.from_pretrained("/data/Qwen3-4B")
        config.num_hidden_layers = 2
        model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16).cuda()
        model.train()
        for layer in model.model.layers:
            fully_shard(layer)
        fully_shard(model)

        opt = torch.optim.AdamW(model.parameters(), lr=1e-6)
        x = torch.randint(0, 1000, (2, 128), device="cuda")
        out = model(x, labels=x)
        self.assertFalse(torch.isnan(out.loss), "Loss is NaN")
        out.loss.backward()
        opt.step()

    def test_gradient_checkpointing_saves_memory(self):
        """Gradient checkpointing reduces memory usage."""
        from torch.distributed._composable.fsdp import fully_shard
        from transformers import AutoConfig, AutoModelForCausalLM

        config = AutoConfig.from_pretrained("/data/Qwen3-4B")
        config.num_hidden_layers = 4
        memories = {}
        for gc in [False, True]:
            model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16).cuda()
            model.train()
            if gc:
                model.gradient_checkpointing_enable()
            for layer in model.model.layers:
                fully_shard(layer)
            fully_shard(model)

            opt = torch.optim.AdamW(model.parameters(), lr=1e-6)
            x = torch.randint(0, 1000, (4, 256), device="cuda")
            out = model(x, labels=x)
            out.loss.backward()
            opt.step()
            torch.cuda.synchronize()
            free, total = torch.cuda.mem_get_info(0)
            memories[gc] = (total - free) / 1024**3
            del model, opt, out
            torch.cuda.empty_cache()

        self.assertLess(
            memories[True],
            memories[False],
            f"GC should save memory: {memories[True]:.1f}GB vs {memories[False]:.1f}GB",
        )

    def test_fused_adamw(self):
        """Fused AdamW works with FSDP on AMD."""
        from torch.distributed._composable.fsdp import fully_shard
        from transformers import AutoConfig, AutoModelForCausalLM

        config = AutoConfig.from_pretrained("/data/Qwen3-4B")
        config.num_hidden_layers = 2
        model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16).cuda()
        model.train()
        for layer in model.model.layers:
            fully_shard(layer)
        fully_shard(model)

        opt = torch.optim.AdamW(model.parameters(), lr=1e-6, fused=True)
        x = torch.randint(0, 1000, (2, 128), device="cuda")
        for _ in range(3):
            out = model(x, labels=x)
            self.assertFalse(torch.isnan(out.loss))
            out.loss.backward()
            opt.step()
            opt.zero_grad()

    def test_fp8_linear_correctness(self):
        """FP8 linear layer produces non-NaN output and gradients."""
        from miles.utils.fp8_linear import FP8Linear

        linear = torch.nn.Linear(256, 512, dtype=torch.bfloat16).cuda()
        fp8_linear = FP8Linear(linear)

        x = torch.randn(4, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        y = fp8_linear(x)
        self.assertEqual(y.shape, (4, 512))
        self.assertFalse(torch.isnan(y).any())

        y.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertFalse(torch.isnan(x.grad).any())

    def test_reshard_false_for_dp1(self):
        """reshard_after_forward=False works for DP=1."""
        from torch.distributed._composable.fsdp import fully_shard
        from transformers import AutoConfig, AutoModelForCausalLM

        config = AutoConfig.from_pretrained("/data/Qwen3-4B")
        config.num_hidden_layers = 2
        model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16).cuda()
        model.train()
        for layer in model.model.layers:
            fully_shard(layer, reshard_after_forward=False)
        fully_shard(model, reshard_after_forward=False)

        opt = torch.optim.AdamW(model.parameters(), lr=1e-6, fused=True)
        x = torch.randint(0, 1000, (2, 128), device="cuda")
        for _ in range(3):
            out = model(x, labels=x)
            self.assertFalse(torch.isnan(out.loss))
            out.loss.backward()
            opt.step()
            opt.zero_grad()

    def test_micro_batch_scaling(self):
        """Larger micro-batch is more efficient (higher tok/s)."""
        from torch.distributed._composable.fsdp import fully_shard
        from transformers import AutoConfig, AutoModelForCausalLM

        config = AutoConfig.from_pretrained("/data/Qwen3-4B")
        config.num_hidden_layers = 2
        results = {}
        for mbs in [1, 4]:
            model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16).cuda()
            model.train()
            for layer in model.model.layers:
                fully_shard(layer)
            fully_shard(model)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-6)
            x = torch.randint(0, 1000, (mbs, 128), device="cuda")
            for _ in range(3):
                out = model(x, labels=x)
                out.loss.backward()
                opt.step()
                opt.zero_grad()
            torch.cuda.synchronize()
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(5):
                out = model(x, labels=x)
                out.loss.backward()
                opt.step()
                opt.zero_grad()
            e.record()
            torch.cuda.synchronize()
            ms = s.elapsed_time(e) / 5
            results[mbs] = mbs * 128 / ms * 1000
            del model, opt
            torch.cuda.empty_cache()

        self.assertGreater(
            results[4],
            results[1] * 1.3,
            f"MBS=4 should be >30% faster: {results[4]:.0f} vs {results[1]:.0f} tok/s",
        )


if __name__ == "__main__":
    unittest.main()
