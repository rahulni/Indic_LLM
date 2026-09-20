# References

I checked every arXiv ID, title and author list below against its arXiv abstract page (September 2026). Years are arXiv submission years; I did not check conference venues. Where the check was weaker, a note says so.

## Papers

- **ZeRO:** Rajbhandari, Rasley, Ruwase, He. *ZeRO: Memory Optimizations Toward Training Trillion Parameter Models.* 2019. [arXiv:1910.02054](https://arxiv.org/abs/1910.02054)
  - Figure 1 (7.5B parameters, 64 GPUs, K = 12): 120 / 31.4 / 16.6 / 1.9 GB.
  - 1T parameters on 1024 GPUs: about 16 GB per GPU.
  - Communication: 2Ψ for P_os+g and 3Ψ (1.5×) for P_os+g+p.
- **ZeRO-Offload:** Ren, Rajbhandari, Aminabadi, Ruwase, Yang, Zhang, Li, He. *ZeRO-Offload: Democratizing Billion-Scale Model Training.* 2021. [arXiv:2101.06840](https://arxiv.org/abs/2101.06840)
- **ZeRO-Infinity:** Rajbhandari, Ruwase, Rasley, Smith, He. *ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning.* 2021. [arXiv:2104.07857](https://arxiv.org/abs/2104.07857)
- **ZeRO++:** Wang, Qin, Jacobs, Holmes, Rajbhandari, Ruwase, Yan, Yang, He. *ZeRO++: Extremely Efficient Collective Communication for Giant Model Training.* 2023. [arXiv:2306.10209](https://arxiv.org/abs/2306.10209)
  - hpZ keeps a secondary, node-local copy of the weights, so the backward all-gather stays inside the node.
  - For 100B parameters on 1024 GPUs (16 per node), the paper reports 114× less memory than DP. `zeropp_hpz` reproduces this.
- **PyTorch FSDP:** Zhao, Gu, Varma, Luo, Huang, Xu, Wright, Shojanazeri, Ott, Shleifer, Desmaison, Balioglu, Damania, Nguyen, Chauhan, Hao, Mathews, Li. *PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel.* 2023. [arXiv:2304.11277](https://arxiv.org/abs/2304.11277)
- **Activation memory:** Korthikanti, Casper, Lym, McAfee, Andersch, Shoeybi, Catanzaro. *Reducing Activation Recomputation in Large Transformer Models.* 2022. [arXiv:2205.05198](https://arxiv.org/abs/2205.05198)
  - Per layer: sbh(34 + 5as/h) with no recomputation; 34sbh with selective recomputation (no tensor parallelism); 2sbh with full recomputation.
- **Megatron-LM:** Shoeybi, Patwary, Puri, LeGresley, Casper, Catanzaro. *Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism.* 2019. [arXiv:1909.08053](https://arxiv.org/abs/1909.08053)
- **LLaMA-2:** Touvron et al. *Llama 2: Open Foundation and Fine-Tuned Chat Models.* 2023. [arXiv:2307.09288](https://arxiv.org/abs/2307.09288)
  - 4k context. The 34B and 70B models use grouped-query attention (GQA).
  - Exact parameter counts are derived from the released configs (for example, [70B config.json](https://huggingface.co/NousResearch/Llama-2-70b-hf/raw/main/config.json)). They are checked in `tests/test_zero_theory.py`.
- **Ring all-reduce:** Patarasuk, Yuan. *Bandwidth optimal all-reduce algorithms for clusters of workstations.* J. Parallel Distrib. Comput. 69 (2009) 117–124. [doi:10.1016/j.jpdc.2008.09.002](https://doi.org/10.1016/j.jpdc.2008.09.002)
  - I verified this through search results (ScienceDirect, ACM DL, the author's PDF). The PDF text itself could not be extracted.
- **GPT-2 XL:** Radford et al. 2019, *Language Models are Unsupervised Multitask Learners.* This is an OpenAI report, with no arXiv ID.
  - The "1558M" model: [OpenAI 1.5B release](https://openai.com/index/gpt-2-1-5b-release/) and the [model card](https://huggingface.co/openai-community/gpt2-xl).
  - The count 1,557,611,200 is derived from the architecture.

## Hardware

- **NVIDIA A100** ([datasheet page](https://www.nvidia.com/en-us/data-center/a100/)): 80GB SXM.
  - BF16: 312 TFLOPS dense (624 TFLOPS with sparsity).
  - Memory: 80GB HBM2e at 2,039 GB/s.
  - NVLink: 600 GB/s.
- **NVIDIA H100** ([datasheet page](https://www.nvidia.com/en-us/data-center/h100/)): H100 SXM.
  - BF16: 1,979 TFLOPS *with sparsity*, so about 989 TFLOPS dense.
  - Memory: 80GB at 3.35 TB/s.
  - NVLink: 900 GB/s.
- **NVLink** ([NVIDIA NVLink page](https://www.nvidia.com/en-us/data-center/nvlink/)): the per-GPU figures are bidirectional. `zero_theory` therefore uses 300 GB/s (A100) and 450 GB/s (H100) per direction.
- **DGX A100** ([datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/nvidia-dgx-a100-datasheet.pdf)): 8× single-port ConnectX-6 200 Gb/s HDR InfiniBand, one per GPU, so 25 GB/s per GPU.
- **DGX H100** ([datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/nvidia-dgx-h100-datasheet.pdf)): 4 OSFP ports serving 8× single-port ConnectX-7 400 Gb/s, so 50 GB/s per GPU.
  - I checked the NIC lines through search summaries of the datasheets, not the PDF text.
- **Measured multi-node bus bandwidth:** an 8-node H200 cluster (8× 400 Gb/s per node) averages about 356 GB/s all-reduce bus bandwidth ([Nebius NCCL example](https://docs.nebius.com/slurm-soperator/jobs/examples/nccl-all-reduce)).
  - This supports the optimistic `network="rail"` model (g × NIC per GPU) against the conservative `"per_nic"` default.

## Software documentation

- **DeepSpeed ZeRO tutorial** (stages 1/2/3 and config): <https://www.deepspeed.ai/tutorials/zero/>
- **PyTorch `ZeroRedundancyOptimizer`** ("Distributed Optimizers"): <https://docs.pytorch.org/docs/stable/distributed.optim.html>
  - It shards optimizer states only, "as described by ZeRO".
