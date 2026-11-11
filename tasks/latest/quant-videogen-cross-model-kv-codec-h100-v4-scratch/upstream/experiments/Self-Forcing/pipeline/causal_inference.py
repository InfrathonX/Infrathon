from typing import List, Optional
import torch
import os
import numpy as np
from tqdm import tqdm
from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper

from demo_utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller, move_model_to_device_with_memory_preservation

from types import SimpleNamespace
from termcolor import cprint

from quant_videogen.compress import get_quantize_fn, compress_kv_cache
from quant_videogen.kv_cache import ChunkedKVCache, offload_kv_cache_layer, onload_kv_cache_layer

class CausalInferencePipeline(torch.nn.Module):
    def __init__(
            self,
            args,
            device,
            generator=None,
            text_encoder=None,
            vae=None
    ):
        # Environment variable for dumping KV cache
        self.dump_kv_level = int(os.getenv("DUMP_KV_LEVEL", "0"))  # 0: off, 1: final, 2: every block
        super().__init__()
        # Step 1: Initialize all models
        self.generator = WanDiffusionWrapper(
            **getattr(args, "model_kwargs", {}), is_causal=True) if generator is None else generator
        self.text_encoder = WanTextEncoder() if text_encoder is None else text_encoder
        self.vae = WanVAEWrapper() if vae is None else vae

        # Step 2: Initialize all causal hyperparmeters
        self.scheduler = self.generator.get_scheduler()
        self.denoising_step_list = torch.tensor(
            args.denoising_step_list, dtype=torch.long)
        if args.warp_denoising_step:
            timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
            self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

        self.num_transformer_blocks = 30
        self.frame_seq_length = 1560

        self.kv_cache1 = None
        self.args = args
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.independent_first_frame = args.independent_first_frame
        self.local_attn_size = self.generator.model.local_attn_size

        print(f"KV inference with {self.num_frame_per_block} frames per block")

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block
            
        # Quant Config using SimpleNamespace
        self.quant_config = SimpleNamespace(**args.quant_config)

        self.generator.model.kv_cache_cpu_offload = getattr(self.quant_config, "kv_cache_cpu_offload", False)


    def quantize_kv_cache(self, tokens_to_quantize_start: int, tokens_to_quantize_end: int, max_tokens_to_quantize: int):
        """
        Quantize a range of the KV cache.  Indices are in token space and
        must be frame-aligned.
        """
        # Check the inputs
        assert tokens_to_quantize_start <= max_tokens_to_quantize and tokens_to_quantize_end <= max_tokens_to_quantize
        cprint(
            f"Quantizing {tokens_to_quantize_end - tokens_to_quantize_start} tokens, "
            f"from {tokens_to_quantize_start} to {tokens_to_quantize_end} tokens. "
            f"The max tokens to quantize is {max_tokens_to_quantize}",
            "light_cyan",
        )

        # Record the time using torch.cuda.Event
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        start_time.record()

        # Do nothing if quantization type is none
        if self.quant_config.quant_type == "none":
            self._print_memory_usage(self.kv_cache1)
            return

        cprint(f"Quantizing kv cache with config:\n{self.quant_config}", "light_blue")

        do_offload = getattr(self.quant_config, "kv_cache_cpu_offload", False)
        cuda_device = torch.device("cuda")

        with torch.no_grad():
            quantize_fn = get_quantize_fn(self.quant_config.quant_type, self.quant_config)

            for layer_idx, layer in enumerate(self.kv_cache1):
                if do_offload:
                    onload_kv_cache_layer(layer, cuda_device)

                # Read the span to quantize — always full precision [B, S, H, D]
                k = layer["k"].read(tokens_to_quantize_start, tokens_to_quantize_end)
                v = layer["v"].read(tokens_to_quantize_start, tokens_to_quantize_end)

                # compress_kv_cache expects [B, H, S, D]
                k = k.permute(0, 2, 1, 3).contiguous()
                v = v.permute(0, 2, 1, 3).contiguous()

                # Codec-internal centroid initialization must not perturb the
                # diffusion RNG stream used after this cache mutation.
                with torch.random.fork_rng(devices=[k.device.index]):
                    k_quant, v_quant = compress_kv_cache(
                        k, v, self.quant_config.quant_type, self.quant_config, quantize_fn
                    )

                # Candidate payloads are opaque. Store them byte-for-byte: the
                # codec, not the model integration, owns all decode metadata.
                layer["k"].store_quantized(
                    tokens_to_quantize_start, tokens_to_quantize_end, k_quant
                )
                layer["v"].store_quantized(
                    tokens_to_quantize_start, tokens_to_quantize_end, v_quant
                )

                if do_offload:
                    offload_kv_cache_layer(layer)

        end_time.record()
        torch.cuda.synchronize()
        duration = start_time.elapsed_time(end_time)
        cprint(f"Quantization KV Cache Time: {(duration / 1000):.2f} s", "light_cyan")

        self._print_memory_usage(self.kv_cache1)

    def _print_memory_usage(self, kv_cache_list: list[dict]):
        """Report allocator state without moving or rewriting opaque payloads."""
        allocated = torch.cuda.memory_allocated() / 1024**2
        peak = torch.cuda.max_memory_allocated() / 1024**2
        cprint(
            f"CUDA Memory: allocated={allocated:.2f} MB | peak={peak:.2f} MB",
            "light_blue",
        )

    def inference(
        self,
        noise: torch.Tensor,
        text_prompts: List[str],
        initial_latent: Optional[torch.Tensor] = None,
        return_latents: bool = False,
        profile: bool = False,
        low_memory: bool = False,
    ) -> torch.Tensor:
        """
        Perform inference on the given noise and text prompts.
        Inputs:
            noise (torch.Tensor): The input noise tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
            text_prompts (List[str]): The list of text prompts.
            initial_latent (torch.Tensor): The initial latent tensor of shape
                (batch_size, num_input_frames, num_channels, height, width).
                If num_input_frames is 1, perform image to video.
                If num_input_frames is greater than 1, perform video extension.
            return_latents (bool): Whether to return the latents.
        Outputs:
            video (torch.Tensor): The generated video tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
                It is normalized to be in the range [0, 1].
        """
        batch_size, num_frames, num_channels, height, width = noise.shape
        if not self.independent_first_frame or (self.independent_first_frame and initial_latent is not None):
            # If the first frame is independent and the first frame is provided, then the number of frames in the
            # noise should still be a multiple of num_frame_per_block
            assert num_frames % self.num_frame_per_block == 0
            num_blocks = num_frames // self.num_frame_per_block
        else:
            # Using a [1, 4, 4, 4, 4, 4, ...] model to generate a video without image conditioning
            assert (num_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (num_frames - 1) // self.num_frame_per_block
        num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
        num_output_frames = num_frames + num_input_frames  # add the initial latent frames
        conditional_dict = self.text_encoder(
            text_prompts=text_prompts
        )

        if low_memory:
            gpu_memory_preservation = get_cuda_free_memory_gb(gpu) + 5
            move_model_to_device_with_memory_preservation(self.text_encoder, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)

        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype
        )

        # Set up profiling if requested
        if profile:
            init_start = torch.cuda.Event(enable_timing=True)
            init_end = torch.cuda.Event(enable_timing=True)
            diffusion_start = torch.cuda.Event(enable_timing=True)
            diffusion_end = torch.cuda.Event(enable_timing=True)
            vae_start = torch.cuda.Event(enable_timing=True)
            vae_end = torch.cuda.Event(enable_timing=True)
            block_times = []
            block_start = torch.cuda.Event(enable_timing=True)
            block_end = torch.cuda.Event(enable_timing=True)
            init_start.record()

        # Step 1: Initialize KV cache to all zeros (size depends on target length when using global attention)
        if self.kv_cache1 is None:
            # When local_attn_size == -1 we need a cache large enough for the entire sequence
            if self.local_attn_size == -1:
                # reserve all output frames (input + to-generate) plus 1 extra block for safety
                target_frames = num_output_frames + 1
            else:
                target_frames = self.local_attn_size

            self._initialize_kv_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device,
                target_frames=target_frames
            )
            self._initialize_crossattn_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device
            )
        else:
            # reset cross attn cache
            for block_index in range(self.num_transformer_blocks):
                self.crossattn_cache[block_index]["is_init"] = False
            # reset kv cache
            for layer in self.kv_cache1:
                layer["global_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
                layer["local_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)

        # Step 2: Cache context feature
        current_start_frame = 0
        if initial_latent is not None:
            timestep = torch.ones([batch_size, 1], device=noise.device, dtype=torch.int64) * 0
            if self.independent_first_frame:
                # Assume num_input_frames is 1 + self.num_frame_per_block * num_input_blocks
                assert (num_input_frames - 1) % self.num_frame_per_block == 0
                num_input_blocks = (num_input_frames - 1) // self.num_frame_per_block
                output[:, :1] = initial_latent[:, :1]
                self.generator(
                    noisy_image_or_video=initial_latent[:, :1],
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                )
                current_start_frame += 1
            else:
                # Assume num_input_frames is self.num_frame_per_block * num_input_blocks
                assert num_input_frames % self.num_frame_per_block == 0
                num_input_blocks = num_input_frames // self.num_frame_per_block

            for _ in range(num_input_blocks):
                current_ref_latents = \
                    initial_latent[:, current_start_frame:current_start_frame + self.num_frame_per_block]
                output[:, current_start_frame:current_start_frame + self.num_frame_per_block] = current_ref_latents
                self.generator(
                    noisy_image_or_video=current_ref_latents,
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                )
                current_start_frame += self.num_frame_per_block

        if profile:
            init_end.record()
            torch.cuda.synchronize()
            diffusion_start.record()

        # Step 3: Temporal denoising loop
        all_num_frames = [self.num_frame_per_block] * num_blocks
        if self.independent_first_frame and initial_latent is None:
            all_num_frames = [1] + all_num_frames
        for chunk_index, current_num_frames in enumerate(tqdm(all_num_frames)):
            if profile:
                block_start.record()

            noisy_input = noise[
                :, current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames]
            
            #########################################################
            # Possiply quantize the KV cache here
            #########################################################
            # When generating the first 8 chunks, we do not quantize them.

            QUANT_FACTOR = 8
            if chunk_index < QUANT_FACTOR or chunk_index % QUANT_FACTOR != 0:
                pass
            else:
                max_tokens_to_quantize = self.kv_cache1[0]["local_end_index"].item()
                cprint(f"\nAt Chunk {chunk_index}, max tokens to quantize: {max_tokens_to_quantize} tokens", "light_cyan")            
                
                # assert tokens_to_quantize == np.sum(all_num_frames[:chunk_index]) * self.frame_seq_length
                
                # Only quantize the previous chunks
                tokens_to_quantize_start = int(np.sum(all_num_frames[:chunk_index - QUANT_FACTOR]) * self.frame_seq_length)
                tokens_to_quantize_end = int(np.sum(all_num_frames[:chunk_index]) * self.frame_seq_length)
            
                self.quantize_kv_cache(tokens_to_quantize_start, tokens_to_quantize_end, max_tokens_to_quantize)
            
            # Step 3.1: Spatial denoising loop
            for index, current_timestep in enumerate(self.denoising_step_list):
                # set current timestep
                timestep = torch.ones(
                    [batch_size, current_num_frames],
                    device=noise.device,
                    dtype=torch.int64) * current_timestep

                if index < len(self.denoising_step_list) - 1:
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length
                    )
                    next_timestep = self.denoising_step_list[index + 1]
                    noisy_input = self.scheduler.add_noise(
                        denoised_pred.flatten(0, 1),
                        torch.randn_like(denoised_pred.flatten(0, 1)),
                        next_timestep * torch.ones(
                            [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
                    ).unflatten(0, denoised_pred.shape[:2])
                else:
                    # for getting real output
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length
                    )

            # Step 3.2: record the model's output
            output[:, current_start_frame:current_start_frame + current_num_frames] = denoised_pred

            # Step 3.3: rerun with timestep zero to update KV cache using clean context
            context_timestep = torch.ones_like(timestep) * self.args.context_noise
            self.generator(
                noisy_image_or_video=denoised_pred,
                conditional_dict=conditional_dict,
                timestep=context_timestep,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=current_start_frame * self.frame_seq_length,
            )

            if profile:
                block_end.record()
                torch.cuda.synchronize()
                block_time = block_start.elapsed_time(block_end)
                block_times.append(block_time)

            # Step 3.4: update the start and end frame indices
            current_start_frame += current_num_frames

        if profile:
            # End diffusion timing and synchronize CUDA
            diffusion_end.record()
            torch.cuda.synchronize()
            diffusion_time = diffusion_start.elapsed_time(diffusion_end)
            init_time = init_start.elapsed_time(init_end)
            vae_start.record()

        # Step 4: Decode the output
        video = self.vae.decode_to_pixel(output, use_cache=False)
        
        # video = (video * 0.5 + 0.5).clamp(0, 1)
        video.mul_(0.5).add_(0.5).clamp_(0, 1)

        if profile:
            # End VAE timing and synchronize CUDA
            vae_end.record()
            torch.cuda.synchronize()
            vae_time = vae_start.elapsed_time(vae_end)
            total_time = init_time + diffusion_time + vae_time
            self.last_profile = {
                "initialization_cache_ms": float(init_time),
                "diffusion_generation_ms": float(diffusion_time),
                "vae_decode_ms": float(vae_time),
                "pipeline_total_ms": float(total_time),
                "block_ms": [float(value) for value in block_times],
                "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(gpu)),
            }

            print("Profiling results:")
            print(f"  - Initialization/caching time: {init_time:.2f} ms ({100 * init_time / total_time:.2f}%)")
            print(f"  - Diffusion generation time: {diffusion_time:.2f} ms ({100 * diffusion_time / total_time:.2f}%)")
            for i, block_time in enumerate(block_times):
                print(f"    - Block {i} generation time: {block_time:.2f} ms ({100 * block_time / diffusion_time:.2f}% of diffusion)")
            print(f"  - VAE decoding time: {vae_time:.2f} ms ({100 * vae_time / total_time:.2f}%)")
            print(f"  - Total time: {total_time:.2f} ms")

        # ----------------------------------------------------------
        # Optional: dump KV / CrossAttn cache for analysis
        # ----------------------------------------------------------
        if self.dump_kv_level >= 1:
            # Only rank-0 process dumps to avoid duplicates in DDP
            if (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0:
                dump_dir = os.path.join(os.getenv("KV_DUMP_DIR", "kv_dumps"))
                os.makedirs(dump_dir, exist_ok=True)
                # Compose filename with prompt hash & frames
                filename = f"kv_cache_frames{num_output_frames}.pt"
                torch.save({
                    "kv_cache": self.kv_cache1,
                    "crossattn_cache": getattr(self, "crossattn_cache", None),
                }, os.path.join(dump_dir, filename))
                print(f"KV cache dumped to {os.path.join(dump_dir, filename)}")

        if return_latents:
            return video, output
        else:
            return video

    def _initialize_kv_cache(self, batch_size, dtype, device, target_frames: int | None = None):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        kv_cache1 = []
        if self.local_attn_size != -1:
            kv_cache_size = self.local_attn_size * self.frame_seq_length
        else:
            # allocate enough tokens for entire sequence when global attention
            total_frames = target_frames if target_frames is not None else 21
            kv_cache_size = total_frames * self.frame_seq_length

        max_num_chunks = kv_cache_size // self.frame_seq_length

        for _ in range(self.num_transformer_blocks):
            kv_cache1.append({
                "k": ChunkedKVCache(batch_size, self.frame_seq_length, 12, 128, max_num_chunks, dtype, device, layout="BSHD"),
                "v": ChunkedKVCache(batch_size, self.frame_seq_length, 12, 128, max_num_chunks, dtype, device, layout="BSHD"),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
            })

        self.kv_cache1 = kv_cache1  # always store the clean cache

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        """
        crossattn_cache = []

        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append({
                "k": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "is_init": False
            })
        self.crossattn_cache = crossattn_cache
