# coding=utf-8
# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/opt/modeling_opt.py
# Copyright 2023 The vLLM team.
# Copyright 2022 The Fairseq Authors and The HuggingFace Inc. team. All rights
# reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only OPT model compatible with HuggingFace weights."""
from typing import Iterable, List, Optional, Tuple

import torch
from torch import nn
from transformers import OPTConfig

from vllm.attention import Attention, AttentionMetadata
from vllm.config import CacheConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               QKVParallelLinear,
                                               ReplicatedLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import SamplerOutput
import time

import sys

from vllm.backgroud_logger import logger
import os
from vllm.test_2 import dump_process, save_dump

# logger = init_logger(__name__)

class OPTLearnedPositionalEmbedding(nn.Embedding):

    def __init__(self, num_embeddings: int, embedding_dim: int):
        # OPT is set up so that if padding_idx is specified then offset the
        # embedding ids by 2 and adjust num_embeddings appropriately. Other
        # models don't have this hack
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def forward(self, positions: torch.Tensor):
        return super().forward(positions + self.offset)


class OPTAttention(nn.Module):

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = True,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        tensor_model_parallel_world_size = (
            get_tensor_model_parallel_world_size())
        total_num_heads = num_heads
        assert num_heads % tensor_model_parallel_world_size == 0
        self.num_heads = total_num_heads // tensor_model_parallel_world_size
        self.head_dim = embed_dim // total_num_heads
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            embed_dim,
            self.head_dim,
            total_num_heads,
            bias=bias,
            quant_config=quant_config,
        )
        self.out_proj = RowParallelLinear(
            embed_dim,
            embed_dim,
            bias=bias,
            quant_config=quant_config,
        )
        self.attn = Attention(self.num_heads,
                              self.head_dim,
                              scale=self.scaling,
                              cache_config=cache_config,
                              quant_config=quant_config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        logger.info("[OPTAttention forward] 0")
        qkv, _ = self.qkv_proj(hidden_states)
        logger.info("[OPTAttention forward] 1 : qkv proj")
        q, k, v = qkv.chunk(chunks=3, dim=-1)
        logger.info("[OPTAttention forward] 2 : qkv chunk")
        attn_output = self.attn(q, k, v, kv_cache, attn_metadata)
        logger.info("[OPTAttention forward] 3 : attn")
        output, _ = self.out_proj(attn_output)
        logger.info("[OPTAttention forward] 4 : out proj")
        return output


class OPTDecoderLayer(nn.Module):

    def __init__(
        self,
        config: OPTConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.self_attn = OPTAttention(
            embed_dim=self.embed_dim,
            num_heads=config.num_attention_heads,
            bias=config.enable_bias,
            cache_config=cache_config,
            quant_config=quant_config,
        )
        self.do_layer_norm_before = config.do_layer_norm_before

        self.self_attn_layer_norm = nn.LayerNorm(
            self.embed_dim,
            elementwise_affine=config.layer_norm_elementwise_affine)
        self.fc1 = ColumnParallelLinear(
            self.embed_dim,
            config.ffn_dim,
            bias=config.enable_bias,
            quant_config=quant_config,
        )
        self.activation_fn = get_act_fn(config.activation_function,
                                        quant_config, config.ffn_dim)
        self.fc2 = RowParallelLinear(
            config.ffn_dim,
            self.embed_dim,
            bias=config.enable_bias,
            quant_config=quant_config,
        )
        self.final_layer_norm = nn.LayerNorm(
            self.embed_dim,
            elementwise_affine=config.layer_norm_elementwise_affine)

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        logger.info(f"[OPT Layer] 0 : Start, current hidden_states: {hidden_states.shape}")
        # Self Attention
        residual = hidden_states
        # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
        if self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)
            logger.info(f"[OPT Layer] 1 : Layer norm before attention")
        hidden_states = self.self_attn(hidden_states=hidden_states,
                                       kv_cache=kv_cache,
                                       attn_metadata=attn_metadata)
        logger.info(f"[OPT Layer] 2 : Self attention")
        hidden_states = residual + hidden_states
        logger.info(f"[OPT Layer] 2.1 : Add residual")
        # 350m applies layer norm AFTER attention
        if not self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)
            logger.info(f"[OPT Layer] 3 : Layer norm after attention")

        # Fully Connected
        residual = hidden_states
        # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
        if self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)
            logger.info(f"[OPT Layer] 4 : Layer norm before ffn")
        hidden_states, _ = self.fc1(hidden_states)
        logger.info(f"[OPT Layer] 5 : FC1")
        hidden_states = self.activation_fn(hidden_states)
        logger.info(f"[OPT Layer] 6 : Activation FN")
        hidden_states, _ = self.fc2(hidden_states)
        logger.info(f"[OPT Layer] 7 : FC2")        
        hidden_states = residual + hidden_states
        logger.info(f"[OPT Layer] 7.1 : Add residual")
        # 350m applies layer norm AFTER attention
        if not self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)
            logger.info(f"[OPT Layer] 8 : Layer norm after ffn")
        
        return hidden_states


class OPTDecoder(nn.Module):

    def __init__(
        self,
        config: OPTConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        # Record the overall start time
        logger.info("[OPT Initialize] Step 0: Start initializing OPTDecoder")
        total_start = time.time()

        super().__init__()
        # Step 1: Set basic attributes
        # t0 = time.time()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.vocab_size = config.vocab_size
        # t1 = time.time()
        # print("Step 1: Basic attribute assignment took {:.3f} ms".format((t1 - t0) * 1000))
        logger.info("[OPT Initialize] Step 1: Basic attribute assignment")

        # Step 2: Initialize token embedding and positional embedding
        # t0 = time.time()
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.word_embed_proj_dim,
        )
        logger.info("[OPT Initialize] Step 1.1: Token embedding initialization")
        self.embed_positions = OPTLearnedPositionalEmbedding(
            config.max_position_embeddings, config.hidden_size)
        # t1 = time.time()
        # print("Step 2: Token and positional embedding initialization took {:.3f} ms".format((t1 - t0) * 1000))
        logger.info("[OPT Initialize] Step 2: Token and positional embedding initialization")

        # Step 3: Initialize project_out if necessary
        # t0 = time.time()
        if config.word_embed_proj_dim != config.hidden_size:
            self.project_out = ReplicatedLinear(
                config.hidden_size,
                config.word_embed_proj_dim,
                bias=False,
                quant_config=quant_config
            )
        else:
            self.project_out = None
        # t1 = time.time()
        # print("Step 3: project_out initialization took {:.3f} ms".format((t1 - t0) * 1000))
        logger.info("[OPT Initialize] Step 3: project_out initialization")

        # Step 4: Initialize project_in if necessary
        # t0 = time.time()
        if config.word_embed_proj_dim != config.hidden_size:
            self.project_in = ReplicatedLinear(
                config.word_embed_proj_dim,
                config.hidden_size,
                bias=False,
                quant_config=quant_config
            )
        else:
            self.project_in = None
        # t1 = time.time()
        # print("Step 4: project_in initialization took {:.3f} ms".format((t1 - t0) * 1000))
        logger.info("[OPT Initialize] Step 4: project_in initialization")

        # Step 5: Initialize final_layer_norm if required
        # t0 = time.time()
        if config.do_layer_norm_before and not config._remove_final_layer_norm:
            self.final_layer_norm = nn.LayerNorm(
                config.hidden_size,
                elementwise_affine=config.layer_norm_elementwise_affine
            )
        else:
            self.final_layer_norm = None
        # t1 = time.time()
        # print("Step 5: final_layer_norm initialization took {:.3f} ms".format((t1 - t0) * 1000))
        logger.info("[OPT Initialize] Step 5: final_layer_norm initialization")

        self.layers = nn.ModuleList([
            OPTDecoderLayer(config, cache_config, quant_config)
            for _ in range(config.num_hidden_layers)
        ])

        # def init_layers():
        #     from vllm.attention.backends.xformers import XFormersBackend
        #     import xformers
        #     self.layers = nn.ModuleList([
        #     OPTDecoderLayer(config, cache_config, quant_config)
        #     for _ in range(config.num_hidden_layers)
        #     ])
        
        # save_dump(init_layers)

        
        # t1 = time.time()
        # print("Step 6: Decoder layers initialization took {:.3f} ms".format((t1 - t0) * 1000))
        logger.info("[OPT Initialize] Step 6: Decoder layers initialization")

        total_end = time.time()
        # print("Total initialization time: {:.3f} ms".format((total_end - total_start) * 1000))

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        logger.info(f"[OPT Decoder forward] 0 : Start input token size {input_ids.shape}, kv cache: {len(kv_caches)}")
        logger.info(f"[OPT Decoder forward] 1 : Begin forward")
        inputs_embeds = self.embed_tokens(input_ids)
        logger.info(f"[OPT Decoder forward] 2 : Token embedding")
        pos_embeds = self.embed_positions(positions)
        logger.info(f"[OPT Decoder forward] 3 : Positional embedding")
        if self.project_in is not None:
            inputs_embeds, _ = self.project_in(inputs_embeds)
            logger.info(f"[OPT Decoder forward] 4 : Project input")
        hidden_states = inputs_embeds + pos_embeds
        logger.info(f"[OPT Decoder forward] 5 : Add token and positional embeddings")


        for i in range(len(self.layers)):
            attn_metadata.layer_id=i    # [ReuseStore]: 每次进入新的layer，更新layer_id
            layer = self.layers[i]
            hidden_states = layer(hidden_states, kv_caches[i], attn_metadata)
            logger.info(f"[OPT Decoder forward] 6.{i} : Decoder layer {i}")

        if self.final_layer_norm is not None:
            hidden_states = self.final_layer_norm(hidden_states)
            logger.info(f"[OPT Decoder forward] 7 : Final layer norm")
        if self.project_out is not None:
            hidden_states, _ = self.project_out(hidden_states)
            logger.info(f"[OPT Decoder forward] 8 : Project output")
        logger.info(f"[OPT Decoder forward] 9 : End forward")
        return hidden_states


class OPTModel(nn.Module):

    def __init__(
        self,
        config: OPTConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.decoder = OPTDecoder(config, cache_config, quant_config)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        return self.decoder(input_ids, positions, kv_caches, attn_metadata)


class OPTForCausalLM(nn.Module):

    def __init__(
        self,
        config,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.model = OPTModel(config, cache_config, quant_config)
        self.lm_head_weight = self.model.decoder.embed_tokens.weight
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.sampler = Sampler()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, kv_caches,
                                   attn_metadata)
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        logits = self.logits_processor(self.lm_head_weight, hidden_states,
                                       sampling_metadata)
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        for name, loaded_weight in weights:
            if "lm_head.weight" in name:
                continue
            if name.startswith("decoder."):
                name = "model." + name

            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
