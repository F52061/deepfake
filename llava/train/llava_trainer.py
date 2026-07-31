import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Sampler

from transformers import Trainer
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    has_length,
    ALL_LAYERNORM_LAYERS,
    logger,
)
from typing import List, Optional


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, 'no ignore status')
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # all samples are in the same modality
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        else:
            indices = get_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)


class LLaVATrainer(Trainer):

    # --- [新增] 分类损失权重，由外部设置 ---
    deepfake_cls_loss_weight: float = 0.0

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Override compute_loss: 组合 LLM language modeling loss + deepfake classification loss.
        如果 batch 中没有 deepfake_labels 或权重为 0，直接返回原有 lm_loss。
        """
        # 提取分类标签（不传入 model.forward，避免 LlamaForCausalLM 不认识该参数）
        deepfake_labels = inputs.pop("deepfake_labels", None)

        # 调用原始 forward（标准 LLM loss）
        outputs = model(**inputs)
        lm_loss = outputs.loss
        # [DEBUG] 打印每一步的真实 lm_loss
        import torch as _t
        if lm_loss is not None:
            print(f'[DBG-LOSS] lm_loss.item()={lm_loss.item():.6f} requires_grad={lm_loss.requires_grad}', flush=True)
        else:
            print('[DBG-LOSS] lm_loss is None', flush=True)

        # [新增] 如果有分类标签且权重 > 0，叠加分类损失
        cls_loss = torch.tensor(0.0, device=lm_loss.device)
        if (
            deepfake_labels is not None
            and self.deepfake_cls_loss_weight > 0
            and hasattr(model, '_deepfake_logits_for_loss')
            and model._deepfake_logits_for_loss is not None
        ):
            logits = model._deepfake_logits_for_loss  # [B, 2]
            cls_loss = F.cross_entropy(logits, deepfake_labels)
            lm_loss = lm_loss + self.deepfake_cls_loss_weight * cls_loss
            print(f'[COMPUTE_LOSS] lm_loss={lm_loss.item():.4f}, cls_loss={cls_loss.item():.4f}, labels={deepfake_labels.tolist()}')

        if return_outputs:
            return (lm_loss, outputs)
        return lm_loss

    def _get_real_model(self, model):
        """DataParallel 下取底层模型"""
        return model.module if hasattr(model, 'module') else model

    def _get_cls_loss(self, model):
        """计算分类 loss，返回 (cls_loss_tensor, cls_loss_scalar)"""
        if not (hasattr(self, '_classification_loader') and self.deepfake_cls_loss_weight > 0):
            print('[CLS_LOSS] skip: no _classification_loader or weight=0')
            return None, 0.0

        try:
            cls_batch = next(self._classification_iter)
        except StopIteration:
            self._classification_iter = iter(self._classification_loader)
            cls_batch = next(self._classification_iter)

        cls_labels = cls_batch['deepfake_labels'].to(model.device)
        cls_images = cls_batch.get('images', None)

        if cls_images is None:
            print('[CLS_LOSS] skip: cls_images is None')
            return None, 0.0

        real_model = self._get_real_model(model)
        clip_processor = getattr(real_model, 'processors', {}).get('clip_processor', None) if hasattr(real_model, 'processors') else None
        vision_tower = real_model.get_vision_tower() if hasattr(real_model, 'get_vision_tower') else None
        if clip_processor is None or vision_tower is None:
            print(f'[CLS_LOSS] skip: clip_processor={clip_processor is not None}, vision_tower={vision_tower is not None}')
            return None, 0.0

        from llava.mm_utils import process_images
        clip_tensor = process_images(cls_images, clip_processor, real_model.config)
        if isinstance(clip_tensor, list):
            clip_tensor = [t.to(model.device, dtype=torch.float16) for t in clip_tensor]
        else:
            clip_tensor = clip_tensor.to(model.device, dtype=torch.float16)

        with torch.no_grad():
            mm_dtype = real_model.get_model().mm_projector[0].weight.dtype
            v_features, _ = real_model.encode_images_multimodal_features(clip_tensor.to(mm_dtype))
            v_image_cls = v_features[:, 0, :].unsqueeze(1)
            v_image_patches = v_features[:, 1:, :]
            clip_vision_features = torch.cat([v_image_cls, v_image_patches], dim=1)

        # ClassificationDataset 的 __getitem__ 已返回 image_tensor (tensor)
        # DataCollator 把它们放在 batch['deepfake_inputs'] 里
        deepfake_inputs_from_batch = cls_batch.get('deepfake_inputs', None)
        if deepfake_inputs_from_batch is not None:
            # deepfake_inputs_from_batch 是 list of [tensor]，需 stack
            tensors = []
            for item in deepfake_inputs_from_batch:
                if isinstance(item, list):
                    tensors.append(item[0])
                else:
                    tensors.append(item)
            deepfake_pixel = torch.stack(tensors, dim=0).to(model.device)
        else:
            deepfake_pixel = clip_tensor

        target_dtype = real_model.deepfake_projector[0].weight.dtype
        deepfake_dict = {
            "images": deepfake_pixel.to(target_dtype),
            "clip_vision_features": clip_vision_features.to(target_dtype),
            "use_cached_clip_text_features": True,
        }

        # 临时启用 deepfake_encoder 中可训练层的梯度
        original_grad_state = {}
        for name, param in real_model.deepfake_encoder.named_parameters():
            original_grad_state[name] = param.requires_grad
            if 'output' in name or 'proj' in name or 'bridge' in name or 'linear' in name:
                param.requires_grad = True

        _ = real_model.encode_deepfake_tokens(deepfake_dict)
        cls_logits = getattr(real_model, '_deepfake_logits_for_loss', None)

        cls_loss_val = 0.0
        cls_loss_tensor = None
        if cls_logits is not None:
            cls_loss_tensor = F.cross_entropy(cls_logits, cls_labels)
            cls_loss_val = cls_loss_tensor.detach().item()
            self.accelerator.backward(self.deepfake_cls_loss_weight * cls_loss_tensor)
            print(f'[CLS_LOSS] computed: cls_loss={cls_loss_val:.4f}, labels={cls_labels.tolist()[:4]}...')
        else:
            print('[CLS_LOSS] skip: cls_logits is None after encode_deepfake_tokens')

        # 恢复原始梯度状态
        for name, param in real_model.deepfake_encoder.named_parameters():
            param.requires_grad = original_grad_state.get(name, False)

        return cls_loss_tensor, cls_loss_val

    def training_step(self, model, inputs):
        """
        Override training_step: 正常 Q&A step + 分类 loss。
        分类 loss 值直接加到返回的 loss 中，使 trainer 日志显示真实总 loss。
        """
        # 正常 Q&A step
        loss = super().training_step(model, inputs)

        # 分类数据混合训练
        cls_loss_tensor, cls_loss_val = self._get_cls_loss(model)

        # 把分类 loss 加到 trainer 日志的 loss 中
        if cls_loss_val > 0:
            loss = loss.detach() + cls_loss_val

        return loss

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                self.args.train_batch_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,
                lengths=lengths,
                group_by_modality=True,
            )
        else:
            return super()._get_train_sampler()

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            if self.args.mm_projector_lr is not None:
                projector_parameters = [name for name, _ in opt_model.named_parameters() if "mm_projector" in name or "deepfake_projector" in name or 'deepfake_token' in name]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                    },
                ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")
        
        # for param_group in self.optimizer.param_groups:
        #     for param in param_group['params']:
        #         print(param.shape)
        return self.optimizer

    def _save_checkpoint(self, model, trial, metrics=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            # Only save Adapter
            keys_to_match = ['mm_projector', 'vision_resampler', 'deepfake_projector']
            if getattr(self.args, "use_im_start_end", False):
                keys_to_match.extend(['embed_tokens', 'embed_in'])

            weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        else:
            super(LLaVATrainer, self)._save_checkpoint(model, trial, metrics)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            pass
        else:
            super(LLaVATrainer, self)._save(output_dir, state_dict)
