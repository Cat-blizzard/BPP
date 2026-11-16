"""Native BPP with a visual human prompt and TIG robot observations.

Training contract: ``obs.rgb`` is (B, 2, 3, 224, 224), ``obs.qpos`` is
(B, 2, 18), ``obs.prompt.obs.rgb`` is (B, P, 3, 224, 224), and ``action``
is (B, 16, 16). Prompt padding uses a boolean (B, P) mask: True is ignored.
RGB is float in [0, 1]; state/action use the existing TIG normalized space.
There are no prompt actions or prompt proprioceptive inputs. Native BPP's
tokenizer, attention pooling, cross-attention, UNet, and diffusion loss are used.
"""

from __future__ import annotations

import argparse
import copy
import math
from collections import deque
from collections.abc import Mapping
from pathlib import Path
import sys
from typing import Any

import torch
from torch import Tensor, nn
from omegaconf import OmegaConf
from diffusers import DDIMScheduler
from timm.layers import AttentionPoolLatent
from torchvision.transforms import ColorJitter

BPP_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_CONFIG = Path(__file__).resolve().parent / 'configs' / 'model.yaml'
MODEL_SCHEMA_VERSION = 2
MODEL_RECIPE = 'bpp-tig-split-clip-frozen-prompt-v2'
LEGACY_MODEL_RECIPE = 'bpp-tig-shared-clip-v1'
if not (BPP_ROOT / 'behavior_prompting').is_dir():
    raise ImportError(f'Official BPP checkout is missing: {BPP_ROOT}')
if str(BPP_ROOT) not in sys.path:
    sys.path.insert(0, str(BPP_ROOT))

from behavior_prompting.train_network.common.augmentation import ImageAugmentation
from behavior_prompting.train_network.model.common.normalizer import (
    LinearNormalizer, Normalizer, SingleFieldLinearNormalizer,
)
from behavior_prompting.train_network.model.prompt.prompt_obs_encoder import (
    PairPromptObsEncoder, PairPromptTransformerTokenizer,
)
from behavior_prompting.train_network.model.vision.transformer_obs_encoder import (
    TransformerObsEncoder,
)
from behavior_prompting.train_network.policy.diffusion_unet_policy import (
    DiffusionUnetPolicy,
)

TensorDict = dict[str, Any]


class _VideoObservationEncoder(TransformerObsEncoder):
    """Keep timm's random initialization for the offline diagnostic path.

    The upstream non-pretrained hook applies a transformer-only initializer to
    timm's Conv2d patch embed and raises. The pretrained=True production path
    uses the upstream hook unchanged; no layer or architecture is replaced.
    """

    def init_weights(self) -> None:
        if self.pretrained:
            super().init_weights()

    def freeze_rgb_backbone(self) -> None:
        """Freeze RGB weights and preprocessing, leaving projections trainable.

        The prompt encoder also embeds the current robot query. Its RGB tokens
        therefore use this same fixed backbone; qpos projection and downstream
        pooling/cross-attention continue to receive gradients.
        """
        self._rgb_backbone_frozen = True
        self.key_model_map.requires_grad_(False)
        self.key_model_map.eval()
        self.key_train_transform_map = copy.deepcopy(self.key_eval_transform_map)
        self.key_non_prompt_train_image_transform_map = nn.ModuleDict({
            key: nn.Identity() for key in self.key_model_map
        })

    def train(self, mode: bool = True) -> _VideoObservationEncoder:
        super().train(mode)
        if getattr(self, '_rgb_backbone_frozen', False):
            self.key_model_map.eval()
        return self


def validate_model_config(config: Mapping[str, Any]) -> None:
    """Keep legacy checkpoints explicit, and reject ambiguous new recipes."""
    version = config.get('schema_version')
    assert version in (1, MODEL_SCHEMA_VERSION), 'Unsupported model config version'
    for key in ['data', 'vision', 'prompt', 'diffusion', 'optimizer']:
        assert isinstance(config.get(key), Mapping), f'Missing model config section: {key}'
    if version == 1:
        assert config.get('recipe', LEGACY_MODEL_RECIPE) == LEGACY_MODEL_RECIPE
        assert not config['vision'].get('separate_prompt_and_receding_obs_encoders', False)
    else:
        assert config.get('recipe') == MODEL_RECIPE, 'Unexpected schema-2 training recipe'
        vision = config['vision']
        assert vision.get('separate_prompt_and_receding_obs_encoders') is True
        assert vision.get('prompt_frozen') is True, 'The prompt RGB backbone must be frozen'
        assert vision.get('receding_frozen') is False, 'The robot RGB backbone must be trainable'
        assert 'frozen' not in vision, 'Use explicit prompt/receding freeze settings in schema 2'


def load_model_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load a portable model configuration without changing the working dir."""
    config_path = DEFAULT_MODEL_CONFIG if path is None else Path(path).expanduser()
    if not config_path.is_absolute():
        config_path = (Path.cwd() / config_path).resolve()
    config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    assert isinstance(config, dict), 'Model config must be a YAML mapping'
    validate_model_config(config)
    return config


def _identity_normalizer() -> Normalizer:
    result = Normalizer()
    for key in ['rgb', 'qpos', 'action']:
        result[key] = SingleFieldLinearNormalizer.create_identity()
    prompt_normalizer = LinearNormalizer()
    prompt_normalizer['rgb'] = SingleFieldLinearNormalizer.create_identity()
    result.set_prompt_normalizer(prompt_normalizer)
    result.requires_grad_(False)
    return result


class BPPVideoPolicy(DiffusionUnetPolicy):
    """A trainable native diffusion policy with validated video-only inputs."""

    def __init__(self, model_config: Mapping[str, Any], pretrained: bool = True) -> None:
        config = copy.deepcopy(dict(model_config))
        validate_model_config(config)
        config['vision']['pretrained'] = bool(pretrained)
        data, vision, prompt, diffusion = (
            config['data'], config['vision'], config['prompt'], config['diffusion'],
        )
        image_size = int(data['image_size'])
        obs_horizon = int(data['obs_horizon'])
        assert image_size == 224, 'The CLIP ViT-B/16 recipe uses 224 x 224 RGB'
        assert int(data['state_dim']) == 18 and int(data['action_dim']) == 16
        assert obs_horizon >= 1 and int(data['prompt_frames']) >= 3
        assert 1 <= int(data['execute_horizon']) <= int(data['action_horizon'])
        shape_meta = OmegaConf.create({
            'image_resolution': image_size,
            'use_prompting': True,
            'prompt_sample_mode': 'pair',
            'prompt_sequence_length': int(data['prompt_frames']),
            'max_prompt_full_demos': 1,
            'prompt_chunk_n_actions': 1,
            'max_sequence_length': int(data['prompt_frames']),
            'action': {'shape': [16], 'horizon': int(data['action_horizon'])},
            'obs': {
                'rgb': {
                    'shape': [3, image_size, image_size], 'horizon': obs_horizon,
                    'type': 'rgb', 'prompt_type': 'observation',
                    'include_in_receding_obs': True,
                    'include_in_prompt_current_obs': True,
                },
                'qpos': {
                    'shape': [18], 'horizon': obs_horizon, 'type': 'low_dim',
                    'prompt_type': 'ignore', 'include_in_receding_obs': True,
                    'include_in_prompt_current_obs': True,
                },
            },
        })
        train_crop = OmegaConf.create([
            {'type': 'RandomCrop', 'ratio': float(vision['train_crop_ratio'])},
        ])
        train_transforms = list(train_crop) + [ColorJitter(**vision['color_jitter'])]
        eval_transforms = OmegaConf.create([
            {'type': 'CenterCrop', 'ratio': float(vision['eval_crop_ratio'])},
        ])
        observation_encoder = _VideoObservationEncoder(
            shape_meta=shape_meta, model_name=str(vision['model_name']),
            pretrained=bool(pretrained), frozen=False,
            n_emb=int(prompt['n_emb']), global_pool='',
            feature_aggregation=str(vision['feature_aggregation']),
            train_image_transforms=ImageAugmentation(shape_meta, train_transforms),
            eval_image_transforms=ImageAugmentation(shape_meta, eval_transforms),
            use_group_norm=True, share_rgb_model=True, use_vision_norm=True,
            concat_time_dimension=False,
            pretrained_lr_scale=float(vision['pretrained_lr_scale']),
        )
        tokenizer = PairPromptTransformerTokenizer(
            shape_meta=shape_meta, obs_encoder=observation_encoder,
            n_emb=int(prompt['n_emb']), concat_prompt_and_receding=False,
            ignore_prompt_obs=False, ignore_prompt_proprio=True,
            ignore_prompt_action=True, merge_prompt_tokens='obs_and_action',
            prompt_attention_pool=AttentionPoolLatent(
                in_features=int(prompt['n_emb']),
                num_heads=int(prompt['attention_pool_heads']),
                mlp_ratio=float(prompt['attention_pool_mlp_ratio']), pool_type='token',
            ),
            use_pool_modality_pos_embed=bool(prompt['use_pool_modality_pos_embed']),
            separate_prompt_and_receding_obs_encoders=bool(
                vision.get('separate_prompt_and_receding_obs_encoders', False)
            ),
        )
        if config['schema_version'] == MODEL_SCHEMA_VERSION:
            # The native tokenizer deep-copies the generic pretrained encoder
            # before either branch receives training updates.
            assert tokenizer.prompt_obs_encoder is not tokenizer.receding_obs_encoder
            tokenizer.prompt_obs_encoder.freeze_rgb_backbone()
        elif bool(vision.get('frozen', False)):
            tokenizer.prompt_obs_encoder.freeze_rgb_backbone()
        encoder = PairPromptObsEncoder(
            shape_meta=shape_meta, obs_encoder=tokenizer,
            prompt_encoder_enabled=False, prompt_with_obs_decoder_enabled=True,
            prompt_with_obs_decoder_n_layer=int(prompt['cross_attention_layers']),
            prompt_with_obs_decoder_n_head=int(prompt['cross_attention_heads']),
            flatten_output=True, n_emb=int(prompt['n_emb']),
            p_drop_attn=float(prompt['attention_dropout']),
            use_attention_sink=bool(prompt['attention_sink']),
            num_attention_sink_tokens=int(prompt['attention_sink_tokens']),
        )
        scheduler = DDIMScheduler(
            num_train_timesteps=int(diffusion['train_timesteps']),
            beta_start=float(diffusion['beta_start']), beta_end=float(diffusion['beta_end']),
            beta_schedule=str(diffusion['beta_schedule']),
            clip_sample=bool(diffusion['clip_sample']), set_alpha_to_one=True,
            steps_offset=0, prediction_type=str(diffusion['prediction_type']),
        )
        super().__init__(
            shape_meta=shape_meta, obs_encoder=encoder, noise_scheduler=scheduler,
            num_inference_steps=int(diffusion['inference_steps']), obs_as_global_cond=True,
            diffusion_step_embed_dim=int(diffusion['diffusion_step_embed_dim']),
            down_dims=tuple(int(dim) for dim in diffusion['down_dims']),
            kernel_size=int(diffusion['kernel_size']), n_groups=int(diffusion['n_groups']),
            cond_predict_scale=bool(diffusion['cond_predict_scale']),
            input_pertub=float(diffusion['input_perturbation']), name='bpp-video-tig',
        )
        self.model_config = config
        self.model_recipe = MODEL_RECIPE if config['schema_version'] == MODEL_SCHEMA_VERSION else LEGACY_MODEL_RECIPE
        self.shape_meta = shape_meta
        self.obs_horizon = obs_horizon
        self.prompt_frames = int(data['prompt_frames'])
        self.execute_horizon = int(data['execute_horizon'])
        self.image_size = image_size
        self.set_normalizer(_identity_normalizer())
        assert self.obs_feature_dim == 4 * obs_horizon * int(prompt['n_emb'])

    def _check_rgb(self, value: Tensor, *, name: str, horizon: int | None) -> None:
        assert isinstance(value, Tensor) and value.ndim == 5, f'{name}: expected BTCHW'
        assert tuple(value.shape[2:]) == (3, self.image_size, self.image_size), name
        assert value.is_floating_point(), f'{name}: use floating RGB in [0,1]'
        assert value.shape[0] > 0 and value.shape[1] > 0, f'{name}: empty batch or time'
        if horizon is not None:
            assert value.shape[1] == horizon, f'{name}: history must have {horizon} frames'
        assert bool(torch.isfinite(value).all()), f'{name}: nonfinite RGB'
        assert bool((value >= -1e-5).all() and (value <= 1.00001).all()), name

    def _check_prompt(self, prompt_dict: Mapping[str, Any]) -> None:
        assert set(prompt_dict) == {'obs', 'metadata'}, 'Video prompt must not contain actions'
        assert set(prompt_dict['obs']) == {'rgb'}, 'Video prompt must contain RGB only'
        rgb = prompt_dict['obs']['rgb']
        self._check_rgb(rgb, name='prompt.rgb', horizon=None)
        assert rgb.shape[1] <= self.prompt_frames, 'Prompt exceeds positional capacity'
        mask = prompt_dict['metadata']['mask']
        assert isinstance(mask, Tensor) and mask.dtype == torch.bool
        assert tuple(mask.shape) == tuple(rgb.shape[:2]) and mask.device == rgb.device
        assert bool((~mask).any(dim=1).all()), 'Every prompt needs at least one valid frame'

    def _check_observation(self, obs_dict: Mapping[str, Any]) -> int:
        self._check_rgb(obs_dict['rgb'], name='query.rgb', horizon=self.obs_horizon)
        batch_size = obs_dict['rgb'].shape[0]
        qpos = obs_dict['qpos']
        assert isinstance(qpos, Tensor) and qpos.is_floating_point()
        assert tuple(qpos.shape) == (batch_size, self.obs_horizon, 18)
        assert qpos.device == obs_dict['rgb'].device and bool(torch.isfinite(qpos).all())
        return batch_size

    def compute_loss(self, batch: TensorDict) -> Tensor:
        batch_size = self._check_observation(batch['obs'])
        self._check_prompt(batch['obs']['prompt'])
        assert not self.obs_encoder.is_prompted, 'Reset inference prompt before training'
        assert batch['obs']['prompt']['obs']['rgb'].shape[0] in (1, batch_size)
        action = batch['action']
        assert isinstance(action, Tensor) and action.is_floating_point()
        assert tuple(action.shape) == (batch_size, self.action_horizon, self.action_dim)
        assert action.device == batch['obs']['rgb'].device and bool(torch.isfinite(action).all())
        loss = super().compute_loss(batch)
        assert loss.ndim == 0 and bool(torch.isfinite(loss)), 'Diffusion loss is nonfinite'
        return loss

    @torch.inference_mode()
    def prompt(self, prompt_dict: TensorDict) -> None:
        self._check_prompt(prompt_dict)
        assert not self.training, 'Call eval() before caching an inference prompt'
        assert not self.obs_encoder.is_prompted, 'Reset before changing the prompt'
        super().prompt(prompt_dict)

    def predict_action(
        self, obs_dict: TensorDict, fixed_action_prefix: Tensor | None = None,
        need_weights: bool = False, average_attn_weights: bool = False,
    ) -> TensorDict:
        batch_size = self._check_observation(obs_dict)
        if 'prompt' in obs_dict:
            self._check_prompt(obs_dict['prompt'])
        else:
            assert self.obs_encoder.is_prompted, 'Call prompt() before inference'
        result = super().predict_action(
            obs_dict, fixed_action_prefix=fixed_action_prefix,
            need_weights=need_weights, average_attn_weights=average_attn_weights,
        )
        assert result['action'].shape == (batch_size, self.action_horizon, self.action_dim)
        assert bool(torch.isfinite(result['action']).all()), 'Nonfinite predicted actions'
        return result

    def get_optimizer(
        self, lr: float = 5e-5, weight_decay: float = 1e-6,
        betas: tuple[float, float] = (0.95, 0.999), **kwargs: Any,
    ) -> torch.optim.AdamW:
        """Native AdamW groups with explicit final LR, without GPU scaling."""
        assert lr > 0 and weight_decay >= 0
        optimizer = super().get_optimizer(lr=lr, weight_decay=weight_decay, betas=betas, **kwargs)
        vision_ids = {
            id(parameter) for parameter in
            self.obs_encoder.obs_encoder.receding_obs_encoder.key_model_map.parameters()
        }
        for index, group in enumerate(optimizer.param_groups):
            group['params'] = [parameter for parameter in group['params'] if parameter.requires_grad]
            if group['params'] and all(id(parameter) in vision_ids for parameter in group['params']):
                group['name'] = 'vision'
                group['lr'] = lr * float(self.model_config['vision']['pretrained_lr_scale'])
            else:
                group['name'] = f'policy_{index}'
        optimizer.param_groups[:] = [group for group in optimizer.param_groups if group['params']]
        return optimizer


def build_policy(
    model_config_path: str | Path | None = None, pretrained: bool = True,
) -> BPPVideoPolicy:
    """Build the full native BPP architecture; False never fetches pretrained weights."""
    return BPPVideoPolicy(load_model_config(model_config_path), pretrained=pretrained)


class BPPAgent(nn.Module):
    """TIG agent returning complete action chunks in TIG normalized space.

    For exact control-step history, pass explicit (B, O, ...) observations or call
    observe() at every control step. Calling only get_action() every four steps
    cannot recover the intervening observation history.
    """

    def __init__(self, policy: BPPVideoPolicy) -> None:
        super().__init__()
        self.policy = policy
        self.pred_horizon = policy.action_horizon
        self.obs_horizon = policy.obs_horizon
        self.act_horizon = policy.execute_horizon
        self._history: deque[tuple[Tensor, Tensor]] = deque(maxlen=self.obs_horizon)
        self._last_observation: Mapping[str, Tensor] | None = None

    @property
    def device(self) -> torch.device:
        return self.policy.device

    def _rgb(self, value: Tensor, *, video: bool) -> Tensor:
        assert isinstance(value, Tensor), 'RGB must be a torch tensor'
        is_uint8 = value.dtype == torch.uint8
        if value.ndim == 4:
            value = value.unsqueeze(0 if video else 1)
        assert value.ndim == 5, 'RGB must be BTCHW or BTHWC'
        if value.shape[-1] == 3:
            value = value.permute(0, 1, 4, 2, 3)
        value = value.to(self.device, dtype=torch.float32)
        if is_uint8:
            value = value / 255.0
        return value.contiguous()

    @torch.inference_mode()
    def prepare_for_eval(
        self, human_video: Tensor | None, robot_obs: Mapping[str, Tensor],
        human_desc: list[str] | None = None, human_vl_ids: list[str] | None = None,
    ) -> None:
        del robot_obs, human_desc, human_vl_ids
        assert human_video is not None, 'BPP-Video requires an actual human demonstration'
        self.clear_cache()
        self.eval()
        rgb = self._rgb(human_video, video=True)
        total_frames, desired = rgb.shape[1], self.policy.prompt_frames
        assert total_frames > 0
        valid_frames = min(total_frames, desired)
        indices = torch.linspace(0, total_frames - 1, valid_frames, device=rgb.device).round().long()
        rgb = rgb.index_select(1, indices)
        if valid_frames < desired:
            padding = rgb[:, -1:].expand(-1, desired - valid_frames, -1, -1, -1)
            rgb = torch.cat([rgb, padding], dim=1)
        mask = torch.arange(desired, device=rgb.device)[None, :].expand(rgb.shape[0], -1) >= valid_frames
        self.policy.prompt({'obs': {'rgb': rgb}, 'metadata': {'mask': mask}})

    def _extract_observation(self, obs: Mapping[str, Tensor]) -> tuple[Tensor, Tensor]:
        rgb_value = obs.get('rgb', obs.get('view_1'))
        qpos = obs.get('qpos', obs.get('state', obs.get('states')))
        assert rgb_value is not None and qpos is not None, 'Need RGB and normalized qpos'
        rgb = self._rgb(rgb_value, video=False)
        qpos = qpos.to(device=self.device, dtype=torch.float32)
        if qpos.ndim == 2:
            qpos = qpos.unsqueeze(1)
        assert qpos.ndim == 3 and qpos.shape[-1] == 18
        assert rgb.shape[:2] == qpos.shape[:2], 'RGB/state histories differ'
        return rgb, qpos

    @torch.inference_mode()
    def observe(self, obs: Mapping[str, Tensor]) -> None:
        if self._last_observation is obs:
            return
        rgb, qpos = self._extract_observation(obs)
        for time_index in range(rgb.shape[1]):
            self._history.append((rgb[:, time_index].clone(), qpos[:, time_index].clone()))
        self._last_observation = obs

    @torch.inference_mode()
    def get_action(self, obs: Mapping[str, Tensor]) -> Tensor:
        rgb, qpos = self._extract_observation(obs)
        if rgb.shape[1] >= self.obs_horizon:
            current = {'rgb': rgb[:, -self.obs_horizon:], 'qpos': qpos[:, -self.obs_horizon:]}
        else:
            self.observe(obs)
            assert self._history, 'Observation history is empty'
            history = list(self._history)
            history = [history[0]] * (self.obs_horizon - len(history)) + history
            current = {
                'rgb': torch.stack([frame[0] for frame in history], dim=1),
                'qpos': torch.stack([frame[1] for frame in history], dim=1),
            }
        return self.policy.predict_action(current)['action']

    def clear_cache(self) -> None:
        self.policy.reset(action_exec_horizon=self.act_horizon)
        self._history.clear()
        self._last_observation = None


def _self_test() -> None:
    """Full-size CPU forward/backward and cached inference, without downloads."""
    torch.set_num_threads(4)
    torch.manual_seed(7)
    config = load_model_config()
    for field, invalid_value in [
        ('prompt_frozen', False), ('receding_frozen', True),
        ('separate_prompt_and_receding_obs_encoders', False),
    ]:
        malformed_config = copy.deepcopy(config)
        malformed_config['vision'][field] = invalid_value
        try:
            validate_model_config(malformed_config)
        except AssertionError:
            pass
        else:
            raise AssertionError(f'Invalid split recipe accepted: {field}')
    policy = build_policy(pretrained=False)
    assert policy.model_config['schema_version'] == MODEL_SCHEMA_VERSION
    assert policy.model_recipe == MODEL_RECIPE
    assert policy.obs_horizon == 2 and policy.prompt_frames == 10
    assert policy.action_horizon == 16 and policy.execute_horizon == 4
    tokenizer = policy.obs_encoder.obs_encoder
    prompt_encoder = tokenizer.prompt_obs_encoder
    robot_encoder = tokenizer.receding_obs_encoder
    prompt_vision = prompt_encoder.key_model_map['rgb']
    robot_vision = robot_encoder.key_model_map['rgb']
    assert prompt_vision is not robot_vision, 'The two RGB branches still share weights'
    for prompt_parameter, robot_parameter in zip(prompt_vision.parameters(), robot_vision.parameters()):
        assert prompt_parameter is not robot_parameter
        assert torch.equal(prompt_parameter, robot_parameter), 'CLIP branches must start identically'
    assert not any(parameter.requires_grad for parameter in prompt_vision.parameters())
    assert all(parameter.requires_grad for parameter in robot_vision.parameters())
    assert all(parameter.requires_grad for parameter in prompt_encoder.key_projection_map['qpos'].parameters())
    batch: TensorDict = {
        'obs': {
            'rgb': torch.zeros(1, 2, 3, 224, 224),
            'qpos': torch.rand(1, 2, 18),
            'prompt': {
                'obs': {'rgb': torch.rand(1, 10, 3, 224, 224)},
                'metadata': {'mask': torch.tensor([[False] * 8 + [True] * 2])},
            },
        },
        'action': torch.zeros(1, 16, 16),
    }
    policy.train()
    assert policy.training and prompt_encoder.training and robot_vision.training
    assert not prompt_vision.training, 'policy.train() re-enabled frozen CLIP training mode'
    with torch.no_grad():
        fixed_rgb, _ = prompt_encoder(batch['obs'], rgb_keys=['rgb'], low_dim_keys=[])
        repeated_rgb, _ = prompt_encoder(batch['obs'], rgb_keys=['rgb'], low_dim_keys=[])
    assert torch.equal(fixed_rgb, repeated_rgb), 'Frozen RGB preprocessing is stochastic'
    prompt_before = next(prompt_vision.parameters()).detach().clone()
    robot_parameter = robot_vision.patch_embed.proj.weight
    robot_before = robot_parameter.detach().clone()
    optimizer = policy.get_optimizer()
    optimized_ids = {id(parameter) for group in optimizer.param_groups for parameter in group['params']}
    assert not optimized_ids.intersection(id(parameter) for parameter in prompt_vision.parameters())
    assert id(robot_parameter) in optimized_ids
    assert any(group['name'] == 'vision' and math.isclose(group['lr'], 5e-6) for group in optimizer.param_groups)
    loss = policy.compute_loss(batch)
    loss.backward()
    assert all(parameter.grad is None for parameter in prompt_vision.parameters())
    assert robot_parameter.grad is not None and bool(robot_parameter.grad.ne(0).any())
    assert any(parameter.grad is not None and bool(parameter.grad.ne(0).any()) for parameter in prompt_encoder.key_projection_map['qpos'].parameters())
    assert any(parameter.grad is not None and bool(parameter.grad.ne(0).any()) for parameter in tokenizer.prompt_attention_pool.parameters())
    assert any(parameter.grad is not None and bool(parameter.grad.ne(0).any()) for parameter in policy.obs_encoder.prompt_with_obs_decoder.parameters())
    assert all(bool(torch.isfinite(parameter.grad).all()) for parameter in policy.parameters() if parameter.grad is not None)
    optimizer.step()
    assert not torch.equal(robot_before, robot_parameter.detach()), 'Optimizer did not update robot CLIP'
    assert torch.equal(prompt_before, next(prompt_vision.parameters())), 'Frozen CLIP weights changed'
    policy.zero_grad(set_to_none=True)
    del optimizer
    policy.eval()
    with torch.inference_mode():
        conditioning, _ = policy.obs_encoder(batch['obs'])
        changed_padding = copy.deepcopy(batch['obs'])
        changed_padding['prompt']['obs']['rgb'][:, -2:] = 1 - changed_padding['prompt']['obs']['rgb'][:, -2:]
        masked_conditioning, _ = policy.obs_encoder(changed_padding)
        assert torch.allclose(conditioning, masked_conditioning, atol=1e-5, rtol=1e-5), 'Padding changed the condition'
    agent = BPPAgent(policy)
    agent.prepare_for_eval(batch['obs']['prompt']['obs']['rgb'], batch['obs'])
    assert policy.obs_encoder.is_prompted
    action = agent.get_action({'rgb': batch['obs']['rgb'], 'state': batch['obs']['qpos']})
    assert action.shape == (1, 16, 16) and bool(torch.isfinite(action).all())
    agent.clear_cache()
    assert not policy.obs_encoder.is_prompted and policy.obs_encoder.prompt_tokens_cache is None
    malformed = copy.deepcopy(batch['obs']['prompt'])
    malformed['metadata']['mask'].fill_(True)
    try:
        policy._check_prompt(malformed)
    except AssertionError:
        pass
    else:
        raise AssertionError('All-padding prompt was accepted')
    print({
        'self_test': 'passed', 'loss': float(loss.detach()),
        'action_shape': list(action.shape), 'conditioning_dim': int(policy.obs_feature_dim),
        'parameters': sum(parameter.numel() for parameter in policy.parameters()),
        'model_recipe': policy.model_recipe,
        'prompt_backbone_frozen': True, 'robot_backbone_updated': True,
        'downloaded_weights': False,
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true', help='Run the offline full-model CPU sanity check')
    parser.parse_args()
    _self_test()
