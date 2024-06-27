import torch
import torch as th
import torch.nn as nn
from modules.diffusionmodules.video_model import VideoUNet, VideoResBlock
from modules.diffusionmodules.util import timestep_embedding
from modules.diffusionmodules.openaimodel import *
from .util import default
from modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
)
from .util import (default, disabled_train, get_obj_from_str,
                    instantiate_from_config, log_txt_as_img)
# vae

from models.autoencoder import AutoencodingEngine
from modules.encoders.modules import FrozenOpenCLIPImagePredictionEmbedder


def disabled_train(self: nn.Module) -> nn.Module:
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

# 重写 svd 的forward 函数, 让可以支持吃注入control条件的输入
class ControlledVideoUNet(VideoUNet):
    # def forward(self, x, timesteps=None, context=None, control=None, only_mid_control=False, **kwargs):
    def forward(
            self,
            x: th.Tensor,
            timesteps: th.Tensor,
            context: Optional[th.Tensor] = None,
            control = None,
            only_mid_control = False,
            y: Optional[th.Tensor] = None,
            time_context: Optional[th.Tensor] = None,
            num_video_frames: Optional[int] = None,
            image_only_indicator: Optional[th.Tensor] = None,
    ):
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional -> no, relax this TODO"
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        if self.num_classes is not None:
            assert y.shape[0] == x.shape[0]
            emb = emb + self.label_emb(y)

        h = x
        for module in self.input_blocks:
            h = module(
                h,
                emb,
                context=context,
                image_only_indicator=image_only_indicator,
                time_context=time_context,
                num_video_frames=num_video_frames,
            )
            hs.append(h)
        h = self.middle_block(
            h,
            emb,
            context=context,
            image_only_indicator=image_only_indicator,
            time_context=time_context,
            num_video_frames=num_video_frames,
        )

        if control is not None:
            h += control.pop()
        for i, module in enumerate(self.output_blocks):
            if only_mid_control or control is None:
                h = th.cat([h, hs.pop()], dim=1)
            else:
                h = torch.cat([h, hs.pop() + control.pop()], dim=1)   # <- control net 注入
            h = module(
                h,
                emb,
                context=context,
                image_only_indicator=image_only_indicator,
                time_context=time_context,
                num_video_frames=num_video_frames,
            )
        h = h.type(x.dtype)
        return self.out(h)


#  ControlNet 部分 照搬diffusionmodules.video_model import VideoUNet 然后添加zero_conv
# 条件图 和 noise concat 送给Control Net
class ControlSVDNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        model_channels: int,
        out_channels: int,
        num_res_blocks: int,
        attention_resolutions: int,
        dropout: float = 0.0,
        channel_mult: List[int] = (1, 2, 4, 8),
        conv_resample: bool = True,
        dims: int = 2,
        num_classes: Optional[int] = None,
        use_checkpoint: bool = False,
        num_heads: int = -1,
        num_head_channels: int = -1,
        num_heads_upsample: int = -1,
        use_scale_shift_norm: bool = False,
        resblock_updown: bool = False,
        transformer_depth: Union[List[int], int] = 1,
        transformer_depth_middle: Optional[int] = None,
        context_dim: Optional[int] = None,
        time_downup: bool = False,
        time_context_dim: Optional[int] = None,
        extra_ff_mix_layer: bool = False,
        use_spatial_context: bool = False,
        merge_strategy: str = "fixed",
        merge_factor: float = 0.5,
        spatial_transformer_attn_type: str = "softmax",
        video_kernel_size: Union[int, List[int]] = 3,
        use_linear_in_transformer: bool = False,
        adm_in_channels: Optional[int] = None,
        disable_temporal_crossattention: bool = False,
        max_ddpm_temb_period: int = 10000,
    ):
        super().__init__()
        assert context_dim is not None

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1

        if num_head_channels == -1:
            assert num_heads != -1

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        if isinstance(transformer_depth, int):
            transformer_depth = len(channel_mult) * [transformer_depth]
        transformer_depth_middle = default(
            transformer_depth_middle, transformer_depth[-1]
        )

        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            if isinstance(self.num_classes, int):
                self.label_emb = nn.Embedding(num_classes, time_embed_dim)
            elif self.num_classes == "continuous":
                print("setting up linear c_adm embedding layer")
                self.label_emb = nn.Linear(1, time_embed_dim)
            elif self.num_classes == "timestep":
                self.label_emb = nn.Sequential(
                    Timestep(model_channels),
                    nn.Sequential(
                        linear(model_channels, time_embed_dim),
                        nn.SiLU(),
                        linear(time_embed_dim, time_embed_dim),
                    ),
                )

            elif self.num_classes == "sequential":
                assert adm_in_channels is not None
                self.label_emb = nn.Sequential(
                    nn.Sequential(
                        linear(adm_in_channels, time_embed_dim),
                        nn.SiLU(),
                        linear(time_embed_dim, time_embed_dim),
                    )
                )
            else:
                raise ValueError()

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1

        def get_attention_layer(
            ch,
            num_heads,
            dim_head,
            depth=1,
            context_dim=None,
            use_checkpoint=False,
            disabled_sa=False,
        ):
            return SpatialVideoTransformer(
                ch,
                num_heads,
                dim_head,
                depth=depth,
                context_dim=context_dim,
                time_context_dim=time_context_dim,
                dropout=dropout,
                ff_in=extra_ff_mix_layer,
                use_spatial_context=use_spatial_context,
                merge_strategy=merge_strategy,
                merge_factor=merge_factor,
                checkpoint=use_checkpoint,
                use_linear=use_linear_in_transformer,
                attn_mode=spatial_transformer_attn_type,
                disable_self_attn=disabled_sa,
                disable_temporal_crossattention=disable_temporal_crossattention,
                max_time_embed_period=max_ddpm_temb_period,
            )

        def get_resblock(
            merge_factor,
            merge_strategy,
            video_kernel_size,
            ch,
            time_embed_dim,
            dropout,
            out_ch,
            dims,
            use_checkpoint,
            use_scale_shift_norm,
            down=False,
            up=False,
        ):
            return VideoResBlock(
                merge_factor=merge_factor,
                merge_strategy=merge_strategy,
                video_kernel_size=video_kernel_size,
                channels=ch,
                emb_channels=time_embed_dim,
                dropout=dropout,
                out_channels=out_ch,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                down=down,
                up=up,
            )


        self.zero_convs = nn.ModuleList([self.make_zero_conv(model_channels)])   #  添加zero_convs

        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    get_resblock(
                        merge_factor=merge_factor,
                        merge_strategy=merge_strategy,
                        video_kernel_size=video_kernel_size,
                        ch=ch,
                        time_embed_dim=time_embed_dim,
                        dropout=dropout,
                        out_ch=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels

                    layers.append(
                        get_attention_layer(
                            ch,
                            num_heads,
                            dim_head,
                            depth=transformer_depth[level],
                            context_dim=context_dim,
                            use_checkpoint=use_checkpoint,
                            disabled_sa=False,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self.zero_convs.append(self.make_zero_conv(ch))              # 添加zero_convs
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                ds *= 2
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        get_resblock(
                            merge_factor=merge_factor,
                            merge_strategy=merge_strategy,
                            video_kernel_size=video_kernel_size,
                            ch=ch,
                            time_embed_dim=time_embed_dim,
                            dropout=dropout,
                            out_ch=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch,
                            conv_resample,
                            dims=dims,
                            out_channels=out_ch,
                            third_down=time_downup,
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                self.zero_convs.append(self.make_zero_conv(ch))              # 添加zero_convs
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels

        self.middle_block = TimestepEmbedSequential(
            get_resblock(
                merge_factor=merge_factor,
                merge_strategy=merge_strategy,
                video_kernel_size=video_kernel_size,
                ch=ch,
                time_embed_dim=time_embed_dim,
                out_ch=None,
                dropout=dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            get_attention_layer(
                ch,
                num_heads,
                dim_head,
                depth=transformer_depth_middle,
                context_dim=context_dim,
                use_checkpoint=use_checkpoint,
            ),
            get_resblock(
                merge_factor=merge_factor,
                merge_strategy=merge_strategy,
                video_kernel_size=video_kernel_size,
                ch=ch,
                out_ch=None,
                time_embed_dim=time_embed_dim,
                dropout=dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self.middle_block_out = self.make_zero_conv(ch)                  # 添加middle_block 的zero_convs
        self._feature_size += ch


        # self.output_blocks = nn.ModuleList([])
        # for level, mult in list(enumerate(channel_mult))[::-1]:
        #     for i in range(num_res_blocks + 1):
        #         ich = input_block_chans.pop()
        #         layers = [
        #             get_resblock(
        #                 merge_factor=merge_factor,
        #                 merge_strategy=merge_strategy,
        #                 video_kernel_size=video_kernel_size,
        #                 ch=ch + ich,
        #                 time_embed_dim=time_embed_dim,
        #                 dropout=dropout,
        #                 out_ch=model_channels * mult,
        #                 dims=dims,
        #                 use_checkpoint=use_checkpoint,
        #                 use_scale_shift_norm=use_scale_shift_norm,
        #             )
        #         ]
        #         ch = model_channels * mult
        #         if ds in attention_resolutions:
        #             if num_head_channels == -1:
        #                 dim_head = ch // num_heads
        #             else:
        #                 num_heads = ch // num_head_channels
        #                 dim_head = num_head_channels
        #
        #             layers.append(
        #                 get_attention_layer(
        #                     ch,
        #                     num_heads,
        #                     dim_head,
        #                     depth=transformer_depth[level],
        #                     context_dim=context_dim,
        #                     use_checkpoint=use_checkpoint,
        #                     disabled_sa=False,
        #                 )
        #             )
        #         if level and i == num_res_blocks:
        #             out_ch = ch
        #             ds //= 2
        #             layers.append(
        #                 get_resblock(
        #                     merge_factor=merge_factor,
        #                     merge_strategy=merge_strategy,
        #                     video_kernel_size=video_kernel_size,
        #                     ch=ch,
        #                     time_embed_dim=time_embed_dim,
        #                     dropout=dropout,
        #                     out_ch=out_ch,
        #                     dims=dims,
        #                     use_checkpoint=use_checkpoint,
        #                     use_scale_shift_norm=use_scale_shift_norm,
        #                     up=True,
        #                 )
        #                 if resblock_updown
        #                 else Upsample(
        #                     ch,
        #                     conv_resample,
        #                     dims=dims,
        #                     out_channels=out_ch,
        #                     third_up=time_downup,
        #                 )
        #             )
        #
        #         self.output_blocks.append(TimestepEmbedSequential(*layers))
        #         self._feature_size += ch
        #
        # self.out = nn.Sequential(
        #     normalization(ch),
        #     nn.SiLU(),
        #     zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        # )

    def make_zero_conv(self, channels):
        return TimestepEmbedSequential(zero_module(conv_nd(self.dims, channels, channels, 1, padding=0)))

    def forward(
        self,
        x: th.Tensor,
        hint: th.Tensor,  # 条件图 和 noise concat 送给Control Net
        timesteps: th.Tensor,
        context: Optional[th.Tensor] = None,
        y: Optional[th.Tensor] = None,
        time_context: Optional[th.Tensor] = None,
        num_video_frames: Optional[int] = None,
        image_only_indicator: Optional[th.Tensor] = None,
    ):
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional -> no, relax this TODO"

        outs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        if self.num_classes is not None:
            assert y.shape[0] == x.shape[0]
            emb = emb + self.label_emb(y)

        x = torch.cat((x, hint), dim=1)
        h = x.type(self.dtype)
        for module, zero_conv in zip(self.input_blocks, self.zero_convs):
            h = module(
                h,
                emb,
                context=context,
                image_only_indicator=image_only_indicator,
                time_context=time_context,
                num_video_frames=num_video_frames,
            )
            outs.append(
                zero_conv(
                    h,
                    emb,
                    context=context
                )
            )
        h = self.middle_block(
            h,
            emb,
            context=context,
            image_only_indicator=image_only_indicator,
            time_context=time_context,
            num_video_frames=num_video_frames,
        )
        outs.append(
            self.middle_block_out(
                h,
                emb,
                context=context
            )
        )
        return outs


#   把所有模型包起来串起来 取名 ControlSVDLDM
class ControlSVDLDM(nn.Module):

    def __init__(
            self,
            model_cfg
    ):
        super().__init__()
        self.svd_unet = ControlledVideoUNet(**model_cfg.network_config)
        self.vae = AutoencodingEngine(**model_cfg.first_stage_config)
        self.clip = FrozenOpenCLIPImagePredictionEmbedder(**model_cfg.conditioner_config)
        self.controlsvdnet = ControlSVDNet(**model_cfg.cldm_network_config)
        self.denoiser = instantiate_from_config(model_cfg.denoiser_config)
        self.sampler = instantiate_from_config(model_cfg.sampler_config)


    @torch.no_grad()
    def load_pretrained(self,
            svd = None,
            vae = None,
            clip = None,
            controlnet = None
    ):
        if svd is not None:
            self.svd_unet.load_state_dict(svd, strict=True)
        if vae is not None:
            self.vae.load_state_dict(vae, strict=True)
        if clip is not None:
            self.clip.load_state_dict(clip, strict=True)

        # 测试阶段加载预训练参数，训练阶段从svd搬参数做初始化
        if controlnet is not None:
            self.controlsvdnet.load_state_dict(controlnet, strict=True)
        else:
            _, _ = self.load_controlnet_from_svdunet()  # 从头训练 从svd unet搬运参数初始化

        for module in [self.vae, self.clip, self.svd_unet, self.controlsvdnet]:
            module.eval()
            module.train = disabled_train
            for p in module.parameters():
                p.requires_grad = False

    @torch.no_grad()
    def load_controlnet_from_ckpt(self, statedict) -> None:
        self.controlsvdnet.load_state_dict(statedict, strict=True)

    @torch.no_grad()
    def load_controlnet_from_svdunet(self):
        unet_sd = self.svd_unet.state_dict()
        scratch_sd = self.controlsvdnet.state_dict()
        init_sd = {}
        init_with_new_zero = set()
        init_with_scratch = set()
        for key in scratch_sd:
            if key in unet_sd:
                this, target = scratch_sd[key], unet_sd[key]
                if this.size() == target.size():
                    init_sd[key] = target.clone()
                else:
                    d_ic = this.size(1) - target.size(1)
                    oc, _, h, w = this.size()
                    zeros = torch.zeros((oc, d_ic, h, w), dtype=target.dtype)
                    init_sd[key] = torch.cat((target, zeros), dim=1)
                    init_with_new_zero.add(key)
            else:
                init_sd[key] = scratch_sd[key].clone()
                init_with_scratch.add(key)
        self.controlsvdnet.load_state_dict(init_sd, strict=True)
        return init_with_new_zero, init_with_scratch

    def vae_encode(self, image: torch.Tensor) -> torch.Tensor:
        return self.vae.encode(image) * self.scale_factor

    def vae_decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(z / self.scale_factor)

    def prepare_condition(self, clean: torch.Tensor, txt: List[str]):
        return dict(
            c_txt=self.clip.encode(txt),
            c_img=self.vae_encode(clean * 2 - 1)
        )

    def forward(self, x_noisy, t,  cond):
        c_txt = cond["c_txt"]
        c_img = cond["c_img"]
        control = self.controlsvdnet(
            x=x_noisy, hint=c_img,
            timesteps=t, context=c_txt
        )
        # control = [c * scale for c, scale in zip(control, self.control_scales)]
        eps = self.svd_unet(
            x=x_noisy, timesteps=t,
            context=c_txt, control=control, only_mid_control=False
        )
        return eps





























