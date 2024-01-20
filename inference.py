from typing import List, Tuple, Optional
import os
import math
from argparse import ArgumentParser, Namespace

import numpy as np
import torch
import einops
import pytorch_lightning as pl
from PIL import Image
from omegaconf import OmegaConf

from ldm.xformers_state import disable_xformers
from model.spaced_sampler import SpacedSampler
from model.cldm import ControlLDM
from model.cond_fn import MSEGuidance
from utils.image import auto_resize, pad
from utils.common import instantiate_from_config, load_state_dict
from utils.file import list_image_files, get_file_name_parts

# 定义一个处理函数，无需计算梯度
@torch.no_grad()
def process(
    model: ControlLDM,
    control_imgs: List[np.ndarray],
    steps: int,
    strength: float,
    color_fix_type: str,
    disable_preprocess_model: bool,
    cond_fn: Optional[MSEGuidance],
    tiled: bool,
    tile_size: int,
    tile_stride: int
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Apply DiffBIR model on a list of low-quality images.
    对一系列低质量图像应用 DiffBIR 模型。
    
    Args:
        model (ControlLDM): Model.
        模型对象。
        control_imgs (List[np.ndarray]): A list of low-quality images (HWC, RGB, range in [0, 255]).
        低质量图像列表(HWC, RGB, 数值范围 [0, 255])。
        steps (int): Sampling steps.
        采样步数。
        strength (float): Control strength. Set to 1.0 during training.
        控制强度,在训练期间设为1.0。
        color_fix_type (str): Type of color correction for samples.
        样本的颜色校正类型。
        disable_preprocess_model (bool): If specified, preprocess model (SwinIR) will not be used.
        如果为True,预处理模型(SwinIR)将不被使用。
        cond_fn (Guidance | None): Guidance function that returns gradient to guide the predicted x_0.
        指导函数,返回梯度以指导预测的 x_0。
        tiled (bool): If specified, a patch-based sampling strategy will be used for sampling.
        如果为True,将使用基于补丁的采样策略。
        tile_size (int): Size of patch.
        补丁大小。
        tile_stride (int): Stride of sliding patch.
        滑动补丁的步幅。
    
    Returns:
        preds (List[np.ndarray]): Restoration results (HWC, RGB, range in [0, 255]).恢复结果(HWC, RGB, 数值范围 [0, 255])。
        stage1_preds (List[np.ndarray]): Outputs of preprocess model (HWC, RGB, range in [0, 255]). 预处理模型的输出(HWC, RGB, 数值范围 [0, 255])。
            If `disable_preprocess_model` is specified, then preprocess model's outputs is the same 
            as low-quality inputs.
            如果指定了 `disable_preprocess_model`,则预处理模型的输出与低质量输入相同。
    """
    # 计算输入图像的数量
    n_samples = len(control_imgs)
    # 初始化SpacedSampler对象，用于图像的采样处理
    sampler = SpacedSampler(model, var_type="fixed_small")

    # 将图像列表转换为PyTorch张量，并归一化到[0, 1]范围
    control = torch.tensor(np.stack(control_imgs) / 255.0, dtype=torch.float32, device=model.device).clamp_(0, 1)

    # 调整张量的形状以匹配模型的输入格式
    control = einops.rearrange(control, "n h w c -> n c h w").contiguous()
    
    # 如果不禁用预处理模型，则对控制图像应用预处理模型
    if not disable_preprocess_model:
        control = model.preprocess_model(control)
    # 设置模型的控制尺度
    model.control_scales = [strength] * 13
    
    # 如果提供了指导函数，则使用指导函数来加载目标
    if cond_fn is not None:
        cond_fn.load_target(2 * control - 1)
    
    # 获取处理后的图像的高度和宽度
    height, width = control.size(-2), control.size(-1)
    shape = (n_samples, 4, height // 8, width // 8)

    # 初始化随机噪声张量
    x_T = torch.randn(shape, device=model.device, dtype=torch.float32)
    
    # 根据是否采用基于补丁的采样策略来选择采样方法
    if not tiled:
        samples = sampler.sample(
            steps=steps, shape=shape, cond_img=control,
            positive_prompt="", negative_prompt="", x_T=x_T,
            cfg_scale=1.0, cond_fn=cond_fn,
            color_fix_type=color_fix_type
        )
    else:
        samples = sampler.sample_with_mixdiff(
            tile_size=tile_size, tile_stride=tile_stride, # 补丁采样
            steps=steps, shape=shape, cond_img=control,
            positive_prompt="", negative_prompt="", x_T=x_T,
            cfg_scale=1.0, cond_fn=cond_fn,
            color_fix_type=color_fix_type
        )
    # 将采样结果处理为图像格式
    x_samples = samples.clamp(0, 1)
    x_samples = (einops.rearrange(x_samples, "b c h w -> b h w c") * 255).cpu().numpy().clip(0, 255).astype(np.uint8)
    
    # 处理控制图像以用于后续比较
    control = (einops.rearrange(control, "b c h w -> b h w c") * 255).cpu().numpy().clip(0, 255).astype(np.uint8)
    
    # 准备最终的预测结果和第一阶段的预测结果
    preds = [x_samples[i] for i in range(n_samples)]
    stage1_preds = [control[i] for i in range(n_samples)]
    
    return preds, stage1_preds

# 定义解析命令行参数的函数
def parse_args() -> Namespace:
    parser = ArgumentParser()
    
    # TODO: add help info for these options
    parser.add_argument("--ckpt", required=True, type=str, help="full checkpoint path")  # 指定模型检查点的完整路径，此参数是必需的
    parser.add_argument("--config", required=True, type=str, help="model config path")  # 指定模型配置文件的路径，此参数是必需的
    parser.add_argument("--reload_swinir", action="store_true")  # 如果指定，将重新加载SwinIR模型
    parser.add_argument("--swinir_ckpt", type=str, default="")  # SwinIR模型的检查点路径，如果不指定，默认为空字符串
    
    parser.add_argument("--input", type=str, required=True)  # 指定输入图像或图像目录的路径，此参数是必需的
    parser.add_argument("--steps", required=True, type=int)  # 指定采样步数，此参数是必需的
    parser.add_argument("--sr_scale", type=float, default=1)  # 指定超分辨率的比例，默认为1
    parser.add_argument("--repeat_times", type=int, default=1)  # 指定重复处理次数，默认为1次

    parser.add_argument("--disable_preprocess_model", action="store_true")  # 如果指定，将不使用预处理模型
    
    # patch-based sampling
    parser.add_argument("--tiled", action="store_true")  # 如果指定，将使用基于补丁的采样方法
    parser.add_argument("--tile_size", type=int, default=512)  # 指定采样时使用的补丁大小，默认为512
    parser.add_argument("--tile_stride", type=int, default=256)  # 指定采样补丁的滑动步长，默认为256
    
    # latent image guidance
    parser.add_argument("--use_guidance", action="store_true")  # 如果指定，将使用潜在图像引导
    parser.add_argument("--g_scale", type=float, default=0.0)  # 指定引导的比例，默认为0.0
    parser.add_argument("--g_t_start", type=int, default=1001)  # 指定开始使用引导的时间步，默认为1001
    parser.add_argument("--g_t_stop", type=int, default=-1)  # 指定停止使用引导的时间步，默认为-1（表示不停止）
    parser.add_argument("--g_space", type=str, default="latent")  # 指定引导的空间，默认为"latent"
    parser.add_argument("--g_repeat", type=int, default=5)  # 指定引导重复的次数，默认为5次
    
    parser.add_argument("--color_fix_type", type=str, default="wavelet", choices=["wavelet", "adain", "none"])  # 指定颜色修正类型，默认为"wavelet"
    parser.add_argument("--output", type=str, required=True)  # 指定输出目录的路径，此参数是必需的
    parser.add_argument("--show_lq", action="store_true")  # 如果指定，将展示低质量输入图像
    parser.add_argument("--skip_if_exist", action="store_true")  # 如果指定，当输出文件已存在时，将跳过处理
    
    parser.add_argument("--seed", type=int, default=231)  # 指定随机种子，默认为231
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda", "mps"])  # 指定使用的设备类型，默认为"cuda"
    
    return parser.parse_args()

def check_device(device):
    # 检查指定的设备类型，并在必要时禁用或调整
    if device == "cuda":
        # check if CUDA is available
        if not torch.cuda.is_available():
            print("CUDA not available because the current PyTorch install was not "
                "built with CUDA enabled.")
            device = "cpu"
    else:
        # xformers only support CUDA. Disable xformers when using cpu or mps.
        disable_xformers()
        if device == "mps":
            # check if MPS is available
            if not torch.backends.mps.is_available():
                if not torch.backends.mps.is_built():
                    print("MPS not available because the current PyTorch install was not "
                        "built with MPS enabled.")
                    device = "cpu"
                else:
                    print("MPS not available because the current MacOS version is not 12.3+ "
                        "and/or you do not have an MPS-enabled device on this machine.")
                    device = "cpu"
    print(f'using device {device}')
    return device

def main() -> None:
    # 主函数，执行整个图像处理流程
    args = parse_args() # 解析命令行参数
    pl.seed_everything(args.seed) # 设置随机种子以确保结果的可重复性
    
    args.device = check_device(args.device) # 检查并设置计算设备（如CPU, CUDA）
    
    # 从配置文件加载模型
    model: ControlLDM = instantiate_from_config(OmegaConf.load(args.config))
    # 加载模型的状态字典
    load_state_dict(model, torch.load(args.ckpt, map_location="cpu"), strict=True)
    
    # reload preprocess model if specified
    # 如果指定了重新加载SwinIR模型，则进行相应的处理
    if args.reload_swinir:
        if not hasattr(model, "preprocess_model"):
            raise ValueError(f"model don't have a preprocess model.")
        print(f"reload swinir model from {args.swinir_ckpt}")
        load_state_dict(model.preprocess_model, torch.load(args.swinir_ckpt, map_location="cpu"), strict=True)
    
    model.freeze()   # 冻结模型，防止在推理过程中发生任何参数更改
    model.to(args.device)   # 将模型移到指定的设备
    
    assert os.path.isdir(args.input)   # 确保输入路径是一个目录
    
    # 遍历输入目录中的所有图像文件
    for file_path in list_image_files(args.input, follow_links=True):
        lq = Image.open(file_path).convert("RGB")   # 打开图像并转换为RGB格式
        # 如果指定了超分辨率比例不等于1，重新调整图像尺寸
        if args.sr_scale != 1:
            lq = lq.resize(
                tuple(math.ceil(x * args.sr_scale) for x in lq.size),
                Image.BICUBIC
            )
        
        # 根据是否采用补丁策略来调整图像大小
        if not args.tiled:
            lq_resized = auto_resize(lq, 512)
        else:
            lq_resized = auto_resize(lq, args.tile_size)
        x = pad(np.array(lq_resized), scale=64)   # 对调整大小后的图像进行填充处理
        
        # 重复处理每个图像多次（如果指定了重复次数）
        for i in range(args.repeat_times):
            # 构建保存路径
            save_path = os.path.join(args.output, os.path.relpath(file_path, args.input))
            parent_path, stem, _ = get_file_name_parts(save_path)
            save_path = os.path.join(parent_path, f"{stem}_{i}.png")
            
            # 检查输出文件是否已存在，根据参数选择跳过或报错
            if os.path.exists(save_path):
                if args.skip_if_exist:
                    print(f"skip {save_path}")
                    continue
                else:
                    raise RuntimeError(f"{save_path} already exist")
            os.makedirs(parent_path, exist_ok=True)   # 创建输出目录
            
            # initialize latent image guidance
            # 初始化潜在图像引导（如果指定了引导）
            if args.use_guidance:
                cond_fn = MSEGuidance(
                    scale=args.g_scale, t_start=args.g_t_start, t_stop=args.g_t_stop,
                    space=args.g_space, repeat=args.g_repeat
                )
            else:
                cond_fn = None
            
            # 处理图像
            preds, stage1_preds = process(
                model, [x], steps=args.steps,
                strength=1,
                color_fix_type=args.color_fix_type,
                disable_preprocess_model=args.disable_preprocess_model,
                cond_fn=cond_fn,
                tiled=args.tiled, tile_size=args.tile_size, tile_stride=args.tile_stride
            )
            pred, stage1_pred = preds[0], stage1_preds[0]
            
            # remove padding
            # 移除填充
            pred = pred[:lq_resized.height, :lq_resized.width, :]
            stage1_pred = stage1_pred[:lq_resized.height, :lq_resized.width, :]
            
            if args.show_lq:
                # 如果指定了显示低质量图像，将处理后的图像与原始图像并排保存
                pred = np.array(Image.fromarray(pred).resize(lq.size, Image.LANCZOS))
                stage1_pred = np.array(Image.fromarray(stage1_pred).resize(lq.size, Image.LANCZOS))
                lq = np.array(lq)
                images = [lq, pred] if args.disable_preprocess_model else [lq, stage1_pred, pred]
                Image.fromarray(np.concatenate(images, axis=1)).save(save_path)
            else:
                # 否则只保存处理后的图像
                Image.fromarray(pred).resize(lq.size, Image.LANCZOS).save(save_path)
            print(f"save to {save_path}")

if __name__ == "__main__":
    main()
