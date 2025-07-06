import logging
import os
import json
import numpy as np
import torch
import torch.multiprocessing as mp
from tqdm import tqdm
from PIL import Image
import argparse
import yaml
import torch.nn.functional as F

from datasets.utils import normalize
from guided_diffusion import (
    DDIMSampler,
    G_DDIMSampler,
)
from guided_diffusion import dist_util
from guided_diffusion.script_util import (
    model_defaults,
    create_model,
    diffusion_defaults,
    create_gaussian_diffusion,
    select_args,
)
from metrics import LPIPS, PSNR, SSIM, Metric
from utils import save_grid, save_image, normalize_image
from utils.config import Config
from utils.logger import get_logger, logging_info
from utils.nn_utils import get_all_paths, set_random_seed
from utils.result_recorder import ResultRecorder
from utils.timer import Timer
from scripts.preprocess_crop import merge_patches

def prepare_model(algorithm, conf, device):
    # logging_info("Prepare model...")
    unet = create_model(**select_args(conf, model_defaults().keys()), conf=conf)
    SAMPLER_CLS = {
        "ddim": DDIMSampler,
        "g_ddim": G_DDIMSampler,
    }
    sampler_cls = SAMPLER_CLS[algorithm]
    sampler = create_gaussian_diffusion(
        **select_args(conf, diffusion_defaults().keys()),
        conf=conf,
        base_cls=sampler_cls,
    )

    logging_info(f"Loading model from {conf.model_path}...")
    unet.load_state_dict(
        dist_util.load_state_dict(
            os.path.expanduser(conf.model_path), map_location="cpu"
        ), strict=False
    )
    unet.to(device)
    if conf.use_fp16:
        unet.convert_to_fp16()
    unet.eval()
    return unet, sampler

def all_exist(paths):
    for p in paths:
        if not os.path.exists(p):
            return False
    return True

def main():
    config = Config(default_config_file="configs/config.yaml", use_argparse=True)
    config.show()

    # # 分配gpu，配置device
    gpu = config.get("gpu",0)
    print(f"指定的GPU: {gpu}")
    torch.cuda.set_device(gpu)
    device = torch.device(f"cuda:{gpu}")  # 字符串格式"cuda:0",中间必须是冒号，在 PyTorch 中，设备字符串的格式如下，使用CPU为 "cpu"；使用GPU为 "cuda:0"

    # 输出目录
    noised_img_name = os.path.splitext(os.path.basename(config.input))[0]
    outdir = os.path.join(config.output, noised_img_name)
    os.makedirs(outdir, exist_ok=True)

    ###################################################################################
    # prepare config, logger and recorder
    ###################################################################################
    subconfig_path = config.get("subconfig_path", "./configs/imagenet.yaml")
    subconfig = Config(default_config_file=subconfig_path, use_argparse=False)
    status = []

    subconfig.update(config.subconfig_updated_params) # 更新子配置文件的公共参数
    logging_info(f"subconf unpdate:{config.subconfig_updated_params}")
    subconfig.show()

    status.append(subconfig.algorithm)
    status.append(str(subconfig["ddim.denoise_step"]))
    status = '-'.join(status)
    logging_info(f"status:{status}")

    all_paths = get_all_paths(outdir,"denoise")
    subconfig.dump(all_paths["path_config"])
    get_logger(all_paths["path_log"], force_add_handler=True)
    recorder = ResultRecorder(
        path_record=all_paths["path_record"],
        initial_record=subconfig,
        use_git=subconfig.use_git,
    )
    set_random_seed(subconfig.seed, deterministic=False, no_torch=False, no_tf=True)

    ###################################################################################
    # prepare data
    ###################################################################################
    datas = []  # 定义一个空列表来存储分割图片、掩码和标签
    data_label = None # 去噪图片信息及分块位置

    # 获取输入的gt图像的路径
    if not os.path.isdir(config.input):  # 如果不是目录，即单张图片
        targets = [config.input]
        data_label_dir = os.path.dirname(config.input)
        for f in os.listdir(data_label_dir):
            if f.endswith(".json"):
                data_label_path = os.path.join(data_label_dir, f)
                with open(data_label_path, "r", encoding="utf-8") as f:
                    data_label = json.load(f)
    else:
        targets = [
            f for f in os.listdir(config.input) if not os.path.isdir(os.path.join(config.input, f))
        ]
        targets = [os.path.join(config.input, f) for f in targets]

    for target in targets:
        if target.endswith("json"):
            with open(target, "r", encoding="utf-8") as f:
                data_label = json.load(f)
    print(data_label)

    names = []  # 定义空列表来存储gt图像的名字
    gt_paths = []
    for target in targets:
        if target.lower().endswith((".jpg", ".jpeg", ".png")):
            # 载入初始图像
            logging_info(f"Processing {target}...")
            gt_paths.append(target)
            image_original = normalize(Image.open(target).convert("RGB"))  # 处理任意形状的图像
            logging_info(f"image_original shape: {image_original.shape}")
            name = os.path.splitext(os.path.basename(target))[0]
            names.append(name)

            class_labels = 0
            logging_info(f"input class labels: {class_labels}")

            data = (image_original, name, class_labels)
            datas.append(data)

    if config.use_denoise:
    # ##################################################################################
    # prepare model and device and metics loss
    # ##################################################################################
        logging_info("Prepare model...")

        unet, sampler = prepare_model(subconfig.algorithm, subconfig, device)

        def model_fn(x, t, y=None, gt=None, **kwargs):
            # assert y is not None
            """
            对于CelebA-HQ,Places数据集上预训练的Diffusion Model是无条件的（Unconditional），没有标签y
            对于ImageNet数据集上预训练的Diffusion Model是有条件的（Conditional），有标签y
            """
            return unet(x, t, y if subconfig.class_cond else None, gt=gt)

        cond_fn = None

        METRICS = {
            "lpips": Metric(LPIPS("alex", device)),
            "psnr": Metric(PSNR(), eval_type="max"),
            "ssim": Metric(SSIM(), eval_type="max"),
        }
        final_loss = []

        # ###################################################################################
        # # start sampling
        # ###################################################################################
        logging_info("Start sampling...")
        timer, num_image = Timer(), 0
        batch_size = subconfig.n_samples

        for i, d in enumerate(tqdm(datas)):
            image, image_name, class_id = d

            logging_info(f"Sampling '{image_name}'...")

            # prepare save dir
            outpath = os.path.join(outdir, image_name)
            os.makedirs(outpath, exist_ok=True)
            sample_dir = os.path.join(outpath, "samples")
            os.makedirs(sample_dir, exist_ok=True)
            base_count = len(os.listdir(sample_dir))
            if config.debug:
                grid_count = max(len(os.listdir(outpath)) - 8, 0)
            else:
                grid_count = max(len(os.listdir(outpath)) - 7, 0)


            # prepare batch data for processing
            batch = {
                "image": image.to(device),
            }
            model_kwargs = {
                "gt": batch["image"].repeat(batch_size, 1, 1, 1),
            }
            if subconfig.class_cond:
                if subconfig.cond_y is not None:
                    classes = torch.ones(batch_size, dtype=torch.long, device=device)
                    model_kwargs["y"] = classes * config.cond_y
                else:
                    classes = torch.full((batch_size,), class_id, device=device)
                    model_kwargs["y"] = classes

            shape = (batch_size, 3, subconfig.image_size, subconfig.image_size)

            all_metric_paths = [
                os.path.join(outpath, i + ".last")
                for i in (list(METRICS.keys()) + ["final_loss"])
            ]
            if subconfig.get("resume", False) and all_exist(all_metric_paths):
                for metric_name, metric in METRICS.items():
                    metric.dataset_scores += torch.load(
                        os.path.join(outpath, metric_name + ".last")
                    )
                logging_info("Results exists. Skip!")
            else:
                # sample images
                samples = []

                timer.start()

                result = sampler.p_sample_loop(
                    model_fn,
                    shape=shape,
                    model_kwargs=model_kwargs,
                    cond_fn=cond_fn,
                    device=device,
                    progress=True,
                    return_all=True,
                    conf=subconfig,
                    # sample_dir=outpath if subconfig["debug"] else None,
                    sample_dir=outpath,
                )
                timer.end()

                for metric in METRICS.values():
                    metric.update(result["sample"], batch["image"])

                if "loss" in result.keys() and result["loss"] is not None:
                    recorder.add_with_logging(
                        key=f"loss_{image_name}", value=result["loss"]
                    )
                    final_loss.append(result["loss"])
                else:
                    final_loss.append(None)

                inpainted = normalize_image(result["sample"]) # sample 在这里从[-1,1] normalize 到 [0,1]
                # logging_info(f"inpainted shape: {inpainted.shape}")
                samples.append(inpainted.detach().cpu())
                samples = torch.cat(samples)
                # print(f"samples shape: {samples.shape}")

                # save images
                # save gt images
                save_grid(
                    normalize_image(batch["image"]),
                    os.path.join(outpath, f"gt.png")
                )

                # save generations
                for sample in samples:
                    # print(f"sample shape: {sample.shape}")
                    save_image(
                        sample,
                        os.path.join(sample_dir, f"gen-{status}-{base_count:05}.png")
                    )
                    base_count += 1
                save_grid(
                    samples,
                    os.path.join(outpath, f"gen-{status}-{grid_count:04}.png"),
                    nrow=batch_size,
                )
                save_grid(
                    samples,
                    os.path.join(outdir, f"gen-{image_name}.png"),
                    nrow=batch_size,
                )
                # save metrics
                for metric_name, metric in METRICS.items():
                    torch.save(metric.dataset_scores[-subconfig.n_iter:], os.path.join(outpath, metric_name + ".last"))

                torch.save(
                    final_loss[-subconfig.n_iter:], os.path.join(outpath, "final_loss.last"))

                num_image += 1
                last_duration = timer.get_last_duration()
                logging_info(
                    "It takes %.3lf seconds for image %s"
                    % (float(last_duration), image_name)
                )

            # report batch scores
            for metric_name, metric in METRICS.items():
                recorder.add_with_logging(
                    key=f"{metric_name}_score_{image_name}",
                    value=metric.report_batch(),
                )

        # report over all results
        for metric_name, metric in METRICS.items():
            mean, colbest_mean = metric.report_all()
            recorder.add_with_logging(key=f"mean_{metric_name}", value=mean)
            recorder.add_with_logging(
                key=f"best_mean_{metric_name}", value=colbest_mean)
        if len(final_loss) > 0 and final_loss[0] is not None:
            recorder.add_with_logging(
                key="final_loss",
                value=np.mean(final_loss),
            )
        if num_image > 0:
            recorder.add_with_logging(
                key="mean time", value=timer.get_cumulative_duration() / num_image
            )

    # 将去噪后的patch合并
    files = [f for f in os.listdir(outdir) if f.lower().endswith(('.png','.jpg','.jpeg')) and not os.path.isdir(os.path.join(outdir,f))]
    # print(files)
    patch_paths = [os.path.join(outdir, f) for f in files]
    # print(patch_paths)
    denoised_patches = []
    for patch_path in patch_paths:
        patch = Image.open(patch_path)
        denoised_patches.append(patch)
    # print(len(denoised_patches))
    merged_save_path = os.path.join(outdir, f'{noised_img_name}-denoised.png')
    output = merge_patches(denoised_patches, data_label['pos'], data_label['image_size'])
    output.save(merged_save_path)

    logging_info(
        f"Samples are ready and waiting for you here: \n{outdir} \n"
    )
    recorder.end_recording()


if __name__ == "__main__":
    main()