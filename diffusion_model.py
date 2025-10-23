import math
import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch.nn.functional as F
from torch import nn
from PIL import Image
# from tqdm import tqdm
from model.utils.unet import UNet, UNetCond, UNetCondDeep
from model.utils.diffuser import Diffuser
import wandb
import copy
# from wandb import Alertlevel

from diffusers import UNet2DModel, UNet2DConditionModel, DDPMScheduler, DDPMPipeline
from diffusers.optimization import get_cosine_schedule_with_warmup

from dataclasses import dataclass

from accelerate import Accelerator
from huggingface_hub import create_repo, upload_folder

from tqdm.auto import tqdm
import os

# CUDA メモリ管理の設定
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

@dataclass
class TrainingConfig:
    image_size = 64  # 生成する画像の解像度
    train_batch_size = 64  # バッチサイズをさらに小さく
    eval_batch_size = 20  # 評価時にサンプリングする画像数
    num_epochs = 100
    gradient_accumulation_steps = 1  # gradient_accumulation_steps を増やす
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 10
    save_model_epochs = 10
    mixed_precision = 'fp16'  # `no` は float32、`fp16` は自動混合精度
    output_dir = 'checkpoint/diffuser-ddpm'  # モデルをローカルおよび HF Hub 上に保存するディレクトリ

    push_to_hub = False  # 保存したモデルを HF Hub にアップロードするかどうか
    hub_model_id = None  # HF Hub にアップロードする際のモデル ID（例: "username/model-name"）
    hub_private_repo = False
    project_name = "diffuser-ddpm"  # プロジェクト名（hub_model_id が指定されていない場合に使用）
    overwrite_output_dir = True  # 再実行時に既存の出力ディレクトリを上書きするか
    seed = 0
    run_name = "ddpm-cifar10-64"  # Wandb の run 名

config = TrainingConfig()

num_timesteps = 1000

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

def show_images(imgs, rows=2, cols=10, labels=None):
    # imgs = imgs.permute(0, 2, 3, 1).cpu().numpy()  # Convert to (N, H, W, C) and move to CPU
    fig = plt.figure(figsize=(cols, rows))
    i = 0
    for r in range(rows):
        for c in range(cols):
            fig.add_subplot(rows, cols, i + 1)
            plt.imshow(imgs[i])
            plt.title(f'{labels[i].item()}' if labels is not None else '')
            plt.axis('off')
            i += 1
    fig.tight_layout()
    return fig

preprocess = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])
# dataset = torchvision.datasets.CIFAR10(root='./data', transform=preprocess, download=True)
dataset = torchvision.datasets.ImageFolder(root='~/../host_files/cifar10-64/train', transform=preprocess)
train_dataloader = DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=2)

# diffuser = Diffuser(num_timesteps=num_timeseteps, device=device)
# model = UNet()
# model = UNetCondDeep(in_ch=3, num_labels=10)
# model = model.to(device)

# model = UNet2DModel(
model = UNet2DConditionModel(
    sample_size=config.image_size,  # the target image resolution
    in_channels=3,  # the number of input channels, 3 for RGB images
    out_channels=3,  # the number of output channels
    layers_per_block=2,
    block_out_channels=(128, 128, 256, 256),
    down_block_types=(
        "DownBlock2D",  # a regular ResNet downsampling block
        "DownBlock2D",
        "AttnDownBlock2D",  # a ResNet downsampling block with attention
        "DownBlock2D",
    ),
    up_block_types=(
        "UpBlock2D",  # a ResNet upsampling block with attention
        "AttnUpBlock2D",
        "UpBlock2D",  # a regular ResNet upsampling block
        "UpBlock2D",
    ),
    mid_block_type="UNetMidBlock2D",  # the mid-block type, a ResNet block with attention
    # time_embedding_type="positional",  # use positional embeddings for time steps
    num_class_embeds=10,  # number of classes for class-conditional generation
    class_embed_type="timestep"  # use timestep embeddings for class labels
)
sample_image = dataset[0][0].unsqueeze(0)
# print('入力形状:', sample_image.shape)
# print('出力形状:', model(sample_image, timestep=0)["sample"].shape)

noise_scheduler = DDPMScheduler(num_train_timesteps=num_timesteps)

optimizer = AdamW(model.parameters(), lr=config.learning_rate)

lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=config.num_epochs * len(train_dataloader)
)

def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i%cols*w, i//cols*h))
    return grid

def evaluate(config, epoch, pipeline, scheduler, model):
    # 各クラスごとに画像を生成 (1回の生成で済むようにクラスラベルを設定)
    num_classes = 10
    images_per_class = config.eval_batch_size // num_classes
    class_labels = []
    for class_idx in range(num_classes):
        class_labels.extend([class_idx] * images_per_class)
    class_labels = torch.tensor(class_labels, device=pipeline.device)

    image = torch.randn((config.eval_batch_size, 3, config.image_size, config.image_size)).to(pipeline.device)

    # images = pipeline(
    #     batch_size=config.eval_batch_size,
    #     generator=torch.manual_seed(config.seed),
    #     class_labels=class_labels
    # ).images

    # Show sampling progress with tqdm. Display current timestep as postfix.
    total_steps = len(scheduler.timesteps)
    for t in tqdm(scheduler.timesteps, desc=f"Sampling (epoch {epoch})", total=total_steps):
        with torch.no_grad():
            # ensure we can display the scalar timestep value
            try:
                t_val = int(t)
            except Exception:
                t_val = t
            # UNet2DConditionModel expects an `encoder_hidden_states` positional arg in
            # its forward signature. Provide it explicitly (None when using class_labels)
            noisy_resiudual = model(image, t, encoder_hidden_states=None, class_labels=class_labels)["sample"]
            prev_image = scheduler.step(noisy_resiudual, t, image).prev_sample
            image = prev_image
        # update postfix for tqdm (not strictly necessary but helpful)

    image = image.clamp(-1, 1)
    image = (image + 1) / 2  # -> [0,1]
    image = (image * 255).round().clamp(0, 255).to(torch.uint8)
    image_np = image.permute(0, 2, 3, 1).contiguous().cpu().numpy()
    images = [Image.fromarray(image_np[i]) for i in range(image_np.shape[0])]

    # 画像をグリッドに配置 (4x5 のグリッド、20枚)
    image_grid = make_grid(images, rows=4, cols=5)

    # 画像を保存
    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch+1:04d}.png")

    # Wandb に画像をログ
    wandb.log({"samples_grid": wandb.Image(image_grid, caption=f"Epoch {epoch+1} grid")}, step=epoch+1)

def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler):
    # Accelerator と Wandb ログの初期化
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="wandb"
    )
    if accelerator.is_main_process:
        if config.push_to_hub:
            repo_id = create_repo(
                repo_id=config.hub_model_id or config.project_name,
                exist_ok=True
            ).repo_id
        accelerator.init_trackers("DDPM_study", config={"name": config.run_name, "config": config.__dict__})

    # すべてを prepare する
    # 順序に特別な意味はありませんが、prepare に渡した順番でアンパックしてください。
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    global_step = 0

    # 学習ループ開始
    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            clean_images = batch[0]
            labels = batch[1]  # クラスラベルを取得
            # 画像に加えるノイズをサンプリング
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            bs = clean_images.shape[0]

            # 各画像に対してランダムなタイムステップをサンプリング
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device).long()

            # 各タイムステップに応じてクリーン画像にノイズを加える
            # （これがフォワード拡散プロセスです）
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with accelerator.accumulate(model):
                # ノイズ残差を予測
                # UNet2DConditionModel requires encoder_hidden_states positional arg; pass None
                noise_pred = model(noisy_images, timesteps, encoder_hidden_states=None, class_labels=labels)["sample"]
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        # 各エポック後にサンプル画像を評価用に生成してモデルを保存
        if accelerator.is_main_process:
            pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)

            if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                evaluate(config, epoch, pipeline, noise_scheduler, model)

            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                if config.push_to_hub:
                    pipeline.save_pretrained(config.output_dir)
                    upload_folder(
                        repo_id=repo_id,
                        folder_path=config.output_dir,
                        commit_message=f"epoch {epoch}",
                        ignore_patterns=["step_*", "epoch_*"],
                    )
                else:
                    pipeline.save_pretrained(config.output_dir)

train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler)

