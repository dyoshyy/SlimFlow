import os

import torch
from torchvision.utils import save_image

import reflow.datasets as datasets
from configs.rectified_flow.cifar10_rf_gaussian import get_config
from models import ddpm, ncsnpp, ncsnv2  # noqa: F401
from models import utils as mutils
from models.ema import ExponentialMovingAverage
from reflow import RectifiedFlow
from reflow import losses as losses
from reflow import sampling as sampling
from reflow.utils import restore_checkpoint

join = os.path.join


def generate_images(
    config,
    noise=None,
    batch_size=16,
    num_steps=100,
    ode_solver="euler",
    class_labels=None,
    device=None,
):
    """
    初期ノイズから画像を生成する関数（ノイズを外部から渡せるようにカスタマイズ）

    Args:
        config: 設定オブジェクト
        model_path: モデルチェックポイントへのパス
        noise: 初期ノイズ（None の場合は内部で生成）
        batch_size: バッチサイズ（noise引数が渡された場合は無視される）
        num_steps: サンプリングステップ数（eulerソルバーの場合）
        ode_solver: ODEソルバー（'rk45'または'euler'）
        class_labels: クラスラベル（条件付き生成の場合）
        device: 計算デバイス（Noneの場合は自動選択）

    Returns:
        生成された画像（バッチ）
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "ckpts/FFHQ_2_flow_200000edm_flip_warmup_300000_30m_checkpoint_15.pth"

    # モデル初期化
    score_model = (
        mutils.create_model(config)
        if config.model.name != "DhariwalUNet"
        else mutils.create_model_edm(config)
    )
    score_model.to(device)
    ema = ExponentialMovingAverage(
        score_model.parameters(), decay=config.model.ema_rate
    )
    optimizer = losses.get_optimizer(config, score_model.parameters())

    # チェックポイントロード
    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)
    state = restore_checkpoint(model_path, state, device=device)
    print(f"モデルを {model_path} から読み込みました")
    ema.copy_to(score_model.parameters())
    score_model.eval()

    # RectifiedFlow初期化
    flow = RectifiedFlow(model=score_model, ema_model=ema, cfg=config)
    flow.model.eval()

    # スケーラー設定
    # scaler = datasets.get_data_scaler(config)
    inverse_scaler = datasets.get_data_inverse_scaler(config)

    # ノイズ処理
    if noise is None:
        # ノイズが提供されていなければ生成
        sampling_shape = (
            batch_size,
            config.data.num_channels,
            config.data.image_size,
            config.data.image_size,
        )
        z0 = flow.get_z0(torch.zeros(sampling_shape, device=device), train=False).to(
            device
        )
    else:
        # 外部から渡されたノイズを使用
        z0 = noise.to(device)
        sampling_shape = z0.shape

    # ODEサンプラー設定
    flow.use_ode_sampler = ode_solver
    if ode_solver == "euler":
        flow.sample_N = num_steps

    # サンプリング実行
    sampling_fn = sampling.get_flow_sampler(
        flow, sampling_shape, inverse_scaler, clip_denoised=True, device=device
    )

    with torch.no_grad():
        x, nfe = sampling_fn(score_model, z=z0, label=class_labels)

    return x, nfe


def main():
    config = get_config()
    config.eval.batch_size = 64
    config.model.nf = 128
    config.model.num_res_blocks = 2
    config.data.image_size = 64
    config.model.ch_mult = (1, 2, 2)
    config.sampling_dir = "./ckpts"
    config.ode_solver = "euler"

    # 初期化
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    noise = torch.randn(
        8, config.data.num_channels, config.data.image_size, config.data.image_size
    ).to(device)
    images_from_custom, _ = generate_images(
        config,
        batch_size=config.eval.batch_size,
        noise=noise,
        num_steps=1,
        ode_solver=config.ode_solver,
    )
    save_image(
        images_from_custom,
        os.path.join(config.sampling_dir, "custom_noise_samples.png"),
        nrow=4,
    )


if __name__ == "__main__":
    main()
