"""
# RoPE 시각화 노트북

이 노트북은 Hugging Face `transformers`가 제공하는 RoPE 초기화 유틸을 사용해서
`position`(가로축)과 `hidden dim`(세로축) 기준의 heatmap을 그립니다.

리팩토링 목표

- 기본 RoPE(`default`)만 바로 실행 가능하게 제공
- 다른 RoPE 변형으로 바꾸기 쉽도록 구조를 정리
- 함수 내부에 함수를 정의하지 않음
- 유사 기능을 묶어서 재사용 가능하게 구성
"""

import math
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Dict, Optional, Tuple, List

import matplotlib.pyplot as plt
import torch
from matplotlib.gridspec import GridSpecFromSubplotSpec
from transformers import ROPE_INIT_FUNCTIONS


@dataclass
class RopeConfig:
    rope_theta: float = 10000.0
    hidden_size: int = 4096
    num_attention_heads: int = 32
    head_dim: Optional[int] = None
    partial_rotary_factor: float = 1.0
    max_position_embeddings: int = 8192

    # transformers의 RoPE 스케일링 설정과 동일한 형태의 dict를 넣습니다.
    # 기본은 default 입니다.
    rope_scaling: Dict[str, Any] = field(default_factory=lambda: {"rope_type": "default"})


@dataclass
class RopeMethod:
    name: str
    rope_type: str
    cfg: RopeConfig
    # 일부 rope_type은 seq_len을 요구합니다. 기본(default)은 보통 None이어도 됩니다.
    seq_len: Optional[int] = None


## theta, cos, sin
def build_position_emb(
    inv_freq: torch.Tensor,  #
    L: int,
    attention_factor: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    # inv_freq = 1 / freq (base)
    # freq = 1 / period
    # inv_freq == period (if period larger, required time for one wave (period) becomes larger...)
    # however large period means that angular speed become smaller (paradox)
    inv_freq_expanded = inv_freq[:, None].float()  # (D/2, 1)
    position_ids = torch.arange(L, device=device, dtype=dtype)[None, :].float()  # (1, L)

    phase = (inv_freq_expanded @ position_ids).transpose(0, 1)  # (D/2, L) -> (L, D/2)
    phase = torch.cat([phase, phase], dim=-1)  #  (L, D)
    return phase, phase.cos() * attention_factor, phase.sin() * attention_factor


def postprocess_theta(theta: torch.Tensor, mode: str) -> torch.Tensor:
    # mode:
    # - "raw": 그대로
    # - "phase_0_2pi": 0 이상 2파이 미만으로 mod
    # - "phase_negpi_pi": -파이 초과 파이 이하 범위로 매핑
    if mode == "raw":
        return theta

    two_pi = 2.0 * math.pi
    phase = torch.remainder(theta, two_pi)

    if mode == "phase_0_2pi":
        return phase

    if mode == "phase_negpi_pi":
        # [0, 2pi) -> (-pi, pi]
        return (phase + math.pi) % two_pi - math.pi

    raise ValueError(f"알 수 없는 theta mode: {mode}")


def restore_to_full_dim(
    rotary_mat: torch.Tensor,
    *,
    head_dim: int,
    rotary_dim: int,
    kind: str,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    partial 외의 부분 0도 강제 입력 (시각화를 위해)
    - cos: 1
    - sin: 0
    - theta: 0
    """

    L = int(rotary_mat.shape[0])

    if rotary_dim == head_dim:
        return rotary_mat

    if kind == "cos":
        full = torch.ones((L, head_dim), device=device, dtype=dtype)
    elif kind == "sin":
        full = torch.zeros((L, head_dim), device=device, dtype=dtype)
    elif kind == "theta":
        full = torch.zeros((L, head_dim), device=device, dtype=dtype)
    else:
        raise ValueError(f"알 수 없는 kind: {kind}")

    full[:, :rotary_dim] = rotary_mat[:, :rotary_dim]
    return full


## 4. heatmap 그리기
"""
matrix: (L, D) >> visualize: (D, L)

"""


def compute_color_limits(
    mats: Dict[str, torch.Tensor],
    *,
    symmetric: bool,
) -> Tuple[float, float]:
    vmin = min(float(m.min().item()) for m in mats.values())
    vmax = max(float(m.max().item()) for m in mats.values())
    if symmetric:
        v = max(abs(vmin), abs(vmax))
        return -v, v
    return vmin, vmax


def plot_heatmap_grid_with_invfreq(
    mats: Dict[str, torch.Tensor],
    inv_freqs: Dict[str, torch.Tensor],
    *,
    title: str,
    vmin: float,
    vmax: float,
    cmap: str = "viridis",
    grid_cols: int = 3,
    origin: str = "lower",
    invfreq_log_x: bool = True,
) -> None:
    """
    mats[name]: (L, D)  -> visualize: (D, L) with imshow
    inv_freqs[name]: (D,) (mats의 D와 정렬되도록 준비되어 있어야 함)
    우측 inv_freq plot은 y축을 dim index로 두어 heatmap의 y축과 정렬합니다.
    """

    names = list(mats.keys())
    n = len(names)
    cols = max(1, int(grid_cols))
    rows = (n + cols - 1) // cols

    # figure를 직접 만들고 outer gridspec을 사용합니다.
    fig = plt.figure(figsize=(cols * 7.0, rows * 4.2), constrained_layout=True)
    outer = fig.add_gridspec(rows, cols)

    im = None
    for i, name in enumerate(names):
        r = i // cols
        c = i % cols

        # 각 패널을 (heatmap | inv_freq) 1x2로 분할
        sub = GridSpecFromSubplotSpec(
            1,
            2,
            subplot_spec=outer[r, c],
            width_ratios=[4.5, 1.5],
            wspace=0.08,
        )
        ax_hm = fig.add_subplot(sub[0, 0])
        ax_fr = fig.add_subplot(sub[0, 1], sharey=ax_hm)

        # --- heatmap ---
        mat = mats[name].T  # (D, L)
        im = ax_hm.imshow(
            mat.numpy(),
            aspect="auto",
            origin=origin,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
        )
        ax_hm.set_title(name, fontsize=10)
        ax_hm.set_xlabel("position")
        ax_hm.set_ylabel("hidden dim")

        # --- inv_freq plot (우측) ---
        f = inv_freqs.get(name, None)
        if f is None:
            ax_fr.set_title("inv_freq (missing)", fontsize=9)
            ax_fr.axis("off")
            continue

        f = f.detach().cpu().float()
        D = int(f.numel())
        y = torch.arange(D, dtype=torch.float32)

        # log 스케일이면 0 제거 필요
        mask = f > 0
        fx = f[mask].numpy()
        fy = y[mask].numpy()

        ax_fr.plot(fx, fy, linewidth=1.0)
        ax_fr.set_xlabel("inv_freq", fontsize=8)
        ax_fr.tick_params(axis="x", labelsize=8)
        ax_fr.tick_params(axis="y", labelleft=False)  # y 라벨은 heatmap에만

        if invfreq_log_x:
            ax_fr.set_xscale("log")
            ax_fr.grid(True, which="both", axis="x", alpha=0.25)
        else:
            ax_fr.grid(True, axis="x", alpha=0.25)

    fig.suptitle(title, fontsize=12)

    # 공용 컬러바
    if im is not None:
        fig.colorbar(im, ax=fig.axes, shrink=0.85)

    plt.show()


def plot_inv_freqs_all(
    inv_freqs: Dict[str, torch.Tensor],
    *,
    title: str = "inv_freqs (all methods)",
    log_y: bool = True,
) -> None:
    """
    각 method의 inv_freq를 한 figure에 겹쳐서 표시합니다.
    x축: dim index, y축: inv_freq
    """
    plt.figure(figsize=(8.5, 4.5))

    for name, f in inv_freqs.items():
        f = f.detach().cpu().float()
        x = torch.arange(int(f.numel()), dtype=torch.float32)

        mask = f > 0
        xx = x[mask].numpy()
        yy = f[mask].numpy()

        if log_y:
            plt.semilogy(xx, yy, label=name, linewidth=1.2)
        else:
            plt.plot(xx, yy, label=name, linewidth=1.2)

    plt.title(title)
    plt.xlabel("dim index")
    plt.ylabel("inv_freq")
    plt.grid(True, which="both", alpha=0.25)
    plt.legend(fontsize=8)
    plt.show()


## 5. RoPE 방법별 행렬 만들기
"""
현재는 `default`만 준비합니다.

다른 방법으로 바꾸려면 `methods` 리스트에 `RopeMethod`를 추가하면 됩니다.
추가 시에도 HF `ROPE_INIT_FUNCTIONS`를 그대로 사용합니다.
"""


def build_rotary_matrices(
    methods: List[RopeMethod],
    *,
    kind: str,
    L: int,
    head_dim_show: int,
    rotary_dim_show: int,
    theta_mode: str,
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[str, torch.Tensor]:
    matrices: Dict[str, torch.Tensor] = {}
    inv_freqs: Dict[str, torch.Tensor] = {}
    # Actually, it is not "inversed" frequency in physics.
    # inversed frequency in physics equals period, but it is contradictory with its purpose (high value means fast angular speed)
    # Considering its use, "angular_speed" is required.

    for m in methods:
        inv_freq, attn_factor = ROPE_INIT_FUNCTIONS[m.rope_type](m.cfg, device, seq_len=m.seq_len)  # <= m.cfg
        inv_freq = inv_freq.detach().to("cpu").float()
        attn_factor = float(attn_factor)

        theta, cos, sin = build_position_emb(inv_freq, L=L, attention_factor=attn_factor, device=device, dtype=dtype)

        if kind == "theta":
            theta = postprocess_theta(theta, mode=theta_mode)
            rotary_mat = theta
        elif kind == "cos":
            rotary_mat = cos
        elif kind == "sin":
            rotary_mat = sin
        else:
            raise ValueError(f"알 수 없는 kind: {kind}")

        full = restore_to_full_dim(
            rotary_mat,
            head_dim=head_dim_show // 2,
            rotary_dim=rotary_dim_show,
            kind=kind,
            device=device,
            dtype=dtype,
        )
        matrices[m.name] = full.detach().cpu()

        # inv_freq도 heatmap의 D와 정렬되도록 패딩(여기서는 D = head_dim_show//2)
        D = head_dim_show // 2
        freq_full = torch.zeros((D,), dtype=torch.float32)
        n_fill = min(int(rotary_dim_show), int(inv_freq.numel()), D)
        freq_full[:n_fill] = inv_freq[:n_fill]
        inv_freqs[m.name] = freq_full

    return matrices, inv_freqs


def plot_kind(
    methods: List[RopeMethod],
    *,
    kind: str,
    L: int,
    head_dim: int,
    rotary_dim: int,
    rope_theta: float,
    partial_rotary_factor: float,
    theta_mode: str,
    global_scale: bool = True,
    symmetric_scale: bool = True,
    cmap: str = "viridis",
    origin: str = "lower",
    grid_cols: int = 3,
    device: torch.device,
    dtype: torch.dtype,
    plot_all_invfreq: bool = False,  # 추가
) -> None:
    matrices, inv_freqs = build_rotary_matrices(
        methods,
        kind=kind,
        L=L,
        head_dim_show=head_dim,
        rotary_dim_show=rotary_dim,
        theta_mode=theta_mode,
        device=device,
        dtype=dtype,
    )

    symmetric = bool(symmetric_scale) if kind in ("cos", "sin") else False
    vmin, vmax = compute_color_limits(matrices, symmetric=symmetric)

    info = (
        f"head_dim={head_dim}, "
        f"rotary_dim={rotary_dim}, "
        f"L={L}, rope_theta={rope_theta}, partial_rotary_factor={partial_rotary_factor}"
    )
    title = f"{kind} | {info}"

    # (1) heatmap 우측에 inv_freq를 함께 표시
    plot_heatmap_grid_with_invfreq(
        mats=matrices,
        inv_freqs=inv_freqs,
        title=title,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        grid_cols=grid_cols,
        origin=origin,
        invfreq_log_x=True,  # 우측 패널은 log x 추천
    )

    # (2) 모든 method inv_freqs를 별도 figure에 표시
    if plot_all_invfreq:
        plot_inv_freqs_all(inv_freqs, title="inv_freqs (all methods)", log_y=True)


## 6. 실행 파라미터
"""
아래 셀만 수정하면 됩니다.
지금은 기본 RoPE(default)만 실행합니다.
"""
# 실행 파라미터
device = torch.device("cpu")  # "cuda"도 가능
dtype = torch.float32

# 모델 형상
rope_theta = 10000.0
hidden_size = 4096
num_attention_heads = 32
head_dim = None  # None이면 hidden_size / num_heads
partial_rotary_factor = 0.9  # 0 <= f <= 1

# 시각화 범위
L = 8192 * 4

# theta 표시 방식
theta_mode = "phase_negpi_pi"  # "raw", "phase_0_2pi", "phase_negpi_pi"

# plot 옵션
cmap = "viridis"
origin = "lower"
grid_cols = 2


assert hidden_size % num_attention_heads == 0, ValueError(
    f"hidden_size({hidden_size})가 num_attention_heads({num_attention_heads})로 나누어 떨어지지 않습니다."
)
assert 0 <= partial_rotary_factor <= 1

# head에 따라 자동 계산
resolved_head_dim = head_dim if head_dim is not None else int(hidden_size // num_attention_heads)

rotary_dim = int(resolved_head_dim * float(partial_rotary_factor))
rotary_dim = rotary_dim - (rotary_dim % 2)  # 짝수로 내림 (중요)
inv_freq_len = rotary_dim // 2  # inv_freq의 길이
resolved_rotary_dim = inv_freq_len
resolved_partial_rotary_factor = rotary_dim / resolved_head_dim


print(f"head_dim={resolved_head_dim}, inv_freq_len={resolved_rotary_dim}, rotary_dim_even={rotary_dim}")


# 기본 config
cfg_partial = partial(
    RopeConfig,
    rope_theta=rope_theta,
    hidden_size=hidden_size,
    num_attention_heads=num_attention_heads,
    head_dim=resolved_head_dim,
    partial_rotary_factor=resolved_partial_rotary_factor,
    max_position_embeddings=L,  # 기본은 크게 중요하지 않지만, 형태를 맞춥니다.)
)


def _build_complete_config(cfg_partial: partial[RopeConfig], name: str, **kwargs) -> list[RopeConfig]:
    match name:
        case "default":
            cfg = cfg_partial(rope_scaling={"rope_type": name})
        case "linear":
            cfg = cfg_partial(rope_scaling={"rope_type": name, "factor": kwargs.get("factor")})
        case "dynamic":
            cfg = cfg_partial(rope_scaling={"rope_type": name, "factor": kwargs.get("factor")})
        case "yarn":
            cfg = cfg_partial(
                rope_scaling={
                    "rope_type": name,
                    "factor": kwargs.get("factor"),
                    "beta_fast": kwargs.get("yarn_beta_fast"),
                    "beta_slow": kwargs.get("yarn_beta_slow"),
                    "truncate": True,
                }
            )
        case "longrope":
            dim_half = kwargs.get("resolved_rotary_dim")
            short_factor = [1.0] * dim_half
            long_factor = torch.linspace(1.0, kwargs.get("longrope_demo_max"), dim_half).tolist()
            cfg = cfg_partial(
                rope_scaling={
                    "rope_type": name,
                    "factor": kwargs.get("factor"),
                    "short_factor": short_factor,
                    "long_factor": long_factor,
                    "original_max_position_embeddings": kwargs.get("original_max_position_embeddings"),
                }
            )
        case "llama3":
            cfg = cfg_partial(
                rope_scaling={
                    "rope_type": name,
                    "factor": kwargs.get("factor"),
                    "low_freq_factor": kwargs.get("llama3_low_freq_factor"),
                    "high_freq_factor": kwargs.get("llama3_high_freq_factor"),
                    "original_max_position_embeddings": kwargs.get("original_max_position_embeddings"),
                }
            )
    return cfg


# rope_scaling = ({"rope_type": rope_type},)
methods: List[RopeMethod] = [
    RopeMethod(
        name=rope_type,
        rope_type=rope_type,
        cfg=_build_complete_config(
            cfg_partial,
            name=rope_type,
            factor=4.0,
            yarn_beta_fast=32.0,
            yarn_beta_slow=1.0,
            longrope_demo_max=4.0,
            llama3_factor=8.0,
            llama3_low_freq_factor=1.0,
            llama3_high_freq_factor=4.0,
            original_max_position_embeddings=L,
            resolved_rotary_dim=resolved_rotary_dim,
        ),
        seq_len=L,
    )
    for rope_type in ["default", "linear", "dynamic", "yarn", "longrope", "llama3"]
]

print("사용 가능한 rope_type:", ", ".join(sorted(ROPE_INIT_FUNCTIONS.keys())))
print(f"head_dim={resolved_head_dim}, rotary_dim={resolved_rotary_dim}")

# cos, sin, theta를 각각 heatmap으로 확인합니다.
plot_kind(
    methods,
    kind="cos",
    L=L,
    head_dim=resolved_head_dim,
    rotary_dim=resolved_rotary_dim,
    rope_theta=rope_theta,
    partial_rotary_factor=partial_rotary_factor,
    theta_mode=theta_mode,
    global_scale=True,
    symmetric_scale=True,
    cmap=cmap,
    origin=origin,
    grid_cols=grid_cols,
    device=device,
    dtype=dtype,
    plot_all_invfreq=True,
)

plot_kind(
    methods,
    kind="sin",
    L=L,
    head_dim=resolved_head_dim,
    rotary_dim=resolved_rotary_dim,
    rope_theta=rope_theta,
    partial_rotary_factor=partial_rotary_factor,
    theta_mode=theta_mode,
    global_scale=True,
    symmetric_scale=True,
    cmap=cmap,
    origin=origin,
    grid_cols=grid_cols,
    device=device,
    dtype=dtype,
    plot_all_invfreq=False,
)
