# MglfLab

> 项目名称：**MglfLab**  
> Python 包名：`mglf_lab`  
> pip 安装包名：`mglf-lab`

基于 Isaaclab_Parkour 当中 teacher 阶段（阶段一）的 Unitree Go2 粗糙地形速度控制项目。策略不依赖 waypoint 或目标点，直接接收遥控速度指令，并利用局部高程扫描适应地形。

## 1. 功能与设计

- 控制指令为 `[vx, vy, wz]`：前后速度、横向速度和偏航角速度。
- 不使用 waypoint、目标位置或导航规划器。
- 高程扫描范围为 `1.6 m × 1.0 m`，分辨率为 `0.1 m`，共 187 个采样点，并随机器人 yaw 旋转。
- 使用 Isaac Lab 的 `ROUGH_TERRAINS_CFG`：上下楼梯、斜坡、离散障碍、箱体和随机粗糙面。
- 上下楼梯高度为 `0.08～0.28 m`，难度随课程等级逐渐提高。
- 使用温和的髋关节姿态惩罚抑制内八，同时保留横移和复杂落脚能力。
- 使用 Isaac Lab 原生 RSL-RL 流程，不依赖 `Isaaclab_Parkour` 的旧版自定义 runner。

有限的训练分布无法保证机器人通过任意未知地形。沟壑、窄桥、跳跃或松软地面等能力仍需加入对应的训练地形和奖励。

## 2. 目录结构

```text
MglfLab/
├── source/mglf_lab/tasks/
│   ├── go2_rough_env_cfg.py    # 环境、地形、观测、奖励和指令
│   ├── rsl_rl_ppo_cfg.py       # PPO 网络和优化参数
│   └── __init__.py             # Gymnasium 任务注册
├── scripts/
│   ├── train.py                # 训练和续训
│   ├── play.py                 # 多环境随机指令测试
│   ├── play_keyboard.py        # 单环境键盘遥控
│   └── cli_args.py             # 公共命令行参数
├── logs/                       # checkpoint 和 TensorBoard 日志
└── setup.py
```

## 3. 环境与依赖

已验证的组合：

- Ubuntu 24.04；
- Python 3.11；
- NVIDIA GPU 和可用的 CUDA 驱动；
- Isaac Sim 5.1；
- Isaac Lab 2.3.2；
- RSL-RL 3.1 系列；
- `torch`、`gymnasium`、`tensorboard`。

Isaac Lab 环境通常已经带有 `torch`、`gymnasium` 和 `rsl-rl-lib`。本项目不主动固定或升级这些核心依赖，避免破坏 Isaac Lab 与 Isaac Sim 的版本组合。

检查当前环境：

```bash
conda activate 你的环境
python --version
python -m pip show isaaclab isaaclab-rl rsl-rl-lib gymnasium torch tensorboard
```

如果只缺少 TensorBoard：

```bash
python -m pip install tensorboard
```

如果 `isaaclab`、`isaaclab_tasks` 或 `isaaclab_rl` 无法加载，应先按照对应版本的 Isaac Lab 安装方法修复基础环境，不建议随意升级 `torch`、Isaac Sim 或 RSL-RL。

## 4. 安装本项目

**前提：环境当中已经安装好了 isaacsim 和 isaaclab**

```bash
conda activate 你的环境
cd ../MglfLab
python -m pip install --no-build-isolation -e .
```

`-e` 是 editable 安装。修改 `source/mglf_lab/` 下的代码后通常无需重新安装。

检查三个入口：

```bash
python scripts/train.py --help
python scripts/play.py --help
python scripts/play_keyboard.py --help
```

注册的任务：

| 任务 | 用途 |
| --- | --- |
| `Go2-Rough-Teleop-v0` | 大规模并行训练 |
| `Go2-Rough-Teleop-Play-v0` | 播放、遥控和可视化 |

脚本已经设置默认任务，日常使用可以省略 `--task`。

## 5. 训练

### 5.1 冒烟测试

首次安装后先运行：

```bash
cd /home/mglf/rc/MglfLab
python scripts/train.py \
  --num_envs 32 \
  --max_iterations 2 \
  --seed 1 \
  --headless
```

能够完成 PPO 更新并生成 `model_*.pt`，说明环境、地形、高程扫描和训练器工作正常。

### 5.2 正式训练

16 GB 显存建议从 1024 个环境开始，再根据显存占用尝试 2048 或 4096(默认)：

```bash
python scripts/train.py \
  --num_envs 1024 \
  --max_iterations 5000 \
  --seed 1 \
  --run_name seed1 \
  --headless
```

日志目录为：

```text
logs/rsl_rl/go2_rough_teleop/日期_时间_run_name/
```

默认每 100 轮保存 checkpoint。`--max_iterations` 在新训练中表示训练轮数，在恢复训练中表示本次额外增加的轮数。

### 5.3 中断训练后继续训练

恢复权重、训练轮数和 Adam 优化器，并自动选择该 run 的最新 checkpoint：

```bash
python scripts/train.py \
  --num_envs 1024 \
  --resume \
  --load_run 2026-08-28_10-00-00_seed1 \
  --checkpoint model_5000.pt \
  --max_iterations 1000 \
  --headless
```

训练脚本中的 `--checkpoint` 是 `--load_run` 目录内的文件名或匹配表达式。

### 5.4 数值发散后恢复

如果出现 `inf`、`nan`、`value_function loss: inf` 或 `normal expects all elements of std >= 0.0`，应回退到最后一个健康 checkpoint，并丢弃旧优化器状态：

```bash
python scripts/train.py \
  --num_envs 1024 \
  --resume \
  --reset_optimizer \
  --load_run 2026-08-28_10-00-00_seed1 \
  --checkpoint model_5000.pt \
  --max_iterations 1000 \
  --run_name recovered \
  --headless
```

当前 PPO 使用固定学习率 `1e-4`，以降低长时间训练后的数值发散风险。

## 6. TensorBoard

训练时在另一个终端启动：

```bash
conda activate 你的环境
tensorboard \
  --logdir ../MglfLab/logs/rsl_rl/go2_rough_teleop \
  --port 6006
```

本机浏览器打开 `http://localhost:6006`。远程训练时可建立 SSH 转发：

```bash
ssh -L 6006:localhost:6006 用户名@远程主机地址
```

## 7. 多环境随机指令播放

`play.py` 同时观察多只 Go2，每个环境独立采样 `[vx, vy, wz]`，每 5～8 秒重新采样。

按 run 名称自动选择最新 checkpoint：

```bash
python scripts/play.py \
  --num_envs 50 \
  --load_run 2026-08-28_10-00-00_seed1
```

直接指定 checkpoint：

```bash
python scripts/play.py \
  --num_envs 50 \
  --checkpoint /home/mglf/rc/MglfLab/logs/rsl_rl/go2_rough_teleop/RUN目录/model_5000.pt
```

## 8. 键盘遥控播放

`play_keyboard.py` 默认创建一只 Go2：

```bash
python scripts/play_keyboard.py \
  --checkpoint /home/mglf/rc/MglfLab/logs/rsl_rl/go2_rough_teleop/RUN目录/model_5000.pt
```

启动后先点击 Isaac Sim 视口：

| 按键 | 功能 |
| --- | --- |
| `W` / `S` | 增加/减少前进速度 `vx` |
| `A` / `D` | 增加/减少横向速度 `vy` |
| `Q` / `E` | 增加/减少偏航角速度 `wz` |
| `Space` / `R` | 三个指令全部归零 |

按键改变的是持续指令，松开后不会自动归零。指令限制为：`vx` 为 `-1.5～1.5 m/s`，`vy` 为 `-0.8～0.8 m/s`，`wz` 为 `-1.5～1.5 rad/s`。

### 8.1 高程扫描可视化

```bash
python scripts/play_keyboard.py \
  --checkpoint /完整路径/model_5000.pt \
  --visualize_height_scan
```

视口会显示策略实际使用的 187 个 RayCaster 落点。关闭该选项可减少渲染开销。

### 8.2 修改键盘步长

```bash
python scripts/play_keyboard.py \
  --checkpoint /完整路径/model_5000.pt \
  --linear_step 0.1 \
  --yaw_step 0.1
```

## 9. 命令行参数

### 9.1 `train.py`

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--task` | `Go2-Rough-Teleop-v0` | 训练任务，通常省略 |
| `--num_envs` | 环境配置值 | 并行环境数量 |
| `--max_iterations` | `3000` | 新训练轮数或本次额外续训轮数 |
| `--seed` | `1` | 随机种子 |
| `--experiment_name` | `go2_rough_teleop` | `logs/rsl_rl/` 下的实验目录 |
| `--run_name` | 空 | 附加在时间戳后的名称 |
| `--resume` | 关闭 | 开启恢复训练 |
| `--reset_optimizer` | 关闭 | 恢复权重和轮数，但不恢复 Adam 状态 |
| `--load_run` | 自动匹配 | 要恢复的 run 目录名或表达式 |
| `--checkpoint` | 自动选择最新 | run 内的 checkpoint 文件名或表达式 |
| `--logger` | `tensorboard` | `tensorboard`、`wandb` 或 `neptune` |
| `--log_project_name` | 空 | W&B 或 Neptune 项目名 |
| `--device` | `cuda:0` | 仿真设备 |
| `--headless` | 关闭 | 无窗口运行，正式训练建议开启 |

### 9.2 `play.py`

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--task` | `Go2-Rough-Teleop-Play-v0` | 播放任务 |
| `--num_envs` | `50` | 同时创建的环境数量 |
| `--seed` | `1` | 地形和随机指令种子 |
| `--load_run` | 自动匹配 | 从实验目录选择 run |
| `--checkpoint` | 自动选择最新 | 可直接传 checkpoint 绝对路径 |
| `--real_time` | 默认开启 | 按真实时间限速 |
| `--no_real_time` | 关闭 | 取消真实时间限速 |
| `--device` | `cuda:0` | 仿真设备 |
| `--headless` | 关闭 | 无窗口播放，适合批量评估 |

### 9.3 `play_keyboard.py`

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--task` | `Go2-Rough-Teleop-Play-v0` | 键盘播放任务 |
| `--num_envs` | `1` | 环境数量，建议保持 1 |
| `--seed` | `1` | 地形随机种子 |
| `--load_run` | 自动匹配 | 从实验目录选择 run |
| `--checkpoint` | 自动选择最新 | 可直接传 checkpoint 绝对路径 |
| `--linear_step` | `0.2` | 每次按键改变的线速度，单位 `m/s` |
| `--yaw_step` | `0.2` | 每次按键改变的角速度，单位 `rad/s` |
| `--visualize_height_scan` | 关闭 | 显示 187 个高程扫描落点 |
| `--real_time` | 默认开启 | 按真实时间速度播放 |
| `--no_real_time` | 关闭 | 取消真实时间限速 |
| `--device` | `cuda:0` | 仿真设备 |

播放时，`--load_run` 和绝对路径 `--checkpoint` 二选一即可。直接传 checkpoint 时不需要指定 task 或 run。Isaac Lab 的 `AppLauncher` 还提供 `--enable_cameras`、`--livestream`、`--experience` 等高级选项，完整列表以各脚本的 `--help` 为准。

## 10. 观测与真实部署

策略观测共 235 维：

| 观测 | 维度 |
| --- | ---: |
| 基座线速度 | 3 |
| 基座角速度 | 3 |
| 投影重力 | 3 |
| 速度指令 `[vx, vy, wz]` | 3 |
| 关节相对位置 | 12 |
| 关节相对速度 | 12 |
| 上一时刻动作 | 12 |
| 局部高程扫描 | 187 |

部署到真实 Go2 时必须复现相同的观测顺序、尺度、坐标系、裁剪范围、扫描网格、控制频率和动作映射。仿真中的 RayCaster 不能直接用于真机，需要用深度相机、激光雷达或其他感知模块生成等价的局部高程图。


