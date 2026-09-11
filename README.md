# MglfLab

> 项目名称：**MglfLab**  
> Python 包名：`mglf_lab`  
> pip 安装包名：`mglf-lab`

基于 Isaaclab_Parkour 当中 teacher 阶段（阶段一）思路的 Unitree Go2、Go2W 与自定义四足机器人粗糙地形速度控制项目。策略不依赖 waypoint 或目标点，直接接收遥控速度指令，并利用局部高程扫描适应地形。

## 1. 功能与设计

- 控制指令为 `[vx, vy, wz]`：前后速度、横向速度和偏航角速度。
- 不使用 waypoint、目标位置或导航规划器。
- 高程扫描范围为 `1.6 m × 1.0 m`，分辨率为 `0.1 m`，共 187 个采样点，并随机器人 yaw 旋转。
- 使用 Isaac Lab 的 `ROUGH_TERRAINS_CFG`：上下楼梯、斜坡、离散障碍、箱体和随机粗糙面。
- Go2 与 UIKA 上下楼梯高度为 `0.05～0.23 m`；Go2W 楼梯专项课程为 `0.08～0.28 m`。
- 使用温和的髋关节姿态惩罚抑制内八，同时保留横移和复杂落脚能力。
- Go2 和 UIKA 使用 12 个腿部位置动作；Go2W 使用 12 个腿部位置动作和 4 个轮子速度动作。
- Go2 从项目内的 URDF + mesh 导入，物理参数已对照原 Isaac Lab Go2 USD 校验；详见 [模型一致性记录](source/mglf_lab/data/Robots/unitree/go2_description/USD_PARITY.md)。
- 使用 Isaac Lab 原生 RSL-RL 流程，不依赖 `Isaaclab_Parkour` 的旧版自定义 runner。

有限的训练分布无法保证机器人通过任意未知地形。沟壑、窄桥、跳跃或松软地面等能力仍需加入对应的训练地形和奖励。

## 2. 目录结构

```text
MglfLab/
├── source/mglf_lab/
│   ├── assets/go2w.py                      # Go2W 执行器与初始状态
│   ├── assets/go2.py                       # Go2 URDF 加载，保留原 USD 训练参数
│   ├── assets/uika.py                      # UIKA URDF、执行器与初始状态
│   ├── data/Robots/unitree/go2_description/ # Go2 URDF、网格与一致性记录
│   ├── data/Robots/unitree/go2w_description/ # Go2W URDF 与网格
│   ├── data/Robots/uika_description/       # UIKA URDF 与网格
│   └── tasks/
│       ├── go2_rough_env_cfg.py            # Go2 环境
│       ├── go2w_rough_env_cfg.py           # Go2W 轮腿环境
│       ├── uika_rough_env_cfg.py           # UIKA 环境
│       ├── rsl_rl_ppo_cfg.py               # Go2 PPO 参数
│       ├── go2w_rsl_rl_ppo_cfg.py          # Go2W PPO 参数
│       └── __init__.py                     # Gymnasium 任务注册
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

项目为三种机器人注册任务；每种机器人的训练和播放共享同一环境配置：

| 任务 | 用途 |
| --- | --- |
| `Go2-Rough-Teleop-v0` | 训练、随机播放、键盘遥控和可视化 |
| `Go2W-Rough-Teleop-v0` | Go2W 复杂地形训练、随机播放、键盘遥控和可视化 |
| `UIKA-Rough-Teleop-v0` | UIKA 粗糙地形训练、随机播放、键盘遥控和可视化 |

脚本默认使用 Go2。训练或播放 Go2W/UIKA 时必须传入对应的 `--task`。

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

### 5.3 Go2W 冒烟测试与正式训练

Go2、Go2W 和 UIKA 启动时均由 Isaac Lab 将本地 URDF 转换为仿真使用的 USD。转换时的
材质名称和 fixed link 合并警告通常不影响训练。

```bash
python scripts/train.py \
  --task Go2W-Rough-Teleop-v0 \
  --num_envs 32 \
  --max_iterations 2 \
  --seed 1 \
  --headless
```

冒烟测试通过后正式训练：

```bash
python scripts/train.py \
  --task Go2W-Rough-Teleop-v0 \
  --num_envs 1024 \
  --max_iterations 10000 \
  --seed 1 \
  --run_name seed1 \
  --headless
```

Go2W 日志独立保存在 `logs/rsl_rl/go2w_rough_teleop/`，不会覆盖 Go2 日志。

Go2W 按 RobotLab v2.3.2 官方 Go2→Go2W 的差分配置：在 Go2 的 12 个腿部位置动作
之外增加 4 个缩放为 `5 rad/s` 的轮速动作；轮角度不进入观测，轮速进入观测；腿部
和轮子的力矩/加速度惩罚分开计算。Go2W 与 Go2 使用相同 PPO 配置，不为轮式版本
单独改变动作裁剪、网络归一化、探索噪声或学习率。

平地滚动时四个轮子持续接地是正常行为，因此 Go2W 关闭 `feet_air_time` 等纯足式
步态项，改用 RobotLab Go2/Go2W 公共底座中的行为约束：移动/静止分级腿姿态、对角
腿对称、朝上奖励、非轮子部件碰地惩罚、轮子接触力和关节限位。高程图作为 MglfLab
额外的 actor 观测保留，但不直接参与奖励计算。

如果旧 checkpoint 已经学会屈腿贴地爬行，建议使用上述正式训练命令从头训练。虽然
也可以在新奖励下续训，但 PPO 可能长时间停留在原来的局部最优姿态，不能把续训结果
作为新配置是否有效的可靠判断。

RobotLab 官方 actor 删除了高程图；MglfLab 为复杂地形能力保留 `1.6 × 1.0 m`
高程观测，并在进入策略前将 RayCaster 漏检产生的 NaN/Inf 转换为有限值。这里不再
启用实验性的机身高度奖励，避免高程奖励本身改变 RobotLab 的 Go2W 行为目标。

MglfLab 的遥控指令上限为 `1.5`，高于 RobotLab 常用的 `1.0`，因此 Go2W 的线速度
和偏航速度指数奖励 `std` 使用 `0.75`（而非 `0.5`），避免高速样本过早进入近零奖励
区。Go2W 的熵系数为 `0.005`，用于减缓 16 维动作噪声在策略成形后继续上涨；其余
PPO 参数仍与 Go2 相同。

Go2W 使用楼梯专项课程：上下楼梯各占 30%，台阶高度训练范围为 `0.08～0.28 m`，其余
40% 保留箱体、随机粗糙面和双向坡面。移动时腿姿态惩罚权重为 `-0.5`，静止倍率为
`10`，在不削弱零指令站姿的前提下允许跨台阶时更大幅度地抬腿和伸腿。Go2 的楼梯
分布和奖励不受此设置影响。

### 5.4 中断训练后继续训练

恢复权重、训练轮数和 Adam 优化器，并自动选择该 run 的最新 checkpoint：

```bash
python scripts/train.py \
  --task Go2-Rough-Teleop-v0 \
  --num_envs 1024 \
  --resume \
  --load_run 2026-08-28_10-00-00_seed1 \
  --checkpoint model_5000.pt \
  --max_iterations 1000 \
  --headless
```

训练脚本中的 `--checkpoint` 是 `--load_run` 目录内的文件名或匹配表达式。

Go2W/UIKA 续训时把任务改为对应 task，程序会自动使用各自独立实验目录。

### 5.5 数值发散后恢复

如果出现 `inf`、`nan`、`value_function loss: inf` 或 `normal expects all elements of std >= 0.0`，应回退到最后一个健康 checkpoint，并丢弃旧优化器状态：

```bash
python scripts/train.py \
  --task Go2-Rough-Teleop-v0 \
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

`play.py` 可同时观察多只 Go2、Go2W 或 UIKA，每个环境独立采样 `[vx, vy, wz]`，每 5～8 秒重新采样。
播放仍使用唯一的训练环境配置，但默认把生成的地形网格从训练时的
`10 × 20` 缩小为 `6 × 6`。6 列分别对应 6 类地形，行方向按难度从低到高排列。可通过 `--terrain_rows` 和 `--terrain_cols` 调整。

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

Go2W 多环境播放：

```bash
python scripts/play.py \
  --task Go2W-Rough-Teleop-v0 \
  --num_envs 20 \
  --checkpoint /home/mglf/rc/MglfLab/logs/rsl_rl/go2w_rough_teleop/RUN目录/model_XXXX.pt
```

UIKA 多环境播放：

```bash
python scripts/play.py \
  --task UIKA-Rough-Teleop-v0 \
  --num_envs 20 \
  --checkpoint /home/mglf/rc/MglfLab/logs/rsl_rl/uika_rough_teleop/RUN目录/model_XXXX.pt
```

## 8. 键盘遥控播放

`play_keyboard.py` 默认创建一只 Go2：

```bash
python scripts/play_keyboard.py \
  --checkpoint /home/mglf/rc/MglfLab/logs/rsl_rl/go2_rough_teleop/RUN目录/model_5000.pt
```

Go2W 键盘遥控：

```bash
python scripts/play_keyboard.py \
  --task Go2W-Rough-Teleop-v0 \
  --checkpoint /home/mglf/rc/MglfLab/logs/rsl_rl/go2w_rough_teleop/RUN目录/model_XXXX.pt \
  --visualize_height_scan
```

UIKA 键盘遥控：

```bash
python scripts/play_keyboard.py \
  --task UIKA-Rough-Teleop-v0 \
  --checkpoint /home/mglf/rc/MglfLab/logs/rsl_rl/uika_rough_teleop/RUN目录/model_XXXX.pt \
  --visualize_height_scan
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
| `--task` | `Go2-Rough-Teleop-v0` | 机器人任务；Go2W/UIKA 必须指定对应 task |
| `--num_envs` | 环境配置值 | 并行环境数量 |
| `--max_iterations` | Go2/UIKA 为 `17000`，Go2W 为 `10000` | 新训练轮数或本次额外续训轮数 |
| `--seed` | `1` | 随机种子 |
| `--experiment_name` | 随任务选择 | Go2 为 `go2_rough_teleop`，Go2W 为 `go2w_rough_teleop`，UIKA 为 `uika_rough_teleop` |
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
| `--task` | `Go2-Rough-Teleop-v0` | 必须与 checkpoint 对应；Go2W/UIKA 使用对应 task |
| `--num_envs` | `50` | 同时创建的环境数量 |
| `--seed` | `1` | 地形和随机指令种子 |
| `--terrain_rows` | `6` | 播放场地的地形行数 |
| `--terrain_cols` | `6` | 播放场地的地形列数 |
| `--load_run` | 自动匹配 | 从实验目录选择 run |
| `--checkpoint` | 自动选择最新 | 可直接传 checkpoint 绝对路径 |
| `--real_time` | 默认开启 | 按真实时间限速 |
| `--no_real_time` | 关闭 | 取消真实时间限速 |
| `--device` | `cuda:0` | 仿真设备 |
| `--headless` | 关闭 | 无窗口播放，适合批量评估 |

### 9.3 `play_keyboard.py`

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--task` | `Go2-Rough-Teleop-v0` | 必须与 checkpoint 对应；Go2W/UIKA 使用对应 task |
| `--num_envs` | `1` | 环境数量，建议保持 1 |
| `--seed` | `1` | 地形随机种子 |
| `--terrain_rows` | `6` | 播放场地的地形行数 |
| `--terrain_cols` | `6` | 播放场地的地形列数 |
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

Go2 策略观测为 235 维：

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

UIKA 策略与 Go2 同为 235 维观测、12 维动作，但关节顺序、默认姿态和执行器参数不同，checkpoint 不能和 Go2 混用。Go2W 策略观测为 243 维：腿部相对位置仍为 12 维，关节速度和上一动作分别增加为 16 维，高程扫描仍为 187 维。连续旋转的 4 个轮子角度不会进入策略，防止角度无限累积破坏观测分布。Go2、Go2W 与 UIKA checkpoint 不能混用。

部署到真实 Go2、Go2W 或 UIKA 时必须复现相同的观测顺序、尺度、坐标系、裁剪范围、扫描网格、控制频率和动作映射。仿真中的 RayCaster 不能直接用于真机，需要用深度相机、激光雷达或其他感知模块生成等价的局部高程图。

## 11. 仿照 Go2 接入自己的四足机器人：UIKA

UIKA 的接入方式是“复用 Go2 rough 训练策略，替换机器人模型和关节配置”。当前已经注册为 `UIKA-Rough-Teleop-v0`。

关键文件：

| 文件 | 作用 |
| --- | --- |
| `source/mglf_lab/data/Robots/uika_description/urdf/uika_simple_collision.urdf` | UIKA 训练用 URDF |
| `source/mglf_lab/data/Robots/uika_description/meshes/` | UIKA 网格文件 |
| `source/mglf_lab/assets/uika.py` | URDF 路径、初始关节角、执行器力矩/速度/PD 参数 |
| `source/mglf_lab/tasks/uika_rough_env_cfg.py` | 继承 Go2 rough 环境，替换 UIKA 机器人、动作关节顺序和脚/机身正则 |
| `source/mglf_lab/tasks/rsl_rl_ppo_cfg.py` | UIKA PPO runner，实验名为 `uika_rough_teleop` |
| `source/mglf_lab/tasks/__init__.py` | Gymnasium task 注册 |
| `setup.py` | 打包时包含 UIKA URDF 与 mesh |

接入新四足时，最少需要按这个顺序改：

1. 把新机器人的 URDF 和 meshes 放进 `source/mglf_lab/data/Robots/你的机器人名_description/`。
2. 如果 URDF 里有 `package://.../meshes/`，改成相对路径，例如 `../meshes/xxx.STL`。
3. 复制 `source/mglf_lab/assets/uika.py`，改 URDF 路径、默认站姿、关节名正则、力矩/速度限制和 PD 参数。
4. 复制 `source/mglf_lab/tasks/uika_rough_env_cfg.py`，改 task 类名、asset import、12 个动作关节顺序、脚部 body 正则和 base body 名。
5. 在 `source/mglf_lab/tasks/rsl_rl_ppo_cfg.py` 里新增 runner 类，只改 `experiment_name` 即可先跑通。
6. 在 `source/mglf_lab/tasks/__init__.py` 里注册新的 task id。
7. 在 `setup.py` 的 `package_data` 中加入新机器人的 `urdf/*` 和 `meshes/*`。

UIKA 冒烟测试：

```bash
python scripts/train.py \
  --task UIKA-Rough-Teleop-v0 \
  --num_envs 32 \
  --max_iterations 2 \
  --seed 1 \
  --headless
```

UIKA 正式训练：

```bash
python scripts/train.py \
  --task UIKA-Rough-Teleop-v0 \
  --num_envs 1024 \
  --max_iterations 10000 \
  --seed 1 \
  --run_name seed1 \
  --headless
```

UIKA 日志独立保存在 `logs/rsl_rl/uika_rough_teleop/`。播放时必须指定同一个 task：

```bash
python scripts/play_keyboard.py \
  --task UIKA-Rough-Teleop-v0 \
  --checkpoint /home/mglf/rc/MglfLab/logs/rsl_rl/uika_rough_teleop/RUN目录/model_XXXX.pt \
  --visualize_height_scan
```
