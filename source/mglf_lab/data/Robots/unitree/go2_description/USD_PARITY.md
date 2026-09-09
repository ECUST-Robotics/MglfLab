# Go2 URDF 与原 USD 的物理参数一致性

2026-09-09，在本机 Isaac Sim 5.1 / Isaac Lab 环境中验证。

训练与播放使用 `assets/go2.py` 中的 `UNITREE_GO2_URDF_CFG`，资产来源为本目录的
`urdf/go2_description.urdf` 与 `meshes/`。Isaac Lab 的 URDF 导入器仍会生成临时
USD 供 PhysX 使用；训练不再读取原远程 `go2.usd`。

## 参考资产

原训练记录使用的资产：

```text
https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/IsaacLab/Robots/Unitree/Go2/go2.usd
```

其相对引用 `Props/instanceable_meshes.usd` 也参与了碰撞形状核对。

SHA-256：

```text
go2.usd
ba171c972b987d8c8fb7157ccad2ba9c0c1fed105755d2d8af46bef96cc11c6d
Props/instanceable_meshes.usd
2902646d0f4c13c9ecae3ac9046e32c7d18902eef9679de9829f642a15c93fb5
```

## 对源 URDF 的调整

源文件与 DAE 网格来自仓库内 RobotLab，许可见项目 `THIRD_PARTY_LICENSES/`。

- 删除 12 个无可视/碰撞几何的 `*_rotor` 连杆及其固定关节。直接导入原 URDF
  会把额外的转子质量并入 base、hip、thigh，总计增加 1.068 kg；原 USD 没有这部分质量。
- 保留脚部和头部固定关节的 `dont_collapse="true"`，保留原来的 19 个刚体。
- 网格路径改为相对 URDF 的 `../meshes/*.dae`，不依赖 ROS package 路径。
- 原有关节限位、其他惯性参数和全部碰撞几何保持源 URDF 数值。

加载配置复制 `UNITREE_GO2_CFG`，仅替换 spawn 为 `UrdfFileCfg`，保留 DC 电机、
初始姿态、软限位系数 0.9、刚体参数、接触传感器与求解器覆盖参数。
导入器的隐式驱动增益为零，由原 DC 电机模型计算力矩。

## 验证结果

- 转换资产：19 个刚体、18 个关节（12 个转动、6 个固定）、27 个碰撞体。
- 比较质量、质心、完整惯性矩阵、刚体变换、关节连接/锚点/轴、角度限位、
  effort/velocity 限制，以及碰撞体类型、尺寸与世界变换，全部通过。
- PhysX CPU 初始化后：刚体和关节顺序一致；质量、惯量、质心、硬/软角度限位、
  初始关节状态、力矩限制、刚度、阻尼、armature 与摩擦参数一致。
- hip/thigh 的速度上限因角度单位转换存在约 `1.90735e-6 rad/s` 的浮点舍入差异；
  并非物理设置改变。校验使用数值容差，不声称 USD 文件或浮点值逐位相同。
- 完成 4 个环境、1 轮 PPO 的 CPU 训练冒烟测试，运行目录为
  `logs/rsl_rl/go2_rough_teleop/2026-09-09_16-31-52_urdf_parity_smoke/`。

此次验证针对物理参数和训练入口；不代表新旧随机训练轨迹逐步相同，也未验证 GPU
长期训练收敛或 DAE 与原 USD 的渲染材质逐像素一致。

## 重复校验

在 Isaac Lab Python 环境中，从 MglfLab 目录运行：

```bash
python scripts/check_go2_urdf.py --headless --reference-usd /路径/Go2/go2.usd
```

本地参考 USD 旁边必须有 `Props/instanceable_meshes.usd`。脚本重新转换当前 URDF，
比较组合后的 USD 物理属性，再创建两个 PhysX articulation 比较实际初始化参数。
任一差异超过容差会报错。修改 URDF 或升级 Isaac Lab / Isaac Sim 后应重新运行。
