## 第一步：MuJoCo actuator test
```bash
cd rc/Mglf_sar
./scripts/run_mujoco_actuator_test.sh
```
## 第二步：Isaac Sim actuator replay
```bash
python scripts/replay_actuator_isaac.py   --task Go2W-Flat-Handstand-Back-v0   --mujoco_log /home/mglf/rc/Mglf_sar/logs/sim2sim/go2w_actuator_mujoco.csv   --output /home/mglf/rc/MglfLab/logs/sim2sim/go2w_actuator_isaac.csv 
```
## 第三步：比较 q/dq
```bash
MPLCONFIGDIR=/tmp/matplotlib python3 scripts/compare_actuator.py
```
## 第四步：调 Isaac Sim actuator 参数
重点看`/home/mglf/rc/MglfLab/source/mglf_lab/assets/go2w.py`当中的参数
```bash
stiffness
damping
effort_limit_sim
velocity_limit_sim
friction
```
经验规则：
```
Isaac 跟踪太慢/幅值偏小：提高 stiffness
Isaac 超调/振荡：提高 damping 或降低 stiffness
Isaac 速度峰值太小：提高 velocity_limit_sim 或降低 damping
Isaac 速度尖峰太大：提高 damping
Isaac torque 长期顶住：提高 effort_limit_sim 或降低目标幅值
Isaac 比 MuJoCo 更“粘”：降低 friction / damping
Isaac 比 MuJoCo 更“滑”：提高 damping / friction
```
如果 wheel 的速度有问题的话也可以调整
```
/home/mglf/rc/MglfLab/source/mglf_lab/tasks/go2w_handstand_env_cfg.py
```
当中的 scale
## 第五步：MuJoCo contact test
```bash
cd /home/mglf/rc/Mglf_sar
./scripts/run_mujoco_contact_test.sh
```
## 第六步：Isaac Sim contact replay
```bash
python scripts/replay_contact_isaac.py \
  --task Go2W-Flat-Handstand-Back-v0 \
  --mujoco_log /home/mglf/rc/Mglf_sar/logs/sim2sim/go2w_contact_mujoco.csv \
  --output /home/mglf/rc/MglfLab/logs/sim2sim/go2w_contact_isaac.csv 
```
## 第七步：比较 base_quat/contact/joint_vel
```bash
MPLCONFIGDIR=/tmp/matplotlib python3 scripts/compare_contact.py
```
## 第八步：调 Isaac Sim contact 参数
## 第九步：Isaac Sim 重新训练或验证 policy
## 第十步：MuJoCo 部署 policy

