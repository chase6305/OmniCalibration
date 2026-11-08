# 使用指南

[返回首页](../README.md)。以下命令均从仓库根目录运行，结果写入 `outputs/`。
安装见首页；Marvin 资产单独提供，路径不同时给演示、标定和可视化命令添加 `--asset-root /path/to/assets`。
所有示例均为合成验证，不能解释为实机标定精度。

## 选择机器人配置

| 配置 | 待估计关节零偏 | 数据根坐标 | 相机条件 |
|---|---:|---|---|
| `configs/marvin_left.json` | 左臂 7 个 | `base_link` | 联合估计相机，默认有机身参考 |
| `configs/marvin_right.json` | 右臂 7 个 | `base_link` | 联合估计相机，默认有机身参考 |
| `configs/marvin_dual.json` | 双臂 14 个 | `base_link` | 联合估计相机，默认有机身参考 |
| `configs/marvin_standalone_left.json` | 左臂 7 个 | `left_arm_base` | 使用 `--fixed-camera` 提供已知外参 |
| `configs/marvin_standalone_right.json` | 右臂 7 个 | `right_arm_base` | 使用 `--fixed-camera` 提供已知外参 |
| `configs/marvin_left_full_scene.json` | 左臂 7 个 | `base_link` | 可匹配右臂表面，须提供可信右臂状态 |

相机自由估计时增加 6 个参数；可选深度尺度再增加 1 个。更换根坐标时必须同步转换外参，不能只替换配置文件。
下文依次介绍 [深度标定](#单臂双臂深度标定)、[独立单臂](#独立单臂与规范自由度)、
[观测回放](#数据回放与真实数据接口)、[腕部阶段](#冻结机械臂修正后标定腕部相机)、
[深度尺度](#深度尺度与可视化对照) 和 [选帧](#从已有观测中减少拟合帧数)。

## 单臂、双臂深度标定

以下示例估计选定机械臂的零偏和固定 RGB-D 相机的 6 个外参修正。使用真实三角面、视场裁剪、自身遮挡和 0.2 mm 轴向深度噪声。固定基座表面为参考系提供锚定；这里的相机是放置在机器人前方的假设观察相机。

```bash
# 左臂：7 个零偏 + 6 个相机参数，同时导出可回放的观测
.venv/bin/python -m omnicalib asset-demo \
  --config configs/marvin_left.json --matching triangle --render-depth \
  --profile robust --frames 8 --iterations 30 --noise-m 0.0002 \
  --camera-position 1.6 0 1.1 --camera-target 0 0 1.0 \
  --save-dataset outputs/left.npz --output outputs/left.json

# 双臂：14 个零偏 + 6 个相机参数
.venv/bin/python -m omnicalib asset-demo \
  --config configs/marvin_dual.json --matching triangle --render-depth \
  --profile robust --frames 8 --iterations 30 --noise-m 0.0002 \
  --camera-position 1.6 0 1.1 --camera-target 0 0 1.0 \
  --output outputs/dual.json

# 右臂使用 configs/marvin_right.json
```

`--profile robust` 使用延后裁剪、0.5 倍局部更新和基于局部不确定度的停止准则。相机初值、邻近初值重试、残差一致性检查均记录在 JSON 中，未使用注入真值挑选解。

## 更接近机载视角的伸手动作

逆运动学生成双手伸到机身前方的姿态，近机身假设相机观察机器人自身：

```bash
.venv/bin/python -m omnicalib asset-demo \
  --config configs/marvin_dual.json --matching triangle --render-depth \
  --posture-mode reaching --profile robust --frames 8 --iterations 30 \
  --noise-m 0.0002 --camera-position 0.18 0 1.23 --camera-target 0.45 0 0.85 \
  --output outputs/onboard_reaching.json
```

IK 只验证关节限位和末端目标，不验证碰撞、速度、力矩或硬件执行可行性。渲染只包含配置中列出的观测网格，不含环境、光学畸变、多路径等完整传感器效应。

## 独立单臂与规范自由度

仅观察单臂运动表面，同时估计相机位姿时，第一关节零偏会与相机位姿耦合。增加姿态数量不能自动消除此自由度。可观察固定几何参考，或提供已知相机外参：

```bash
# 独立 left_arm.urdf；相机已知，仅估计 7 个零偏
.venv/bin/python -m omnicalib asset-demo \
  --config configs/marvin_standalone_left.json --fixed-camera \
  --matching triangle --independent-sampling \
  --output outputs/standalone.json

# 退化对照：完整模型中仅观测左臂，去掉固定基座参考
.venv/bin/python -m omnicalib asset-demo \
  --config configs/marvin_left.json --matching triangle --render-depth \
  --no-anchor --output outputs/unanchored.json
```

退化对照应给出 `rank=12/13` 并保留名义修正。`success` 表示内部求解器成功；是否接受整组修正还取决于 `icp_converged`、`quality_passed`、参数界限、秩、条件数和用户指定的 sigma 阈值。最终决定见 `proposal.block_supported` 和 `proposal.block_reasons`。

## 数据回放与真实数据接口

```bash
.venv/bin/python -m omnicalib calibrate \
  --config configs/marvin_left.json --input outputs/left.npz \
  --matching triangle --profile robust --iterations 30 \
  --output outputs/left_replayed.json
```

NPZ 格式包含关节名称、同步读数、已分割的相机坐标点云和名义外参，支持每帧不同数量的点及未标定关节的运动。可选合成真值单独保存；标定入口不会读取它。真实数据转换示例见 [数据格式](DATA_FORMAT.md)。

编码器噪声会同时影响同一帧的许多点。已知其标准差时，可使用共享误差模型，例如：

```bash
.venv/bin/python -m omnicalib calibrate \
  --config configs/marvin_dual.json --input your_dataset.npz \
  --profile robust --loss linear --residual-sigma-m 0.0002 \
  --assumed-encoder-sigma-deg 0.01 --max-joint-sigma-deg 0.1 \
  --iterations 30 --output outputs/calibration.json
```

这些数值是示例噪声假设，不是 Marvin 硬件规格。`--encoder-noise-deg` 用于合成注入；`--assumed-encoder-sigma-deg` 才是估计器使用的噪声假设，两者独立。

## 冻结机械臂修正后标定腕部相机

```bash
.venv/bin/python -m omnicalib staged-demo \
  --config configs/marvin_dual.json --moving-root \
  --output outputs/staged_dual.json
```

先用深度估计机械臂零偏，再冻结该结果，通过同步的**米制相机位姿**和机器人 FK 估计腕部外参；已知根部运动会先投影到腕部挂载。这里的腕部位姿是合成测量，尚未实现真实互视、RGB-D VO 或图像前端。

## 统计验证和图像

```bash
# 完整多种子评估；逐案例保存检查点、代码快照、输入哈希，可续跑；基准脚本的资产路径由对应配置指定
.venv/bin/python scripts/benchmark.py \
  --plan configs/benchmark_final.json --output outputs/heldout --resume

# 导出可分享的 PNG 和 SVG 图
.venv/bin/python scripts/plot_benchmark.py outputs/heldout
.venv/bin/python scripts/preview_depth.py --random-posture
```

报告输出到 `outputs/`。冻结实验与历史结果见 [实验索引](EXPERIMENTS.md)，方法假设见 [复现边界](REPRODUCTION.md)。

## 深度尺度与可视化对照

可选地联合估计深度乘数，适用于具有可信米制几何参考的观测。下面注入 0.5% 深度尺度偏差；优化器不读取该真值，所用噪声假设需要单独声明。

```bash
.venv/bin/python -m omnicalib asset-demo \
  --config configs/marvin_dual.json --matching triangle --render-depth \
  --frames 8 --iterations 35 --noise-m 0.0002 --residual-sigma-m 0.0002 \
  --depth-scale-bias 0.005 --estimate-depth-scale --step-relaxation 0.5 \
  --profile robust --camera-position 1.6 0 1.1 --camera-target 0 0 1.0 \
  --save-dataset outputs/scale.npz --output outputs/scale.json

.venv/bin/python -m omnicalib calibrate \
  --config configs/marvin_dual.json --input outputs/scale.npz \
  --estimate-depth-scale --residual-sigma-m 0.0002 --iterations 35 \
  --profile robust --step-relaxation 0.5 --output outputs/scale_replayed.json

.venv/bin/python scripts/preview_calibration.py \
  --config configs/marvin_dual.json --input outputs/scale.npz \
  --report outputs/scale_replayed.json --animate --output outputs/scale_preview.png
```

返回 `depth_scale` 满足 `p_corrected = depth_scale * p_raw`。全局 CAD 尺寸与深度尺度可能不可区分；此选项不代替独立米制参考。尺度参数也参与整组可观测性和不确定度判断。光轴噪声模型、参数协方差和每帧/连杆残差见 [数据格式](DATA_FORMAT.md)。

![尺度案例：名义与候选几何的深度对照](figures/scale_calibration.png)

[八帧动画](figures/scale_calibration.gif)；图中轴向误差包含轮廓和遮挡变化，不是 ICP 点面残差或实机精度。第二轮实验与失败记录见 [持续优化记录](ROUND2_LOG.md)。

## 从已有观测中减少拟合帧数

```bash
.venv/bin/python scripts/select_frames.py \
  --config configs/marvin_dual.json --input your_dataset.npz \
  --count 6 --strategy information --output outputs/selected.npz

.venv/bin/python -m omnicalib calibrate \
  --config configs/marvin_dual.json --input outputs/selected.npz \
  --profile robust --iterations 35 --output outputs/selected_calibration.json
```

选择器使用名义几何雅可比，按增量 log-det 排序，仅读取同步关节和表面点；导出的子集不复制合成真值。排序正则项不参与最终标定和秩诊断。该启发式使用各向同性独立点噪声评分，不能保证最小真实误差，也不生成可直接执行的机器人动作。已知相机或同时估计尺度时，选择与标定两步应使用一致的 `--fixed-camera`、`--estimate-depth-scale` 参数。

独立右臂使用 `configs/marvin_standalone_right.json --fixed-camera`。左右独立模块分别以 `left_arm_base`、`right_arm_base` 为根；整机配置以 `base_link` 为根。若只标定左臂但输入包含整机点云，可使用 `configs/marvin_left_full_scene.json` 并逐帧提供已可信的右臂状态；具体参考条件见 [数据接口](DATA_FORMAT.md#单臂标定时的整机点云)。

当前 Marvin 两个肩部第一关节轴线共线：去掉固定机身参考后，**双臂也有一个公共肩部/相机歧义，秩为 19/20**。更多姿态不能消除该自由度。默认三种整机配置均包含固定参考；原理与 30 次退化对照见 [方法说明](REPRODUCTION.md#marvin-双臂也存在公共规范自由度) 和 [补充验证](SUPPLEMENTARY.md)。
