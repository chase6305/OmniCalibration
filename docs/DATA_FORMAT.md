# 数据格式和真实观测接口

标定入口接受已同步、已分割的三维观测，不直接读取 ROS bag 或原始 RGB-D 图像。`omnicalib.dataset` 提供版本化 NPZ 格式，使用 `allow_pickle=False` 读取，并采用原子替换写入，避免中断时损坏已有数据或结果。

## 坐标和单位

- `T_AB` 将 B 中的坐标变换到 A，列向量左乘。
- `camera_nominal` 为 `T_BC`，B 必须是配置使用的 URDF 根坐标系；C 为点云所在相机坐标系。
- 点和位移单位为米，角度为弧度。针孔渲染采用 OpenCV 光学坐标：x 向右、y 向下、z 向前。
- 关节顺序按名称匹配；`calibrate` 能重排同一关节集合，不接受集合不一致的数据。
- 估计符号为 `q_corrected = q_measured + delta_q`。
- 固定相机的外参在每帧相对 B 不变。移动腕部相机应使用腕部轨迹接口，不应直接当作固定相机输入。

## 创建数据文件

```python
import numpy as np
from omnicalib.assets import load_asset_config
from omnicalib.dataset import CalibrationDataset, save_dataset
from omnicalib.upper import DepthFrame

config, urdf, robot = load_asset_config("configs/marvin_left.json")

# 以下三个变量来自实际采集与同步适配器：
# q_measured: (F, 7)，按 config["joint_names"] 排列，单位 rad。
# clouds_camera: F 个 (N_i, 3) 数组，机器人表面点，单位 m。
# T_BC_nominal: (4, 4)，光学相机坐标到 robot.root 的名义外参。
frames = [DepthFrame(q, cloud)
          for q, cloud in zip(q_measured, clouds_camera, strict=True)]

data = CalibrationDataset(
    joint_names=config["joint_names"],
    frames=frames,
    camera_nominal=T_BC_nominal,
    root_frame=robot.root,
    camera_frame="depth_camera_optical",
    metadata={"synthetic": False, "source": "your_capture"},
)
save_dataset("observations.npz", data)
```

点数 `N_i` 可以逐帧变化。无效深度、背景、非机器人表面和明显异常点应在输入前处理。不要把像素坐标、毫米深度值或未消除畸变的射线直接当作米制点云。

采集数据没有合成真值时，不填写 `truth`。合成文件中的真值也与观测分开保存；`calibrate` 调用 `load_dataset(include_truth=False)`，不会读取它。

## 另一只机械臂或夹爪同时运动

若只标定左臂，右臂、夹爪等非标定关节可以逐帧提供已知读数：

```python
frame = DepthFrame(
    q_left,
    cloud_camera,
    fixed_positions={
        "SHOULDER_PITCH_R_J1": q_right_shoulder,
        "LEFT_HAND_FINGER_1": q_left_finger,
    },
)
```

所有帧应提供相同的非标定关节名称集合。这里的 `fixed_positions` 表示该帧中保持已知、不作为优化参数的状态，并不要求其跨帧不动。未提供的状态使用配置值或零。mimic 子关节由 URDF 自动计算，不应作为独立状态或零偏参数重复输入。

若未标定关节本身存在未知偏置，其误差仍可能进入当前标定结果；该接口不会自动估计未选定的参数。

## NPZ 数组

| 字段 | 形状/类型 | 含义 |
|---|---|---|
| `format_version` | 标量整数，当前 1 | 格式版本 |
| `joint_names` | `(N,)` 字符串 | 待标定独立角关节名称 |
| `q` | `(F, N)` 浮点 | 同步编码器读数 |
| `points_camera` | `(P, 3)` 浮点 | 所有帧点云顺序拼接 |
| `point_offsets` | `(F+1,)` 整数 | 第 i 帧为 `[offsets[i]:offsets[i+1]]` |
| `camera_nominal` | `(4, 4)` 浮点 | `T_BC`，必须属于 SE(3) |
| `root_frame`, `camera_frame` | 标量字符串 | 根坐标和点云坐标名称 |
| `fixed_joint_names` | `(M,)` 字符串 | 不参与估计的独立关节 |
| `fixed_positions` | `(F, M)` 浮点 | 对应逐帧状态 |
| `metadata_json` | 标量字符串 | 来源、假设、可选内参等 JSON 对象 |
| `truth_json` | 标量字符串 | 可选合成评分真值；真实数据为 null |

格式不包含自动同步、深度单位识别、镜头畸变求解或点云分割；可选相对 CAD 的尺度估计见下文。元数据可记录时间戳，但当前标定器不估计时间偏移。

## 噪声与接受条件

`--residual-sigma-m` 是独立点到表面噪声的假定尺度。已知编码器标准差时，使用 `--assumed-encoder-sigma-deg`，低秩协方差会体现同一帧的共享编码器误差。该参数不会从合成真值自动读取。

仅高斯噪声实验可选择 `--loss linear`。Huber、soft-L1 和 Cauchy 均可用；非线性损失下局部 Hessian 是条件近似，不是全部系统误差的置信保证。

JSON 接受条件包含：

1. 最后内层求解成功，ICP 外层连续达到稳定条件；早期预算耗尽记录在 history 与 all_inner_success；
2. 参数未触及界限；
3. 测量残差与声明的噪声模型一致，默认归一化 RMS 不超过 2，最终丢弃比例不超过 --max-rejection-fraction（默认 0.2，含裁剪和距离剔除）；
4. 参数块满秩且条件数在限制内；
5. 若用户指定 sigma 上限，全部估计参数满足对应上限。

整组参数中任一条件不满足，就保留整组名义修正，避免将耦合解的一部分单独写入。`proposal` 始终只是 JSON 建议，不会执行硬件或文件模型更新。

`quality_passed` 无法证明 CAD 完全正确。网格尺寸、连杆安装误差、深度尺度误差或时间不同步可能造成仍然满秩的有偏解；应使用独立留出数据、实际注入或外部参考验证。

## 腕部相机轨迹

`omnicalib.wrist.calibrate_eye_in_hand` 接受：

- 已冻结的机械臂零偏；
- 同步关节帧；
- 相机在一致参考下的**米制** 4×4 位姿序列；
- 挂载 link 和可选名义 `T_LC`；
- 根坐标相对世界的已知位姿序列，根部不动时可省略。

先计算每帧 `T_WL = T_WB @ T_BL(q_corrected)`，再构造相对运动。相机和机器人位姿序列各自的世界坐标可以有不同的固定全局规约，因为相对运动会消去该全局变换。尺度、时间和每个序列内部的帧方向必须正确。

未经尺度恢复的单目 VO 不满足这个完整 SE(3) 腕部接口。当前仓库不包含真实 VO/互视图像前端。

## 可选深度尺度与光轴噪声（第二轮）

`--estimate-depth-scale` 将 `depth_log_scale` 加入估计块，输出 `depth_scale = exp(depth_log_scale)`。使用 `p_corrected = depth_scale * p_raw`，默认关闭。`--depth-scale-log-bound 0.05` 是对数坐标的对称界限；`--max-depth-scale-sigma-pct` 是对数尺度标准差的百分数近似，例如 0.05 表示约 0.05%。该尺度只相对于已信任的米制 CAD；整体几何尺寸误差与传感器尺度可能不可区分，不能自动完成毫米/米识别。

`--point-noise-model axial` 假定 `--residual-sigma-m` 是独立光轴 Z 噪声标准差，以像素射线和模型法向传播到点面残差。`--axial-range-power 2` 对应标准差随距离平方增长，0 表示固定 Z 标准差。`--axial-floor-fraction` 给法向噪声乘数设下限，默认 0.25，以免近切向射线获得无限权重。对角深度权重和共享编码器协方差在每次对应点求解中冻结，下一外迭代重新计算；不包含权重对参数的导数或协方差行列式项。

`diagnostics.covariance` 和 `diagnostics.correlation` 仅在估计块满秩时输出；单位顺序为关节 rad、可选相机 [m,m,m,rad,rad,rad]、可选无量纲对数尺度。秩亏时留空。`residual_profile` 报告保留的每帧/每连杆残差数量、均值、RMS、绝对值 95 分位数，单位 mm；它不是独立样本显著性检验，也不包含被裁剪或未关联的点。

若元数据包含 `intrinsics`（`Pinhole` 的 width/height/fx/fy/cx/cy/near/far），可用 `scripts/preview_calibration.py` 输出名义/候选模型的可见深度对照。脚本也不读取 `truth`。

光轴噪声求解先用两轮各向同性权重建立对应点，再切换到声明的轴向模型；达到最终模型并有连续成功、稳定的内层求解后才能停止。`--profile robust` 的局部步长乘数为 0.5，`--step-relaxation` 可以覆盖；停止条件检查完整拟议步长。早期 SciPy 状态 0 表示耗尽函数评估预算，允许在下一轮重建关联后继续；它不等于最终稳定求解失败。

## 单臂标定时的整机点云

默认 `marvin_left.json` 只拟合左臂及基座参考表面。输入应对应配置列出的 surface/anchor links；另一只手臂或环境并不会自动作为可匹配模型。若整机机器人点云仍含大量未配置表面，可能触发覆盖率或残差拒绝。

如果右臂的运动学和逐帧状态已独立可信，可用 `configs/marvin_left_full_scene.json`：仍只估计左臂七个零偏，但也建模右臂表面，通过每帧 `fixed_positions` 提供右臂读数。右臂此时可为相机提供参考；它存在未知零偏或读数噪声时会影响结果，当前共享编码器协方差只针对选定的待标定关节。需要同时估计右臂零偏时应使用双臂配置，而不是把未知状态当作精确参考。

独立右臂模块使用 `configs/marvin_standalone_right.json`、`right_arm_base` 根坐标，独立左臂使用 `left_arm_base`。它们与完整机器人 `base_link` 坐标不同，数据文件不能只换配置而不转换点云外参的根坐标。

## 距离平方噪声的后续验证配置

在本仓库精确 CAD 的开发对照后，另用 20 个未见种子验证了以下配置（无编码器噪声、每次 8 帧）：

```bash
.venv/bin/python -m omnicalib calibrate \
  --config configs/marvin_dual.json --input your_dataset.npz \
  --point-noise-model axial --axial-range-power 2 --axial-floor-fraction 0.8 \
  --residual-sigma-m 0.0002 --profile robust --iterations 50 \
  --output outputs/quadratic_calibration.json
```

这只是与声明的距离平方 Z 噪声相配的一组实验配置，不是 Marvin 硬件噪声规格。新种子上 20/20 稳定，最大关节误差最坏约 0.0434°，边际 ±1.96σ 覆盖率约 93.6%；仍是条件近似。未更改原 400 次评估中的权重下限或预算，原结果与不足均保留。

共享编码器模型假定不同帧之间的读数误差独立；时间相关漂移、回差或未标定参考臂的不确定度未纳入当前协方差。增加帧数也不能自动消除这类误差。

## 在原候选解上继续求解

当前求解器默认会对符合条件的候选追加一次局部续算；未稳定的报告中可能已经包含该尝试。
`local_continuation=False` 可禁用，`continuation_iterations` 可限制每次续算的外迭代次数。
这两个选项通过 Python API 的 `solver_options` 传入，当前 CLI 没有对应开关。
预算和异常回退的完整含义见 [续算预算](BUDGET_v032.md)。下例用于显式继续一个保留的候选。

若 `icp_converged=false`，但残差和覆盖率正常，可在**相同数据、模型、噪声、裁剪与约束**下，从末次估计继续，而非把未稳定解写回模型。`calibrate_dataset` 的 `solver_options['initial']` 顺序为：选定关节零偏、可选相机六维 Lie 修正、可选对数尺度。相机固定时不追加六个零坐标。

例如一个“尺度＋5% 离群点”合成案例，在原 35 次预算中未停止；从该候选值开始、启用已有裁剪、局部步长改为 0.25 后约 12 秒达到严格停止条件。原案例仍保留为未接受，后续结果单独保存；这不是新的独立验证样本，也不构成任意失败都能通过缩步恢复的保证。

```python
import json
import numpy as np
from omnicalib.assets import calibrate_dataset
from omnicalib.reporting import write_json

previous = json.load(open('outputs/calibration.json'))
initial = list(previous['offsets_rad'])
if previous['estimate_camera']:
    initial += previous['camera_correction']
if previous['estimate_depth_scale']:
    initial += [float(np.log(previous['depth_scale']))]
options = dict(previous['solver_options'])
options.update(initial=initial, step_relaxation=0.25,
               trim_start_iteration=0, axial_start_iteration=0,
               allow_restarts=False)
# 以下 sigma 与 trim 必须是前次实际使用的条件，而非由真值倒推。
result = calibrate_dataset(
    'configs/marvin_dual.json', 'observations.npz',
    estimate_camera=previous['estimate_camera'],
    estimate_depth_scale=previous['estimate_depth_scale'],
    residual_sigma_m=0.0002, trim_fraction=0.1, outer_iterations=60,
    solver_options=options,
)
write_json('outputs/continued.json', result)
```

若前次使用了已知外参、无锚定、sigma 上限等条件，续算也要保持相同条件。低残差不能替代数据来源、模型一致性和独立验证。
