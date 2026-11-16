# BPP 的 TIG 仿真适配

本分支提供 BPP video-only 策略、IG-10K 数据适配、可恢复的多卡训练器和闭环评测入口。它保留 BPP 原生 tokenizer、pooling、交互注意力、UNet 和扩散目标，采用 BPP 已有的双编码器选项。

当前发布配方为 `bpp-tig-split-clip-frozen-prompt-v2`：**冻结 prompt 侧 CLIP，训练机器人直接观测侧 CLIP**。这是以最新官方 ACT/DP 默认模块分工作为参照的 BPP 适配选择，并非 TIG 对所有方法的统一限制。没有加载 LIBERO 或其他机器人策略 checkpoint。

## 模型和数据契约

| 项目 | 设置 |
|---|---|
| 视觉初始化 | 两个 CLIP ViT-B/16 从相同通用预训练权重开始，随后独立 |
| prompt 分支 | RGB 骨干冻结、固定 eval 和 CenterCrop；状态投影与 pooling 可训练 |
| 机器人直接观测分支 | CLIP 可训练，随机图像增强保留 |
| 交互与动作策略 | pooling、cross-attention、状态投影、UNet 可训练 |
| 人类输入 | 同任务人类示范，10帧 RGB224；没有人类动作、关节或特权任务状态 |
| 机器人输入 | 每个控制步更新的2帧RGB和qpos18 |
| 动作 | 预测16步joint16绝对命令，执行前4步后重新查询 |
| 扩散 | 50训练时间步、16步DDIM推理、epsilon目标、EMA |
| 归一化 | TIG ACT/DP 的逐task-level q01/q99；BPP内部为identity，仅变换一次 |
| C45 | 180个机器人task-level仓库、9,000条机器人轨迹、45个人类任务、2,250条示范、3,039,478个机器人帧锚点 |

prompt 编码器也为交互注意力编码当前机器人图像；这一条 query 分支使用冻结 CLIP，机器人直接观测流使用可训练 CLIP。不能把整个 prompt 模块冻结。

正式 ZS 沿用所选 ACT/DP processor 的目标仓库统计校准：**目标任务无参数更新，但使用目标机器人轨迹产生的元数据统计**。这不是完全 source-only 的统计协议。若更换归一化，应从训练到推理统一设计，不能只在推理时替换尺度。

## 版本与安装

- BPP 上游基线：`58125e682acceefef3d60c071212dc483a4db1c9`。
- TIG 公开代码参考：`3193957e3b0187d086e44d79e0e4af9845fbd4fd`。
- CPU 验证环境使用 Python 3.12、PyTorch 2.7.1+cu128、TorchVision 0.22.1+cu128。该环境还访问了既有TIG环境中的共享依赖；`requirements-tig.txt` 记录适配相关包版本，不是整个仿真系统的完整环境锁文件，也没有宣称在空机器上重建环境已验证。新服务器应独立安装依赖，不能直接复制含有本机`.pth`引用的虚拟环境。

先按 TIG 官方说明安装其环境、资产和 FFmpeg 共享库。然后在同一隔离 Python 环境中安装本仓库适配依赖；根据机器驱动选择兼容的 PyTorch 构建。下面是已有 CUDA12.8 环境时的示例：

```bash
git clone --recurse-submodules --branch tig-integration https://github.com/Cat-blizzard/BPP.git
cd BPP
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-tig.txt
python -m pip install -e . --no-deps
mkdir -p external
git clone https://github.com/imitator-game/The-Imitator-Game.git external/The-Imitator-Game
git -C external/The-Imitator-Game checkout 3193957e3b0187d086e44d79e0e4af9845fbd4fd
export TIG_SOURCE_ROOT="$(realpath external/The-Imitator-Game)"
python -m pip install -e "$TIG_SOURCE_ROOT" --no-deps
```

这里的`--no-deps`仅注册TIG源码，前提是已经按TIG说明准备所需运行依赖。固定版本TIG的`pyproject.toml`同时覆盖其他模型，指定`timm==0.9.10`、`diffusers>=0.36.0`，与本适配已验证的`timm==1.0.20`、`diffusers==0.35.1`不同。不要在装好本适配后再次无约束执行TIG的全依赖安装并覆盖这些版本。其他TIG方法应使用各自环境；该说明不代表所有模型能共享同一组依赖。

TIG 资产、IG 数据和通用 CLIP 权重均须另行准备；本仓库不包含它们。CLIP 权重可预先缓存，CPU 自测使用随机初始化且不下载权重。

从本仓库根目录运行命令，并显式指定三个绝对目录：

```bash
export TIG_SOURCE_ROOT=/absolute/path/to/The-Imitator-Game
export IG_HUMAN_ROOT=/absolute/path/to/imitator_human_v1
export IG_SIM_ROOT=/absolute/path/to/imitator_sim_v1_zed2i
export PYTHONPATH="$PWD:$TIG_SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
```

不要复制旧工作站的 NCCL 禁用选项到 H100 环境。发布配置没有强制禁用 P2P、NVLink 或 InfiniBand。

## 训练配置与预算

| 配置文件 | 卡数示例 | 每卡microbatch | 全局batch | 更新次数 | 5%预热 |
|---|---:|---:|---:|---:|---:|
| `infrastructure/bpp_tig/configs/c45.yaml` | 4 | 32 | 128 | 237,460 | 11,873 |
| `infrastructure/bpp_tig/configs/c45_h100x8.yaml` | 8 | 32 | 256 | 118,730 | 5,937 |

两个配置都是 **30,394,880次窗口抽样**，接近 `10 × 3,039,478`，尾部按完整更新向上取整。采样继续采用 BPP 的 task-level 均衡、有放回抽样；这不是官方 DP 的逐帧无放回10个epoch，也不表示相同计算量、相同优化轨迹或已经证明收敛。两个batch配置是独立训练配方，不能互相作为严格恢复。

优化器沿用 BPP 的 AdamW：策略LR `5e-5`，可训练视觉骨干LR `5e-6`，betas `(0.95, 0.999)`，策略weight decay `1e-6`、视觉参数按原生分组，梯度裁剪 `0.3`，bf16，5%预热后余弦衰减。warmup采用四舍五入，H100配置的5936.5步取5937。

```bash
# 四卡示例；不会自动选择或占用GPU。
torchrun --standalone --nproc_per_node=4 -m infrastructure.bpp_tig.train \
  --config infrastructure/bpp_tig/configs/c45.yaml --output-dir runs/tig-c45-v2

# 八卡H100配置；需自行选择可用GPU并确认显存/通信。
torchrun --standalone --nproc_per_node=8 -m infrastructure.bpp_tig.train \
  --config infrastructure/bpp_tig/configs/c45_h100x8.yaml --output-dir runs/tig-c45-v2-h100x8
```

正式C45会校验数据规模、配方和预算。小数据或短工程检查必须标注 `--development`；不能把小样本结果描述成完整C45。GPU型号和峰值算力不能确定实际耗时，应先测稳定吞吐。

## CPU 自测

下列命令只运行合成数据检查，不训练策略、不创建仿真环境。策略测试以随机权重完成真实完整尺寸的前向、反向、AdamW更新和缓存推理，不下载CLIP权重。

```bash
export CUDA_VISIBLE_DEVICES=""
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
python -m behavior_prompting.train_network.utils.prompt_util
python -m infrastructure.bpp_tig.configuration
python -m infrastructure.bpp_tig.data
python -m infrastructure.bpp_tig.policy --self-test
python -m infrastructure.bpp_tig.train --self-test
python -m infrastructure.bpp_tig.evaluate --self-test
python -m infrastructure.bpp_tig.preserve_checkpoints --self-test
python -m infrastructure.bpp_tig.source_dev_runner --self-test
```

训练前开启新终端，或明确恢复原本的GPU可见性和联网设置；上面的环境变量专用于CPU自测。

## 暂停与恢复

向运行输出目录写入 `STOP` 文件后，训练器在更新边界保存完整checkpoint并正常退出：

```bash
touch runs/tig-c45-v2/STOP
```

恢复前确认 `status.json` 与 `checkpoints/latest.json` 已指向完整checkpoint；移除 `STOP`，使用相同源码、运行配置、数据身份和卡数调用原训练命令并增加 `--resume CHECKPOINT_PATH`。checkpoint保存模型、EMA、优化器、scheduler、各rank随机状态和采样进度身份。

旧版共享且可训练CLIP实验应从它自己的冻结源码恢复。不能用旧版已经接受IG梯度的CLIP初始化本版，再声称“从第一步冻结通用骨干”；模型键名相同也不构成配方一致。

## 评测与当前限制

评测入口保留 TIG 原生环境和动作控制。正式Seen/ZS每个checkpoint为40个task-level组、每组10回合，共400回合。固定参考视频、帧索引、场景和策略随机种子并保存每回合身份；正式unseen不参与训练时长或checkpoint选择。

先为新运行生成源任务开发协议；四任务×四等级×三seed共48回合，使用训练源任务，不能称为额外held-out测试。生成器实际检查所选TIG提交和配置文件哈希，并从训练配置读取checkpoint候选：

```bash
export RUN_DIR="$PWD/runs/tig-c45-v2"
python -m infrastructure.bpp_tig.evaluate \
  --freeze-source-dev-template infrastructure/bpp_tig/configs/source_dev_c45_v2.template.json \
  --training-config infrastructure/bpp_tig/configs/c45.yaml \
  --output "$RUN_DIR/source-dev-protocol.json"
```

H100运行需同时改为其训练配置及独立输出目录。不要复用另一配置生成的协议。固定预算最终EMA结果与开发集选择结果分开报告。

新训练首次产生完整checkpoint后，可生成绑定该运行真实配置、源码、resume签名的保留配置。此命令不会凭空填写旧运行哈希：

```bash
python -m infrastructure.bpp_tig.preserve_checkpoints \
  --initialize-run "$RUN_DIR" --archive-dir "$RUN_DIR/source-dev/checkpoints" \
  --protocol "$RUN_DIR/source-dev-protocol.json" --write-config "$RUN_DIR/source-dev/retention.json"
```

随后分别运行保留器与 `source_dev_runner`，按各自 `--help` 指定新运行目录和空闲评测GPU；本仓库不自动注册后台服务或分配GPU。直接评测某个开发checkpoint也可使用 `evaluate --source-dev-protocol PROTOCOL --checkpoint CHECKPOINT --output OUTPUT --video`。

正式测试命令示例：

```bash
python -m infrastructure.bpp_tig.evaluate \
  --checkpoint "$RUN_DIR/checkpoints/update-00237460.pt" \
  --data-config infrastructure/bpp_tig/configs/c45.yaml \
  --output "$RUN_DIR/formal" --video
```

八卡配置最终checkpoint为 `update-00118730.pt`，应传入对应的 `c45_h100x8.yaml`。正式评测命令会运行仿真；本次发布没有执行它。

环境跟随固定官方ACT/DP入口的实际等级开关行为，不单凭相同Gym ID认定场景相同，也不私自将通用camera helper的另一条路径覆盖为标准修复。必须记录所用版本与实际等级配置。

SR采用原生 `success_at_end`。代码口径SubSR采用末步 `last_reward`，对应官方仓库中一份Zero-WAM C15、400回合参考记录的核对映射；保留 `success_once`、reward、return等原始字段。这不等于论文阶段阈值Sub-SR，也不能推广为所有历史提交的统一定义。

本版实现了原生推理阶段的prompt缓存；**尚未实现离线训练raw CLIP特征缓存**。不能用“缓存后预计更快”作为已经测得的吞吐或时长。新配方尚未完成GPU/DDP长训练、H100吞吐测试或正式400回合评测；CPU验证结果不替代这些结果。

2026-11-17已执行CPU验证：完整尺寸策略前向/反向和真实优化器更新、冻结骨干不变、推理缓存/reset、配置与预算、精确恢复、非整千checkpoint保存、来源快照、评测终点及保留队列自测均通过。读取了完整C45元数据核对上述规模，并实际解码三个跨episode边界的机器人帧锚点及人类10帧示范。另对一个Seen和一个Unseen参考视频验证50条参考池、官方uniform_jitter帧索引及重复读取像素一致性；没有运行机器人仿真或用这些样本更新模型。此检查不等于全视频内容审计或正式成功率结果。

旧版开发结果属于另一份源码、配方和运行身份，不能作为本版成功率。新的开发协议应在新运行身份下重新固定；checkpoint保留与选择不能复用旧运行的签名和哈希。

## 来源

- [TIG官方ACT/DP训练与评测代码](https://github.com/imitator-game/The-Imitator-Game/tree/3193957e3b0187d086e44d79e0e4af9845fbd4fd/examples/baselines)
- [BPP原生优化配置](https://github.com/real-stanford/behavior_prompting/blob/58125e682acceefef3d60c071212dc483a4db1c9/behavior_prompting/train_network/config/libero_policy_dunetp.yaml)
- [BPP原生编码器分离实现](https://github.com/real-stanford/behavior_prompting/blob/58125e682acceefef3d60c071212dc483a4db1c9/behavior_prompting/train_network/model/prompt/prompt_obs_encoder.py)
- [TIG参考指标核对](https://github.com/imitator-game/The-Imitator-Game/blob/3193957e3b0187d086e44d79e0e4af9845fbd4fd/examples/baselines/astra/docs/METRICS.md)

上游BPP许可与作者归属保留在原README及LICENSE中。适配代码通过导入使用TIG，不打包数据、权重或整个TIG代码库。
