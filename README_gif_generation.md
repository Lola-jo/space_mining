# 太空采矿环境评估和GIF生成指南

这个工具允许你加载和评估多智能体协作太空采矿环境中的训练模型，并生成可视化的GIF动画。

## 基本用法

要评估模型并生成GIF，请运行：

```bash
python enjoy_space_mining.py --exp_path /path/to/experiment
```

这将自动加载实验中的最佳模型，运行5个评估回合，并生成GIF动画。

## 参数说明

以下是可用的命令行参数：

- `--exp_path`：**必需**，实验结果的路径
- `--iteration`：指定要评估的迭代编号
- `--sample`：指定要评估的样本编号
- `--seed`：设置随机种子（默认：42）
- `--episodes`：设置评估的回合数（默认：5）
- `--benchmark`：使用基准模型而不是训练模型
- `--gif_name`：自定义GIF文件名
- `--gif_fps`：设置GIF的帧率（默认：30）
- `--render_width`：设置渲染宽度（默认：640）
- `--render_height`：设置渲染高度（默认：480）
- `--render_mode`：渲染模式，"rgb_array"用于生成GIF，"human"用于实时查看
- `--max_steps`：设置每个回合的最大步数（覆盖配置文件中的值）
- `--skip_frames`：设置在GIF中跳过的帧数，以减小文件大小（默认：2）

## 示例

1. 使用默认参数评估最佳模型：

```bash
python enjoy_space_mining.py --exp_path /home/user/experiments/space_mining_experiment
```

2. 评估特定迭代和样本的模型：

```bash
python enjoy_space_mining.py --exp_path /home/user/experiments/space_mining_experiment --iteration 2 --sample 1
```

3. 生成高质量GIF（更高分辨率，更高帧率）：

```bash
python enjoy_space_mining.py --exp_path /home/user/experiments/space_mining_experiment --render_width 1280 --render_height 720 --gif_fps 60 --skip_frames 1
```

4. 实时查看模型行为而不生成GIF：

```bash
python enjoy_space_mining.py --exp_path /home/user/experiments/space_mining_experiment --render_mode human
```

5. 使用不同的随机种子进行评估：

```bash
python enjoy_space_mining.py --exp_path /home/user/experiments/space_mining_experiment --seed 100
```

## 随机种子的重要性

使用不同的随机种子可以帮助评估模型在不同初始条件下的稳健性。每次评估会使用从指定种子开始的连续值，例如，如果设置`--seed 42 --episodes 5`，将使用种子42, 43, 44, 45, 46进行五个回合的评估。

## GIF文件

生成的GIF文件会保存在两个位置：
1. 模型所在目录下
2. 当前工作目录下

这样可以方便你在不同的地方找到生成的GIF文件。

## 疑难解答

如果遇到问题：

1. 确保已安装所有必要的依赖：
   ```bash
   pip install stable-baselines3 gymnasium numpy imageio
   ```

2. 检查模型文件是否存在。模型文件通常命名为`model.zip`。

3. 如果环境注册失败，可能需要直接导入环境文件。脚本会尝试自动处理这种情况。

4. 内存不足时可以减小渲染分辨率或增加`skip_frames`值。 