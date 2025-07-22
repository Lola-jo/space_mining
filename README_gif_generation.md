# Guidelines for Space Mining Environmental Assessment and GIF Generation

This tool allows you to load and evaluate training models in multi-agent collaborative space mining environments, and generate visual GIF animations.

##  Basic usage

To evaluate the model and generate a GIF, please run:

```bash
python enjoy_space_mining.py --exp_path /path/to/experiment
```

This will automatically load the best model from the experiment, run 5 evaluation rounds, and generate GIF animations.

## Parameter Description

The following are available command-line parameters:

-` -- exp_cath `: * * Required * *, path of experimental results
-` -- iteration `: Specify the iteration number to be evaluated
--- sample: Specify the sample number to be evaluated
--- seed: Set a random seed (default: 42)
--- episodes: Set the number of rounds for evaluation (default: 5)
--- benchmark: Use benchmark models instead of training models
-` -- gif_name `: Customize GIF file name
--- gif_fps: Set the frame rate of GIF (default: 30)
--- renderw_width: Set rendering width (default: 640)
--- render_ceight: Set rendering height (default: 480)
--- render_made: rendering mode, "rgc_array" is used to generate GIFs, "human" is used for real-time viewing
--- max_steps: Set the maximum number of steps per turn (overwrite the value in the configuration file)
-` -- skip_frames `: Set the number of frames skipped in GIF to reduce file size (default: 2)

## Example

1. Evaluate the best model using default parameters:

```bash
python enjoy_space_mining.py --exp_path /home/user/experiments/space_mining_experiment
```

2. Evaluate models for specific iterations and samples:

```bash
python enjoy_space_mining.py --exp_path /home/user/experiments/space_mining_experiment --iteration 2 --sample 1
```

3. Generate high-quality GIFs (higher resolution, higher frame rate):

```bash
python enjoy_space_mining.py --exp_path /home/user/experiments/space_mining_experiment --render_width 1280 --render_height 720 --gif_fps 60 --skip_frames 1
```

4. Real time viewing of model behavior without generating GIFs:

```bash
python enjoy_space_mining.py --exp_path /home/user/experiments/space_mining_experiment --render_mode human
```

5. Use different random seeds for evaluation:

```bash
python enjoy_space_mining.py --exp_path /home/user/experiments/space_mining_experiment --seed 100
```

## The Importance of Random Seeds

Using different random seeds can help evaluate the robustness of the model under different initial conditions. Each evaluation will use consecutive values starting from the specified seed. For example, if ` -- seed 42-- episodes 5 ` is set, seeds 42, 43, 44, 45, 46 will be used for five rounds of evaluation.

## GIF file

The generated GIF file will be saved in two locations:
1. In the directory where the model is located
2. Under the current working directory

This will make it easier for you to find the generated GIF files in different places.

## Troubleshooting

If encountering problems:

1. Ensure that all necessary dependencies have been installed:
```bash
pip install stable-baselines3 gymnasium numpy imageio
```

2. Check if the model file exists. Model files are usually named 'model. zip'.

If the environment registration fails, it may be necessary to directly import the environment file. The script will attempt to automatically handle this situation.

When there is insufficient memory, you can reduce the rendering resolution or increase the 'skip_frames' value.