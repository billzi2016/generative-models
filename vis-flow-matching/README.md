# vis-flow-matching

This directory is a standalone MNIST Flow Matching visualization demo. The goal is to turn the theory into an animation you can inspect directly.

Target behavior:

```text
Gaussian noise x0 -> ODE flow -> MNIST digit x1
```

The generated GIF uses `10` columns by `6` rows by default:

- each column corresponds to one digit class `0..9`
- each row is a different random sample from the same class
- the animation shows the Flow Matching sampling trajectory from Gaussian noise to a clean digit

## Algorithm

Training uses the most basic Rectified Flow / Flow Matching objective:

```text
x0 ~ N(0, I)
x1 ~ MNIST(label)
t ~ Uniform(0, 1)
xt = (1 - t) * x0 + t * x1
v_target = x1 - x0
v_theta = model(xt, t, label)
loss = MSE(v_theta, v_target)
```

Sampling starts from noise and uses Euler ODE integration:

```text
dx / dt = v_theta(x, t, label)
```

## Training

By default `torchvision.datasets.MNIST(download=True)` downloads MNIST into `dataset/torchvision`.

```bash
python vis-flow-matching/train.py \
  --data-dir dataset/torchvision \
  --output-dir vis-flow-matching/runs/mnist_flow \
  --epochs 100 \
  --batch-size 256 \
  --lr 2e-4 \
  --patience 3 \
  --min-delta 1e-4 \
  --seed 42
```

On Mac, the script prefers `mps`, otherwise it falls back to `cuda` or `cpu`. Training uses the official MNIST `train=True` split for training and the official `train=False` test split for validation, saving `best.pt` according to `val_loss`. The maximum is `100` epochs, but training stops early if `val_loss` fails to improve by more than `1e-4` for `3` consecutive epochs.

Outputs:

```text
vis-flow-matching/runs/mnist_flow/best.pt
vis-flow-matching/runs/mnist_flow/last.pt
vis-flow-matching/runs/mnist_flow/metrics.csv
vis-flow-matching/runs/mnist_flow/metrics.jpg
```

At the end of every epoch, the script appends to `metrics.csv` and overwrites `metrics.jpg`. Titles and axis labels stay in English to avoid matplotlib Chinese-font issues.

By default only `best.pt` and `last.pt` are kept. If you want periodic checkpoints, pass this explicitly:

```bash
--checkpoint-every-epochs 10
```

## Generate The 10x6 GIF

```bash
python vis-flow-matching/make_gif.py \
  --checkpoint vis-flow-matching/runs/mnist_flow/best.pt \
  --output vis-flow-matching/runs/mnist_flow/mnist_flow_10x6.gif \
  --rows 6 \
  --cols 10 \
  --steps 60 \
  --fps 12 \
  --seed 42
```

Output:

```text
vis-flow-matching/runs/mnist_flow/mnist_flow_10x6.gif
```

## Visualization

MNIST conditional Flow Matching sampling animation: each column corresponds to digit `0..9`, each row is a different random sample in that class, and the animation shows how Gaussian noise gradually flows into readable handwritten digits.

![MNIST Flow Matching](runs/mnist_flow/mnist_flow_10x6.gif)

Training curve: the x-axis is epoch and the y-axis is loss, used to inspect whether training and validation errors are decreasing steadily and whether early stopping triggers at a reasonable point.

![MNIST Metrics](runs/mnist_flow/metrics.jpg)

## File Notes

- `model.py`: a compact class-conditional U-Net, taking `x_t, t, label` and predicting the velocity field
- `train.py`: train MNIST Flow Matching
- `make_gif.py`: generate the `10 x 6` GIF from a trained checkpoint

The demo GIF and JPG are committed to git. Checkpoints, CSV files, and config files are still ignored.
