# vis-2d-flowmatching

This directory is a 2D Flow Matching visualization project. The point is to directly watch particles move along a learned velocity field toward a target distribution.

It includes two classic distributions:

- `ring`
- `moons`

Both entry points share the same Flow Matching system:

- `distributions.py`: target-distribution samplers in 2D
- `model.py`: shared MLP velocity field
- `system.py`: training, Euler ODE sampling, GIF rendering, vector-field rendering
- `run_ring.py`: ring entry point
- `run_moons.py`: two-moons entry point

## Equations

Training objective:

```text
x0 ~ N(0, I)
x1 ~ target distribution
t ~ Uniform(0, 1)
xt = (1 - t) * x0 + t * x1
v_target = x1 - x0
loss = MSE(v_theta(xt, t), v_target)
```

Sampling:

```text
dx / dt = v_theta(x, t)
```

## Ring

```bash
python vis-2d-flowmatching/run_ring.py \
  --output-dir vis-2d-flowmatching/runs/ring \
  --epochs 3000 \
  --batch-size 2048 \
  --lr 2e-4 \
  --gif-steps 160 \
  --fps 10 \
  --hold-final-frames 30 \
  --seed 42
```

Outputs:

```text
vis-2d-flowmatching/runs/ring/ring_flow.gif
vis-2d-flowmatching/runs/ring/ring_vector_field.jpg
vis-2d-flowmatching/runs/ring/metrics.jpg
vis-2d-flowmatching/runs/ring/best.pt
```

## Moons

```bash
python vis-2d-flowmatching/run_moons.py \
  --output-dir vis-2d-flowmatching/runs/moons \
  --epochs 3000 \
  --batch-size 2048 \
  --lr 2e-4 \
  --gif-steps 160 \
  --fps 10 \
  --hold-final-frames 30 \
  --seed 42
```

Outputs:

```text
vis-2d-flowmatching/runs/moons/moons_flow.gif
vis-2d-flowmatching/runs/moons/moons_vector_field.jpg
vis-2d-flowmatching/runs/moons/metrics.jpg
vis-2d-flowmatching/runs/moons/best.pt
```

## Visualization

- `*_flow.gif`: the full process of particles moving from Gaussian noise to the target distribution
- `*_vector_field.jpg`: velocity-field snapshots at `t=0.0`, `t=0.5`, and `t=1.0`
- `metrics.jpg`: training loss curve, overwritten in place, with all figure text kept in English

### Ring

Ring sampling animation: 2D points start from Gaussian noise and gradually move along the learned velocity field toward a ring distribution.

![Ring Flow](runs/ring/ring_flow.gif)

Ring vector field: shows the direction and overall flow structure of the 2D velocity field at `t=0.0`, `t=0.5`, and `t=1.0`.

![Ring Vector Field](runs/ring/ring_vector_field.jpg)

Ring training curve: used to judge whether optimization on the ring experiment is converging and whether the loss has reached a stable regime.

![Ring Metrics](runs/ring/metrics.jpg)

### Moons

Moons sampling animation: 2D points gradually flow from Gaussian noise into the target two-moons distribution.

![Moons Flow](runs/moons/moons_flow.gif)

Moons vector field: shows the velocity-field structure of the two-moons experiment at different time slices.

![Moons Vector Field](runs/moons/moons_vector_field.jpg)

Moons training curve: used to inspect the loss trend and final convergence stability of the two-moons experiment.

![Moons Metrics](runs/moons/metrics.jpg)

If training has already finished and you only want to regenerate the GIF and vector-field plots from a checkpoint, you can skip training:

```bash
python vis-2d-flowmatching/run_ring.py \
  --output-dir vis-2d-flowmatching/runs/ring \
  --skip-train \
  --checkpoint vis-2d-flowmatching/runs/ring/best.pt \
  --gif-steps 160 \
  --fps 10 \
  --hold-final-frames 30 \
  --seed 42
```

The demo GIF and JPG are committed to git. Checkpoints, CSV files, and config files are still ignored.
