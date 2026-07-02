# test

This directory stores repository-level dry-run tests and unit tests.

Test goals:

- verify Python file syntax
- verify HDF5 latent read / write logic
- verify forward-pass tensor shapes for DDPM / DiT / Flow Matching / GAN
- verify scripts can pass basic checks without requiring the full DAF dataset

What these tests do not do:

- run real training
- download model weights
- write large checkpoints
- read the full `dataset/raw/fullMin256` image tree

## How To Run

```bash
python -m unittest discover -s test -p "test_*.py"
```

Or:

```bash
bash test/run_dry_tests.sh
```

Temporary test files are written to:

```text
test/tmp/
```

That directory is already included in `.gitignore`.
