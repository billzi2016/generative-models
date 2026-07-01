# test

本目录放仓库级 dry-run / unittest。

测试目标：

- 验证 Python 文件语法正确。
- 验证 HDF5 latent 读写逻辑。
- 验证 DDPM / DiT / Flow Matching / GAN 的模型前向 shape。
- 验证脚本不会依赖真实 DAF 大数据才能完成基础检查。

测试不做的事情：

- 不跑真实训练。
- 不下载模型权重。
- 不写大 checkpoint。
- 不读取 `dataset/raw/fullMin256` 全量图片。

## 运行方式

```bash
python -m unittest discover -s test -p "test_*.py"
```

或者：

```bash
bash test/run_dry_tests.sh
```

测试临时文件写入：

```text
test/tmp/
```

该目录已加入 `.gitignore`。
