# Probe regularization 检查（2026-10-02）

## 已确认的证据

输入：`F:/AgentWorkspaces/Temporary/model_iter0200.hdf5` 和
`20261002_163719_ptyrad_log.txt`。原始文件未修改。

- HDF5 配置为 `state=True, weight=0.1, mode=primary, aperture_fraction=0.85`。
- 第 200 轮保存的每个 batch 的 `loss_probe_reg` 都精确为零，平均值也为零。
- 用保存的 probe 和采样参数重新计算：原始 R = 0.944764，乘权重后为
  **0.0944764**。梯度范数为 **0.00593532**，非零且有限。
- 同样参数下，0400 和 0800 的加权指标分别为 0.0969331 和 0.0110078。
  0200 的粗糙度仍接近 0400；这些不同迭代结果不是受控的收敛对照实验。
- 日志中的环境为 Linux、PyTorch 2.5.1、A100，启用了编译，关闭了参数验证。
  加载位置为 `/home/zehao/anaconda3/envs/ptyrad/lib/python3.12/site-packages/ptyrad/`。

保存的 loss 是迭代过程中 batch 的平均值，重新计算的是迭代结束并施加约束后的值，
因此两者不要求相等。但该文件不支持此前“只是四位小数显示为零”的解释。

## 发现与修复

旧记录逻辑使用不检查长度的 `zip(loss_names, losses)`。若配置有六项而实际
forward 只返回旧版五项，第六项预分配的零会原样保存，且不报错。
回归测试复现了此静默遗漏路径；现在会在反向传播前明确报错。

另一个独立问题是 forward 返回固定顺序的列表，而记录名称来自配置字典顺序。
关闭参数验证或重新读取 HDF5 时可能发生错位。现在返回顺序与配置名称一致。
此外增加孔径配置完成时的初始加权 loss 日志，并用六位有效数字显示逐项 loss。
正则项数学公式和默认权重未变。

**尚未确认原 Linux 运行的最终根因。** 本地完整实现计算正常；旧安装包或不完整更新
可以解释现象，但仅凭这些文件不能排除原 GPU 编译路径的问题。
本次验证环境为 CPU/PyTorch 2.14.1，已通过 eager、Dynamo eager 和 AOT eager 测试；
未验证原 A100/PyTorch 2.5.1 Inductor，也未重跑实验数据的完整重建。

## 在原重建环境验证与应用

将本次修改后的仓库同步到 Linux，在实际启动重建的 conda 环境中，从仓库根目录运行：

```sh
python -m pip install -e . --no-deps
python tools/check_probe_loss.py /path/to/model_iter0200.hdf5 --device cuda
python tools/check_probe_loss.py /path/to/model_iter0200.hdf5 --device cuda --compile
```

脚本会打印实际加载的 loss 源文件、重新计算的正则项和梯度范数，并检查 forward
确实包含该项。第二次检查还比较编译与非编译的值和梯度。
此 checkpoint 预期加权值约为 0.0944764，不能为零。
若仅编译检查失败，先设置 `compiler_configs: {disable: true}` 重建。
若两者通过，再从原初始化重新运行并比较数据误差、probe 和 object；不能只以 R 降低判断成功。

