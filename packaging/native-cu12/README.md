# jammi-ai-native-cu12

The compiled, in-process Jammi engine built for NVIDIA GPUs — the `jammi_native`
PyO3 extension with CUDA and the fused attention kernels, for compute capability
8.0 and newer (A100, A10, L4, L40S, H100, RTX 30/40-series).

```bash
pip install jammi-ai jammi-ai-native-cu12
```

```python
import jammi

db = jammi.connect("file:///data")   # the embedded engine, on the GPU
```

Install this **or** [`jammi-ai-native`](https://pypi.org/project/jammi-ai-native/),
the CPU build — both provide the `jammi_native` module. The CUDA libraries the
engine links come from NVIDIA's `nvidia-*-cu12` wheels, which this package
depends on; the host needs only an NVIDIA driver. Choose the device with
`[gpu] device` in the engine config; the CPU remains available on a machine with
no GPU.

The dist ships in lockstep at the shared workspace version with `jammi-ai`,
`jammi-ai-native` and `jammi-server`.
