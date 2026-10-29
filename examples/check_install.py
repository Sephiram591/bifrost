"""Reports the versions bifrost depends on and the devices JAX can see.

Run it after installing:  python examples/check_install.py
"""

# --8<-- [start:check]
import importlib.metadata as metadata
import platform

import jax

print(f"{'python':10}", platform.python_version())
for name in ("bifrost", "gdsfactory", "fdtdx", "jax", "jaxlib"):
    print(f"{name:10}", metadata.version(name))
print(f"{'devices':10}", jax.devices())  # [CudaDevice(id=0)] on a GPU node, [CpuDevice(id=0)] otherwise
# --8<-- [end:check]
