"""Reports the versions bifrost depends on and the devices JAX can see.

Run it after installing:  python examples/check_install.py
"""

# --8<-- [start:check]
import importlib.metadata as metadata
import platform

import jax

print(f"{'python':18}", platform.python_version())
for name in ("bifrost-photonics", "gdsfactoryx", "fdtdx", "jax", "jaxlib"):  # distribution names, as pip lists them
    print(f"{name:18}", metadata.version(name))
print(f"{'devices':18}", jax.devices())  # the devices of JAX's backend, e.g. [CudaDevice(id=0)] or [CpuDevice(id=0)]
# --8<-- [end:check]
