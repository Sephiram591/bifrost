import importlib.util
import pathlib

import pytest

# opt-in timing of the StackDevice geometry steps and the FDTD solve
# (FDTDX_TIMINGS=<name> or --fdtdx-timings=<name>; see fdtdx_timing_plugin.py),
# loaded by path so it works with any import mode
_spec = importlib.util.spec_from_file_location(
    "fdtdx_timing_plugin", pathlib.Path(__file__).with_name("fdtdx_timing_plugin.py")
)
assert _spec is not None and _spec.loader is not None
_fdtdx_timing = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fdtdx_timing)


def pytest_addoption(parser: pytest.Parser) -> None:
    """Adds the options of the fdtdx timing plugin."""
    _fdtdx_timing.pytest_addoption(parser)


def pytest_configure(config: pytest.Config) -> None:
    """Registers the fdtdx timing plugin when it is turned on."""
    _fdtdx_timing.configure(config)


@pytest.fixture(scope="session", autouse=True)
def activate_generic_pdk() -> None:
    """Activate the generic PDK for all tests."""
    from gdsfactory.gpdk import PDK

    PDK.activate()


@pytest.fixture
def kcl(request: pytest.FixtureRequest):  # noqa: ANN201
    """Create a per-test KCLayout namespace for parallel test isolation."""
    import kfactory as kf
    from kfactory.kcell import clean_name

    return kf.KCLayout(name=clean_name(request.node.name))
