# Getting started

A first simulation end to end, following
[`examples/fdtdx_elevator_coupler.py`](https://github.com/Sephiram591/bifrost/blob/main/examples/fdtdx_elevator_coupler.py).

1. Build a component with gdsfactory and a `LayerStack`.
2. Create the device: `bf.StackDevice.from_layer_stack(component, layer_stack, materials=..., background=...)`.
3. Get the simulation volume and place sources and detectors on the ports
   (see [Layer stacks and StackDevice](guide/stack-device.md)).
4. Place everything with `fdtdx.place_objects`, merge `bf.init_stack_params(objects)` into the parameters and apply
   them with `bf.apply_stack` instead of `fdtdx.apply_params`.
5. Run with `fdtdx.run_fdtd`.
6. To set up and inspect the run interactively, open `bf.view(stack)` (see [Viewer](viewer.md)).

Run simulations on a GPU. The complete script with a sweep is in [Examples](examples/elevator-coupler.md).
