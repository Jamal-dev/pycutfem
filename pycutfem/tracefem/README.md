# Trace-FEM Link Domains

`pycutfem.tracefem.TraceLinkInterface` and the UFL measure `dTraceLink`
describe discrete trace/link entities. They are intended for line integrations
whose traces are supplied by explicit station or quadrature tables, such as
cohesive links and finite-aperture interface elements.

This is deliberately separate from `NonMatchingInterface`. A non-matching
interface is a common refinement of two mesh facets with one owner element per
side. A trace-link domain is an element-like integration row with its own DOF
union, sided maps, normals, weights, owner ids and optional quadrature state.

The backend table contract is validated by `TraceLinkInterface` and
`DofHandler.precompute_trace_link_factors`. Forms assembled with `dTraceLink`
run through the same python, jit and cpp kernel machinery as other UFL forms,
but the domain type remains `trace_link` so quadrature-state layouts and error
messages stay explicit.

## Fracture Networks

`TraceStationEntity2D` and `TraceFractureNetwork2D` provide a fixed-mesh
topology layer for 2D fracture/link rows. A station entity stores two explicit
traces:

- negative-side station coordinates and field DOFs;
- positive-side station coordinates and field DOFs;
- optional owner ids for element-wise material data;
- an optional `state_source_id` used when transferring quadrature history after
  a topology update.

The network does not allocate DOFs and does not remesh. Inserting or extending a
fracture is a pure topology operation: the caller selects or creates the side
DOFs, builds a new `TraceStationEntity2D`, and converts the network to a
`TraceLinkInterface` with `to_trace_link_interface()`. The resulting object is
assembled with:

```python
dTraceLink(metadata={"trace": trace_link, "quadrature": "lobatto", "q": 2})
```

This keeps fracture propagation separate from `NonMatchingInterface`. A
non-matching interface still means a facet/refinement integration domain; a
trace-fracture network means element-like trace rows with their own station
shape functions and quadrature state.

`plan_fracture_extensions_from_damage()` is a deterministic geometry planner
for propagation experiments. It returns extension plans for entities whose
damage history exceeds the configured threshold. It deliberately returns
geometry only; production code must still provide the new side DOFs explicitly.

`transfer_trace_quadrature_state()` remaps entity-indexed state arrays by
`entity_id` and `state_source_id`. Existing entities are copied exactly, new
entities copy from their source when present, and unrelated inserted entities
are initialized from the supplied default.
