# iPreGraph

iPreGraph is a C++ implementation of filtered approximate nearest neighbor
search with squared Euclidean (L2) distance. It combines graph traversal with
attribute indexes for numerical range and categorical label predicates.

The repository contains a two-attribute quadtree implementation and legacy
segment-tree variants using Z-order and label filtering. Programs in `tests/`
include command-line builders, benchmark drivers, and a small regression test
for the migrated iRangeGraph multi-attribute method.

## Build

Use a C++17 compiler (GCC or Clang on Linux, or a MinGW-w64 target on Windows)
and CMake 3.23.1 or newer. Python 3 is required for the tests, which are enabled
by default. Threads are detected by CMake; OpenMP is not required. Native MSVC
is not supported because the timing code uses `sys/time.h`.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 2
ctest --test-dir build --output-on-failure
```

Run these commands from the repository root. Executables are written to
`build/tests/`. HNSW headers are bundled in `include/`; no external ANN library
installation is required. For Windows, use a MinGW-w64 toolchain and Ninja
(for example, add `-G Ninja -DCMAKE_CXX_COMPILER=g++` to configuration).

Single-configuration builds default to Release. An explicit Debug or
RelWithDebInfo configuration is honored. Enable `-DIPREGRAPH_NATIVE_ARCH=ON`
only when binaries are intended for the build machine's CPU. To build programs
without Python or regression tests, use `-DBUILD_TESTING=OFF`.

GitHub Actions builds and runs the test suite with both GCC and Clang on Linux.
The tests create small datasets locally; no dataset downloads are needed.

## Programs

| Program | Source | Purpose |
| --- | --- | --- |
| `build_quadtree_index` | `tests/build_quadtree_index.cpp` | Build a quadtree and graph layers; save vectors, attributes, nodes, and edges in one index. |
| `search_quadtree` | `tests/search_quadtree.cpp` | Benchmark two numerical attributes combined with AND. |
| `build_segment_index` | `tests/build_segment_index.cpp` | Build a legacy segment-tree graph index over the supplied vector order. |
| `search_zorder` | `tests/search_zorder.cpp` | Search numerical attributes using Morton/Z-order bounds; choose Z-order or probabilistic iRangeGraph traversal. |
| `search_zorder_label` | `tests/search_zorder_label.cpp` | Search label OR and numerical range AND predicates using `(z_code, label)` order. |
| `search_label` | `tests/search_label.cpp` | Search categorical label OR predicates supplied as JSON. |
| `search_mixed` | `tests/search_mixed.cpp` | Search label OR and numerical range AND predicates using `(label, z_code)` order. |

`search_mixed` uses `(label, z_code)` row order and is built from
`tests/search_mixed.cpp`. Ground truth is optional; when supplied, its shape must
match the loaded query count and `--query_K`.

## Quadtree input formats

Point IDs are zero-based base-vector row numbers. Attributes must use the same
row order as the base vectors. The quadtree preserves original IDs and requires
no Z-order sorting.

### Vectors

Both formats below use little-endian values:

| Format | Layout |
| --- | --- |
| `fvecs` | Each vector has an `int32` dimension followed by that many `float32` components. |
| `bin` | One `int32` count and one `int32` dimension, followed by all `float32` vectors in row-major order. |

`--data_format auto` and `--query_format auto` select `fvecs` for the `.fvecs`
extension and `bin` for other extensions. Set the format explicitly if the
extension does not match the contents. Base and query dimensions must match.

### Attributes and ranges

Attributes contain two finite numbers per base vector. Flat and nested numeric
JSON arrays are accepted. For example, three attribute rows are:

```json
[[10, 100], [20, 200], [30, 300]]
```

Ranges contain exactly four numbers per query in the order
`xmin, xmax, ymin, ymax`. Bounds are inclusive and combined with AND.
One query rectangle can be written as:

```json
[[[10, 30], [100, 250]]]
```

This selects points with the first attribute in `[10, 30]` and the second in
`[100, 250]`. Nested arrays are flattened in row-major order.

### Ground truth

Provide exactly `query_count * query_K` integer IDs in flat or nested JSON
arrays. Each query row contains its exact nearest qualifying neighbors. Use `-1`
to pad rows with fewer than `query_K` matches. Other IDs must exist, be unique
within a row, and satisfy that query's rectangle. Recall is divided by
`query_count * query_K`, including padded slots.

## Quadtree quick start

### Build an index

Create the index output directory, then run:

```bash
mkdir -p indexes
./build/tests/build_quadtree_index \
  --data_path data/base.fvecs \
  --attribute_file data/attributes.json \
  --index_file indexes/tree.qidx \
  --M 16 --ef_construction 100 \
  --leaf_size 32 --max_depth 32 --threads 8
```

As an alternative to JSON, use `--attr_path data/attributes.csv` with two numeric
columns per row and no header. Supply exactly one attribute input option.

The tree partitions points by count and constructs approximate relative
neighborhood graphs from leaves to root. Graph edges are stored by tree depth
and original point ID. The saved `.qidx` contains vectors and attributes, so
queries do not need the original base files.

| Build option | Default | Meaning |
| --- | --- | --- |
| `--M` | `16` | Maximum graph degree. |
| `--ef_construction` | `100` | Candidate search budget during construction. |
| `--leaf_size` | `32` | Point-count threshold for a leaf. |
| `--max_depth` | `32` | Maximum tree depth. |
| `--threads` | `1` | Construction worker count. |
| `--progress` | `1` | Set to `0` to disable construction progress. |

### Run a query benchmark

```bash
./build/tests/search_quadtree \
  --index_file indexes/tree.qidx \
  --query_path data/query.fvecs \
  --range_saveprefix data/ranges.json \
  --groundtruth_saveprefix data/groundtruth.json \
  --result_saveprefix results/run_ \
  --query_K 10 --ef 100,70,50,30,10 \
  --entry_points 5 --edge_limit 16 --seed 42 \
  --recall_stop 0
```

Despite their names, `--range_saveprefix` and `--groundtruth_saveprefix` take
complete JSON file paths. `--result_saveprefix` is an output prefix: this example
creates `results/run_mixed-quadtree.csv`, including its parent directory.

Each output row has no header and contains `ef,recall,qps`. The default
`--recall_stop` is `0.9`: the driver stops after writing the first row below that
recall. Set it to `0` to evaluate every supplied EF value. Larger EF values allow
a wider search beam and generally increase query cost.

`--entry_points` defaults to `5` distinct qualifying starting points.
`--edge_limit` defaults to the index's `M` and is shared across tree levels for
each expanded point. The query seed is `seed + query ID`, reused across EF
values. Search can traverse out-of-range points for up to two DFS hops to find
qualifying candidates.

Reported search time covers graph traversal. Eligibility preprocessing,
entry-point sampling, loading, and result evaluation are outside the timed
region. Optional `--data_path`, `--attribute_file`, and `--M` arguments check
consistency with the saved index. Both quadtree programs support `--help`.

## Legacy segment-tree workflow

Legacy programs read vectors in the `bin` layout described above. Build the
index over the exact vector order used by the search driver:

```bash
./build/tests/build_segment_index \
  --data_path data/base_sorted.bin \
  --index_file indexes/segment.bin \
  --M 16 --ef_construction 250 --threads 8
```

The legacy index and `.qidx` formats are not interchangeable. Keep `--M`
consistent between construction and search.

For `search_zorder`, sort vectors and their numerical attributes by
the Morton codes computed by `CalculateMultiAttributeZOrder` before building.
Use non-negative integer attributes and regenerate ground-truth IDs for the
sorted row order. The default is two attributes; set `--attr_num A` for other
dimensions. Each query supplies `A` inclusive `[low, high]` pairs combined with
AND, so flattened query rows contain `2 * A` integers.

```bash
./build/tests/search_zorder \
  --data_path data/base_sorted.bin --query_path data/query.bin \
  --attribute_file data/attributes_sorted.json \
  --range_saveprefix data/ranges.json \
  --groundtruth_saveprefix data/groundtruth.json \
  --index_file indexes/segment.bin \
  --result_saveprefix results/zorder_ --M 16 --query_K 10
```

### iRangeGraph multi-attribute search

The same executable exposes the migrated iRangeGraph method with
`--method irange`. Omitting `--method` retains the existing `zorder` method.
For example, with four numerical attributes:

```bash
./build/tests/search_zorder \
  --data_path data/base_sorted.bin --query_path data/query.bin \
  --attribute_file data/attributes_4d.json --attr_num 4 \
  --range_saveprefix data/ranges_4d.json \
  --groundtruth_saveprefix data/groundtruth.json \
  --index_file indexes/segment.bin \
  --result_saveprefix results/irange_ --M 16 --query_K 10 \
  --method irange
```

This method uses the existing Morton row ordering and segment-tree graph.
All numerical intervals must hold for a point to enter the result heap.
Nonmatching points inside the Morton cover may be explored as transit vertices,
with acceptance probability `1 / (1 + exp(step))`; a matching point resets
that step counter. `MaxStep` defaults to 20. This is an approximate search and
preserves the source implementation's early stopping rule.

The header exposes `search_irange`, `TopDown_nodeentries_search_mutil`,
`SelectEdgeMutil`, and `setprob` on `iPreGraph_Search_Zorder`. The original
`mutil` spelling is kept for method compatibility. The low-level search initializes
its probability table when needed. Its optional seed controls entry selection;
transit sampling uses the C `rand()` state. Set both when comparing seeded runs.

Output is `<prefix>mixed-irange.csv` with no header and columns
`ef,recall,qps,dco,hops`. The EF sweep stops below recall 0.9, as in the source
method. Predicate flags are cached outside the timed loop and require about
`base_count * query_count` bytes per query domain. Timing includes Morton bounds,
entry selection, and graph traversal; it excludes flag preparation and evaluation.

Legacy EF lists are defined in the driver source files. CSV suffixes and metric
columns vary by search method; consult the corresponding search header when
processing results.

For label OR search, `search_label` takes `--attr_json`, `--label_json`,
and `--groundtruth_json`, along with vector, index, result, `--M`, and `--query_K`
arguments. Attributes must be ordered by the selected label column because
matching label blocks are located by binary search. `--attr_num` defaults to
`3`; `--label_col` defaults to `0`. Query labels are a JSON array of label lists.

`search_zorder_label` expects three attribute columns: a label in column
zero and two numerical attributes. Sort rows by `(z_code, label)`. Each query
contains a label list followed by one closed range per numerical attribute:

```json
[[[3, 7], [10, 20], [100, 200]]]
```

This means `(label = 3 OR label = 7) AND (10 <= a0 <= 20) AND
(100 <= a1 <= 200)`. The standalone mixed driver uses the same predicate shape
but requires `(label, z_code)` row order. Keep vectors, attributes, index rows,
and ground-truth IDs aligned whenever data is reordered.

## Tests

The self-contained test creates synthetic 2D and 4D data, including Morton codes
wider than 64 bits. It checks exact full-range results on complete graphs,
predicate validity, duplicate IDs, empty results, input validation, and CSV output.
It also checks aligned allocation and deallocation. The CLI smoke test builds
actual segment-tree and quadtree indexes and queries all search variants against
independent exact nearest-neighbor results, including an empty mixed predicate.
These checks do not require downloaded datasets or measure performance.

```bash
cmake -S . -B build -DBUILD_TESTING=ON
cmake --build build --parallel 2
ctest --test-dir build --output-on-failure
```

## Code map

Project-owned source files use lowercase `snake_case` names. Headers use `.h`,
program sources use `.cpp`, and regression tests use the `_test.cpp` suffix.
Names identify the operation or component: `search_*`, `build_*`, `*_builder`,
and `*_utils`. Bundled HNSW headers retain their upstream names. Executable names
match their source filenames without the `.cpp` suffix (plus `.exe` on Windows).

| File | Responsibility |
| --- | --- |
| `include/quadtree_index.h` | Tree partitioning, persistence, predicate flags, and filtered traversal. |
| `include/quadtree_graph_builder.h` | Bottom-up graph construction and a reusable worker pool. |
| `include/quadtree_cli.h` | Quadtree options and validated vector/JSON/CSV input. |
| `include/quadtree_progress.h` | Construction progress reporting. |
| `include/segment_graph_builder.h` | Legacy segment-tree graph construction. |
| `include/segment_tree_utils.h`, `include/multi_attribute_utils.h` | Legacy loaders, attribute structures, and Morton encoding. |
| `include/search_zorder.h`, `include/search_zorder_label.h`, `include/search_label.h`, `include/search_mixed.h`, `include/search_multi.h` | Legacy filtered search variants. |
| `include/search_containers.h`, `include/memory_utils.h` | Search containers, aligned allocation, and prefetch helpers. |
| `include/hnswlib.h`, `include/hnswalg.h`, `include/space_*.h` | Bundled HNSW interfaces and distance kernels. |

## License

The repository includes the Apache License, Version 2.0; see [LICENSE](LICENSE).
