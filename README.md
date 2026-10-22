# iPreGraph

iPreGraph is a C++ implementation of filtered approximate nearest neighbor
search with squared Euclidean (L2) distance. It combines graph traversal with
attribute indexes for numerical range and categorical label predicates.

The repository contains a two-attribute quadtree implementation and legacy
segment-tree variants using Z-order and label filtering. Programs in `tests/`
are command-line builders and benchmark drivers, rather than unit tests.

## Build

Use Linux with a C++17 compiler, CMake 3.23.1 or newer, and OpenMP support.
The current configuration uses GCC-specific flags, POSIX headers, and x86
optimizations. Native Windows builds require platform and build changes.

```bash
cmake -S . -B build
cmake --build build --parallel 8
```

Run these commands from the repository root. Executables are written to
`build/tests/`. HNSW headers are bundled in `include/`; no external ANN library
installation is required.

The current configuration sets Debug flags in `tests/CMakeLists.txt` and adds
optimization flags globally. Review both CMake files before reporting benchmark
results; selecting a build type alone does not remove these settings. Newer
compilers may require updating legacy compiler flags.

## Programs

| Program | Purpose |
| --- | --- |
| `buildindex-quadtree` | Build a quadtree and graph layers; save vectors, attributes, nodes, and edges in one index. |
| `search_multi-quadtree` | Benchmark two numerical attributes combined with AND. |
| `buildindex` | Build a legacy segment-tree graph index over the supplied vector order. |
| `search_multi-zorder` | Search two numerical attributes using Morton/Z-order bounds. |
| `search_multi-zorder-label` | Search label OR and numerical range AND predicates using `(z_code, label)` order. |
| `search_multi-or-json` | Search categorical label OR predicates supplied as JSON. |

`tests/search_mixed.cpp` is a standalone mixed label/numerical driver using
`(label, z_code)` order. It is not registered as a CMake target. Its optional
ground-truth loader currently receives fixed arguments of 1,000 queries and 10
neighbors; adapt these before using other ground-truth sizes.

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
./build/tests/buildindex-quadtree \
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
./build/tests/search_multi-quadtree \
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
./build/tests/buildindex \
  --data_path data/base_sorted.bin \
  --index_file indexes/segment.bin \
  --M 16 --ef_construction 250 --threads 8
```

The legacy index and `.qidx` formats are not interchangeable. Keep `--M`
consistent between construction and search.

For `search_multi-zorder`, sort vectors and their two numerical attributes by
the Morton codes computed by `CalculateMultiAttributeZOrder` before building.
Use non-negative integer attributes and regenerate ground-truth IDs for the
sorted row order. Query ranges use the same four-value layout as the quadtree.

```bash
./build/tests/search_multi-zorder \
  --data_path data/base_sorted.bin --query_path data/query.bin \
  --attribute_file data/attributes_sorted.json \
  --range_saveprefix data/ranges.json \
  --groundtruth_saveprefix data/groundtruth.json \
  --index_file indexes/segment.bin \
  --result_saveprefix results/zorder_ --M 16 --query_K 10
```

Legacy EF lists are defined in the driver source files. CSV suffixes and metric
columns vary by search method; consult the corresponding search header when
processing results.

For label OR search, `search_multi-or-json` takes `--attr_json`, `--label_json`,
and `--groundtruth_json`, along with vector, index, result, `--M`, and `--query_K`
arguments. Attributes must be ordered by the selected label column because
matching label blocks are located by binary search. `--attr_num` defaults to
`3`; `--label_col` defaults to `0`. Query labels are a JSON array of label lists.

`search_multi-zorder-label` expects three attribute columns: a label in column
zero and two numerical attributes. Sort rows by `(z_code, label)`. Each query
contains a label list followed by one closed range per numerical attribute:

```json
[[[3, 7], [10, 20], [100, 200]]]
```

This means `(label = 3 OR label = 7) AND (10 <= a0 <= 20) AND
(100 <= a1 <= 200)`. The standalone mixed driver uses the same predicate shape
but requires `(label, z_code)` row order. Keep vectors, attributes, index rows,
and ground-truth IDs aligned whenever data is reordered.

## Code map

| File | Responsibility |
| --- | --- |
| `include/quadtree_index.h` | Tree partitioning, persistence, predicate flags, and filtered traversal. |
| `include/quadtree_graph_builder.h` | Bottom-up graph construction and a reusable worker pool. |
| `include/quadtree_cli.h` | Quadtree options and validated vector/JSON/CSV input. |
| `include/quadtree_progress.h` | Construction progress reporting. |
| `include/construction.h` | Legacy segment-tree graph construction. |
| `include/utils.h`, `include/utils_multi.h` | Legacy loaders, attribute structures, and Morton encoding. |
| `include/iRG_search*.h` | Legacy filtered search variants. |
| `include/searcher.hpp`, `include/memory.hpp` | Search containers, aligned allocation, and prefetch helpers. |
| `include/hnswlib.h`, `include/hnswalg.h`, `include/space_*.h` | Bundled HNSW interfaces and distance kernels. |

## License

The repository includes the Apache License, Version 2.0; see [LICENSE](LICENSE).
