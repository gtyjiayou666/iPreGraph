# iPreGraph

iPreGraph implements filtered approximate nearest neighbor search with squared
Euclidean (L2) distance. It builds reusable base graphs over a hierarchy of
attribute-based object groups and selects edges during search to compose a
predicate-dedicated graph (PDG).

This repository contains the segment-tree implementation for numeric, label,
and combined label/numeric experiments, the probabilistic iRangeGraph comparison
on the same index, and a two-attribute quadtree implementation for the index
structure comparison. The command-line drivers accept the predicate formats
listed below; they do not provide a general SQL or Boolean-expression parser.

## Build and test

Requirements:

- CMake 3.23.1 or newer and a C++17 compiler.
- GCC or Clang on Linux, or a MinGW-w64 toolchain on Windows.
- Python 3 for the regression tests, enabled by default.

Run from the repository root:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 2
ctest --test-dir build --output-on-failure
```

Executables are written to `build/tests/`. HNSW headers are bundled in `include/`;
no external ANN library installation is required. Threads are detected by CMake;
OpenMP is not required. Native MSVC is not supported because timing uses
`sys/time.h`. On Windows, use a MinGW-w64 toolchain, for example with
`-G Ninja -DCMAKE_CXX_COMPILER=g++`.

Single-configuration builds default to Release. Use `-DBUILD_TESTING=OFF` to
build without Python or regression tests. `-DIPREGRAPH_NATIVE_ARCH=ON` enables
CPU-specific compilation for the build machine. GitHub Actions builds and tests
with both GCC and Clang on Linux.

## Experiment programs

| Program | Purpose | Required base-vector order |
| --- | --- | --- |
| `build_segment_index` | Build hierarchical base graphs over the supplied row order. | The order required by the search driver. |
| `search_zorder` | Numeric range conjunctions; `--method zorder` or `--method irange`. | Morton/Z-order over all numeric attributes. |
| `search_label` | Membership in a query label set. | Ascending label. |
| `search_zorder_label` | Label membership AND numeric range constraints. | `(Morton code, label)`. |
| `search_mixed` | Label membership AND numeric range constraints. | `(label, Morton code)`. |
| `build_quadtree_index` | Build the quadtree comparison index. | Any order; original row IDs are preserved. |
| `search_quadtree` | Two numeric range constraints combined with AND. | The order stored in the quadtree index. |

Program sources have the same names under `tests/`. The quadtree is an
alternative index structure; it is not needed to run the segment-tree methods.
External baselines such as ACORN, NaviX, and Faiss-IVFPQ are not bundled here.

## Prepare the data

### Vectors and IDs

All segment-tree programs use a little-endian binary vector file:

```text
int32 vector_count
int32 dimension
float32 vectors[vector_count][dimension]
```

The quadtree programs also accept `fvecs`, where each vector is preceded by its
own `int32` dimension. Their `--data_format auto` and `--query_format auto` options
select `fvecs` for `.fvecs` files and the binary layout above otherwise.

Object IDs are zero-based base-vector row numbers. Before building a segment-tree
index, sort the base vectors and attributes together and remap ground-truth IDs
to the new row order. Keep query vectors, query predicates, and ground-truth
query rows aligned. Sorting the base does not require changing query vectors or
predicate values. The index builder does not perform attribute sorting.

For Morton sorting, use the encoding in
`encode_morton_key` and compare keys with `morton_key_less`
in `include/multi_attribute_utils.h`. Attribute 0 contributes the least-significant
bit of each interleaved bit plane; multiword keys support more than 64 bits.
Numeric attributes used by the segment-tree drivers must be non-negative integers.

### Attributes and predicates

Numeric-only attributes contain one integer per dimension and object. Flat or
nested JSON arrays are accepted. A two-attribute example is:

```json
[[10, 100], [20, 200], [30, 300]]
```

Each numeric query supplies one inclusive `[low, high]` pair per attribute;
all pairs are combined with AND:

```json
[[[10, 30], [100, 250]]]
```

For label-only search, attributes contain the label column selected by
`--label_col`. With `--attr_num 1 --label_col 0`, an attribute file can be
`[0, 0, 1, 3, 3]`. A query file lists the allowed labels for each query:

```json
[[0, 3], [1]]
```

For hybrid search, the default attribute layout is `[label, a0, a1]`. Each query
contains a label list followed by the inclusive numeric ranges:

```json
[[[3, 7], [10, 20], [100, 200]]]
```

This represents `(label = 3 OR label = 7) AND (10 <= a0 <= 20) AND
(100 <= a1 <= 200)`. `search_mixed` accepts `--attr_num` and `--label_col`;
`search_zorder_label` currently fixes three columns with the label in column 0.

### Ground truth

Ground-truth JSON contains `query_count * query_K` integer IDs, either flat or
nested by query. IDs must refer to the base order used by that index. Use `-1`
for missing results when fewer than `query_K` objects qualify. Reported recall
uses `query_count * query_K` as its denominator, including padded slots.

Ground truth is required by the supplied drivers except `search_mixed`, which
reports recall as `-1` when `--groundtruth_json` is omitted.

## Segment-tree experiments

### Build the index

```bash
mkdir -p indexes results
./build/tests/build_segment_index \
  --data_path data/numeric/base_sorted.bin \
  --index_file indexes/numeric.bin \
  --M 40 --ef_construction 1000 --threads 64
```

Build a separate index for each base-vector ordering used below, changing
`--data_path` and `--index_file` accordingly. Keep `--M` consistent between
construction and search. The segment-tree index stores adjacency lists; search
also requires the base-vector and attribute files. Segment summaries are built
when the index is loaded, rather than stored in the adjacency file.

### Numeric filtering

```bash
./build/tests/search_zorder \
  --data_path data/numeric/base_sorted.bin \
  --query_path data/query.bin \
  --attribute_file data/numeric/attributes_sorted.json \
  --range_saveprefix data/numeric/ranges.json \
  --groundtruth_saveprefix data/numeric/groundtruth.json \
  --index_file indexes/numeric.bin \
  --result_saveprefix results/numeric_ \
  --M 40 --query_K 10 --attr_num 2 --method zorder
```

For four numeric attributes, use `--attr_num 4` with the corresponding sorted
base, attributes, four-attribute predicates, ground truth, and index. The
`--range_saveprefix` and `--groundtruth_saveprefix` arguments are complete JSON
file paths despite their names.

To run the iRangeGraph comparison, use the same command with `--method irange`.
This adaptation uses the same Morton-sorted segment-tree index and applies the
full numeric predicate to results. It probabilistically explores non-qualifying
points within the covering range. It is not an independently built index sorted
by a single numeric attribute.

### Label filtering

Build `indexes/labels.bin` over `data/labels/base_sorted.bin`, sorted by label.
For a file containing only the label attribute:

```bash
./build/tests/search_label \
  --data_path data/labels/base_sorted.bin \
  --query_path data/query.bin \
  --attr_json data/labels/attributes_sorted.json \
  --label_json data/labels/queries.json \
  --groundtruth_json data/labels/groundtruth.json \
  --index_file indexes/labels.bin \
  --result_saveprefix results/labels \
  --M 40 --query_K 10 --attr_num 1 --label_col 0
```

For multi-column attribute files, specify the actual `--attr_num` and label
column. The selected label column must be nondecreasing in base-vector order.

### Hybrid filtering: Z-order first

Build `indexes/hybrid_z.bin` using vectors sorted by `(Morton code, label)`.
The attribute file has three columns: label, numeric attribute 0, numeric
attribute 1.

```bash
./build/tests/search_zorder_label \
  --data_path data/hybrid_z/base_sorted.bin \
  --query_path data/query.bin \
  --attribute_file data/hybrid_z/attributes_sorted.json \
  --range_saveprefix data/hybrid/queries.json \
  --groundtruth_saveprefix data/hybrid_z/groundtruth.json \
  --index_file indexes/hybrid_z.bin \
  --result_saveprefix results/hybrid_z \
  --M 40 --query_K 10
```

### Hybrid filtering: label first

Build `indexes/hybrid_label.bin` using vectors sorted by `(label, Morton code)`.
It can use the same query predicates as the Z-order-first experiment, but requires
its own remapped ground-truth IDs.

```bash
./build/tests/search_mixed \
  --data_path data/hybrid_label/base_sorted.bin \
  --query_path data/query.bin \
  --attr_json data/hybrid_label/attributes_sorted.json \
  --query_json data/hybrid/queries.json \
  --groundtruth_json data/hybrid_label/groundtruth.json \
  --index_file indexes/hybrid_label.bin \
  --result_saveprefix results/hybrid_label \
  --M 40 --query_K 10 --attr_num 3 --label_col 0
```

### Search settings and additional method interfaces

Segment-tree EF lists are defined in the corresponding `tests/search_*.cpp`
files. In particular, `search_zorder_label` currently runs only `ef=20`; edit its
`SearchEF` list and rebuild to obtain a recall/QPS curve. These drivers do not
currently expose an `--ef` command-line option. With ground truth supplied, their
EF sweeps stop after recording a row with recall below 0.90.

The following experiment-related methods remain available in the headers but
are not selected by the current command-line drivers:

| Method | Interface |
| --- | --- |
| Numeric Post-filter | `iPreGraph_Search_Zorder::benchmark_post_filter` in `include/search_zorder.h`. |
| Label Post-filter | `iPreGraph_Search_Label::benchmark_post_filter` in `include/search_label.h`. |
| Label-first hybrid Post-filter | `iPreGraph_Search_Mixed::benchmark_post_filter` in `include/search_mixed.h`. |
| Label range merging | `search_merged_ranges` and `merge_small_gaps` in `include/search_label.h`. |

Calling these methods requires selecting them in a driver and rebuilding. There
is no command-line switch for range merging or tree fanout in the supplied
segment-tree drivers. Removed experimental prototypes and old binary-range
loaders are no longer part of the public source tree.

## Quadtree structure comparison

Quadtree attributes contain exactly two finite numeric values per object. The
index retains the supplied row IDs and stores vectors, attributes, nodes, and
edges together. No Morton sorting is required, and its `.qidx` format is distinct
from the segment-tree adjacency format.

```bash
./build/tests/build_quadtree_index \
  --data_path data/base.fvecs \
  --attribute_file data/attributes.json \
  --index_file indexes/tree.qidx \
  --M 40 --ef_construction 1000 \
  --leaf_size 32 --max_depth 32 --threads 64

./build/tests/search_quadtree \
  --index_file indexes/tree.qidx \
  --query_path data/query.fvecs \
  --range_saveprefix data/ranges.json \
  --groundtruth_saveprefix data/groundtruth.json \
  --result_saveprefix results/quadtree_ \
  --query_K 10 --ef 100,70,50,30,10 \
  --entry_points 5 --edge_limit 40 --seed 42 --recall_stop 0
```

Query rectangles use the same two-attribute JSON layout as numeric filtering.
Ground-truth IDs must refer to the input order used to build this quadtree.
A CSV attribute file with two columns and no header can be supplied through
`--attr_path` instead of `--attribute_file`.

Both quadtree programs support `--help`. Build defaults are `M=16`,
`ef_construction=100`, `leaf_size=32`, `max_depth=32`, and `threads=1`.
Search defaults to five qualifying entry points, `edge_limit=M`, and `seed=42`.
The default `--recall_stop 0.9` ends the EF sweep after the first row below 0.90;
set it to `0` to run the full sweep. `--progress 0` disables build progress.

## Results and measurement

`--result_saveprefix` is an output prefix, not a directory. Parent directories
are created as needed. CSV files have no header:

| Driver or mode | Suffix appended to the prefix | Columns |
| --- | --- | --- |
| `search_zorder --method zorder` | `mixed-zorder_ways.csv` | `ef,recall,qps,dco,hop` |
| `search_zorder --method irange` | `mixed-irange.csv` | `ef,recall,qps,dco,hop` |
| `search_label` | `_label_or.csv` | `ef,recall,qps,dco,hop` |
| `search_zorder_label` | `_zorder_label_memory.csv` | `ef,recall,qps,dco,hop` |
| `search_mixed` | `_mixed.csv` | `ef,recall,qps,dco,hop` |
| `search_quadtree` | `mixed-quadtree.csv` | `ef,recall,qps` |

`dco` and `hop` are the recorded distance-computation and expansion counters per
query. The `mixed` prefix in numeric output names is the loader's query-domain
key; it does not indicate a label/numeric predicate.

Object-level predicate flags and input/index loading are outside the reported
query timer. Segment-summary tests and graph traversal run during search.
Segment-tree timing also includes the range-location and entry-selection work
inside each method's timed interval. The quadtree timer excludes entry-point
sampling. Consult the timing boundaries in the relevant search method when
comparing results. The iRangeGraph benchmark caches predicate flags for all
queries in a domain, requiring approximately `base_count * query_count` bytes.

`build_segment_index` prints graph construction time in seconds before index
serialization. For total process time and peak resident memory on Linux, run
the desired command with `/usr/bin/time -v`. Peak RSS covers the entire process,
including loading. When reporting segment-tree disk usage including vectors,
add the base-vector file size to the saved adjacency-file size; MBRs and bitmaps
are reconstructed in memory. The quadtree file already contains its vectors.

## Regression tests

`irange_multi_test` covers two- and four-attribute predicates, Morton keys wider
than 64 bits, predicate validity, duplicate IDs, empty results, input validation,
and aligned memory allocation. `mixed_range_test` checks direct-edge range
bounds and correct range selection after cross-range transitions.
`cli_smoke_test.py` builds small segment-tree and
quadtree indexes and runs all exposed search modes against independently computed
ground truth. Tests generate their own data and do not measure full-dataset
performance.

## Function naming

Project-owned functions use `snake_case`. Class names, executable names, CLI
options, index formats, and CSV suffixes are unchanged. Bundled HNSW APIs retain
their upstream names.

| Name or prefix | Responsibility |
| --- | --- |
| `search_pdg`, `search_irange`, `search_post_filter` | Run one ANN query. |
| `benchmark_pdg`, `benchmark_irange`, `benchmark_post_filter` | Run the query/EF loops and write benchmark metrics. |
| `select_direct_edges` | Select direct qualifying neighbors using segment summaries. |
| `select_intra_range_edges` | Select neighbors restricted to one position range. |
| `select_cross_range_edges` | Collect qualifying endpoints through bounded DFS. |
| `select_cross_range_edges_from_ranges` | Guide that DFS with an explicit range set. |
| `may_match_*` | Return `true` when a segment cannot be rejected by its summary. |
| `load_*`, `parse_*`, `validate_*` | Read inputs, parse values, and check invariants. |

Function names identify their current roles; earlier prototype names such as
`DFS_new1`, `SelectEdgeMutil`, and `TopDown_nodeentries_search_or_Pro` have been
replaced. Code calling those C++ interfaces must use the new names. Existing
command-line experiment scripts do not need to change.

## Code map

| File | Responsibility |
| --- | --- |
| `include/segment_graph_builder.h` | Segment-tree base-graph construction and serialization. |
| `include/segment_tree_utils.h` | Vector/label loaders, segment trees, and label bitmaps. |
| `include/multi_attribute_utils.h` | Numeric summaries, JSON loaders, and Morton encoding. |
| `include/search_zorder.h` | Numeric iPreGraph search, iRangeGraph comparison, and Post-filter. |
| `include/search_label.h` | Label search, Post-filter, and range merging. |
| `include/search_zorder_label.h` | Z-order-first hybrid search. |
| `include/search_mixed.h` | Label-first hybrid search and Post-filter. |
| `include/quadtree_index.h`, `include/quadtree_graph_builder.h` | Quadtree construction, persistence, and search. |
| `include/quadtree_cli.h`, `include/quadtree_progress.h` | Quadtree input handling and progress reporting. |
| `include/search_containers.h`, `include/memory_utils.h` | Visited flags, aligned allocation, and prefetch helpers. |
| `include/hnswlib.h`, `include/hnswalg.h`, `include/space_*.h` | Bundled HNSW interfaces and distance kernels. |

## License

See [LICENSE](LICENSE) for the Apache License, Version 2.0.
