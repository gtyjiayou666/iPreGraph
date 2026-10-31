#!/usr/bin/env python3
"""Build tiny real indexes and check every query driver against exact neighbors.

Uses only the Python standard library. Run after building the C++ executables:
  python tests/cli_smoke_test.py --bin-dir build/tests --work-dir build/cli-smoke
These are correctness smoke checks, not recall or throughput benchmarks.
"""

import argparse
import csv
import json
import math
from pathlib import Path
import struct
import subprocess
import sys


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def write_vectors(path, vectors):
    with path.open("wb") as stream:
        stream.write(struct.pack("<ii", len(vectors), len(vectors[0])))
        for vector in vectors:
            stream.write(struct.pack("<" + "f" * len(vector), *vector))


def write_json(path, value):
    path.write_text(json.dumps(value, separators=(",", ":")), encoding="utf-8")


def ground_truth(base, queries, allowed):
    result = []
    for query, ids in zip(queries, allowed):
        ranked = sorted((sum((a - b) ** 2 for a, b in zip(query, base[pid])), pid)
                        for pid in ids)
        require(ranked, "smoke query must have a qualifying point")
        result.append([ranked[0][1]])
    return result


class Smoke:
    def __init__(self, binaries, directory):
        self.binaries = binaries.resolve()
        self.directory = directory.resolve()
        self.directory.mkdir(parents=True, exist_ok=True)

    def executable(self, name, required=True):
        for path in (self.binaries / name, self.binaries / (name + ".exe")):
            if path.is_file():
                return path
        require(not required, "missing executable: " + name)
        return None

    def run(self, name, arguments, log_name=None):
        command = [str(self.executable(name)), *map(str, arguments)]
        completed = subprocess.run(command, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True,
                                   encoding="utf-8", errors="replace", timeout=120)
        log = self.directory / ((log_name or name) + ".log")
        log.write_text(completed.stdout, encoding="utf-8")
        require(completed.returncode == 0,
                f"{name} failed ({completed.returncode}); see {log}\n{completed.stdout[-3000:]}")

    def check_csv(self, path, columns, name, expected_rows=None, expected_recall=1.0):
        require(path.is_file(), f"{name}: missing result CSV {path}")
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.reader(stream))
        require(rows, f"{name}: empty result CSV")
        if expected_rows is not None:
            require(len(rows) == expected_rows, f"{name}: expected {expected_rows} EF rows, got {len(rows)}")
        for row in rows:
            require(len(row) == columns, f"{name}: unexpected CSV layout {row}")
            values = list(map(float, row))
            require(math.isfinite(values[0]) and values[0] > 0, f"{name}: invalid EF")
            require(math.isclose(values[1], expected_recall, abs_tol=1e-6),
                    f"{name}: tiny-fixture recall must be {expected_recall}, got {values[1]}")
            # Very short runs can round to zero at the platform timer resolution.
            # In that case positive infinity is valid here; throughput is not tested.
            require(not math.isnan(values[2]) and values[2] > 0, f"{name}: invalid QPS")
            require(all(math.isfinite(value) and value >= 0 for value in values[3:]),
                    f"{name}: invalid traversal counters")
        print(f"PASS: {name}: {len(rows)} EF rows, expected recall={expected_recall:g}", flush=True)

    def execute(self):
        work = self.directory
        # Monotone attributes preserve both Morton and label/Morton row orders.
        # Distinct vector distances avoid ambiguous nearest-neighbor ties.
        base = [[float(i), float(i * i) / 4] for i in range(8)]
        queries = [base[i] for i in (0, 3, 6)]
        attributes = [[i, i] for i in range(8)]
        labels = [i // 2 for i in range(8)]
        bounds = [[[0, 7], [0, 7]], [[2, 5], [2, 5]], [[6, 7], [6, 7]]]
        label_sets = [[0, 1, 2, 3], [1, 2], [3]]
        numeric_ids = [[pid for pid, point in enumerate(attributes)
                        if all(lo <= value <= hi for value, (lo, hi) in zip(point, rectangle))]
                       for rectangle in bounds]
        label_ids = [[pid for pid, label in enumerate(labels) if label in requested]
                     for requested in label_sets]
        mixed_ids = [sorted(set(numeric) & set(categorical))
                     for numeric, categorical in zip(numeric_ids, label_ids)]
        write_vectors(work / "base.bin", base)
        write_vectors(work / "queries.bin", queries)
        write_json(work / "attributes.json", attributes)
        write_json(work / "ranges.json", bounds)
        write_json(work / "numeric_gt.json", ground_truth(base, queries, numeric_ids))
        write_json(work / "label_attributes.json", [[labels[i], *attributes[i]] for i in range(8)])
        write_json(work / "labels.json", label_sets)
        write_json(work / "label_gt.json", ground_truth(base, queries, label_ids))
        write_json(work / "mixed_queries.json", [[requested, *rectangle]
                   for requested, rectangle in zip(label_sets, bounds)])
        write_json(work / "mixed_gt.json", ground_truth(base, queries, mixed_ids))

        self.run("build_segment_index", ["--data_path", work / "base.bin", "--index_file", work / "segment.bin",
                                "--M", 16, "--ef_construction", 32, "--threads", 2])
        require((work / "segment.bin").stat().st_size > 0, "segment index is empty")
        common = ["--data_path", work / "base.bin", "--query_path", work / "queries.bin",
                  "--index_file", work / "segment.bin", "--M", 16, "--query_K", 1]
        for method in ("zorder", "irange"):
            prefix = work / (method + "_")
            self.run("search_zorder", [*common, "--attribute_file", work / "attributes.json",
                     "--range_saveprefix", work / "ranges.json", "--groundtruth_saveprefix", work / "numeric_gt.json",
                     "--result_saveprefix", prefix, "--attr_num", 2, "--method", method], method)
            suffix = "mixed-zorder_ways.csv" if method == "zorder" else "mixed-irange.csv"
            self.check_csv(Path(str(prefix) + suffix), 5, method, 32)

        self.run("build_quadtree_index", ["--data_path", work / "base.bin", "--attribute_file", work / "attributes.json",
                 "--index_file", work / "tree.qidx", "--M", 16, "--ef_construction", 32,
                 "--leaf_size", 2, "--max_depth", 8, "--threads", 2, "--progress", 0])
        require((work / "tree.qidx").stat().st_size > 0, "quadtree index is empty")
        prefix = work / "quadtree_"
        self.run("search_quadtree", ["--index_file", work / "tree.qidx", "--query_path", work / "queries.bin",
                 "--range_saveprefix", work / "ranges.json", "--groundtruth_saveprefix", work / "numeric_gt.json",
                 "--result_saveprefix", prefix, "--query_K", 1, "--ef", "16,8", "--recall_stop", 0,
                 "--entry_points", 5, "--edge_limit", 16, "--seed", 42])
        self.check_csv(Path(str(prefix) + "mixed-quadtree.csv"), 3, "quadtree", 2)

        prefix = work / "label"
        self.run("search_label", [*common, "--attr_json", work / "label_attributes.json",
                 "--label_json", work / "labels.json", "--groundtruth_json", work / "label_gt.json",
                 "--result_saveprefix", prefix, "--attr_num", 3, "--label_col", 0])
        self.check_csv(Path(str(prefix) + "_label_or.csv"), 5, "label OR", 19)

        prefix = work / "zorder_label"
        self.run("search_zorder_label", [*common, "--attribute_file", work / "label_attributes.json",
                 "--range_saveprefix", work / "mixed_queries.json", "--groundtruth_saveprefix", work / "mixed_gt.json",
                 "--result_saveprefix", prefix])
        self.check_csv(Path(str(prefix) + "_zorder_label_memory.csv"), 5, "Z-order + label", 1)

        if self.executable("search_mixed", required=False):
            prefix = work / "mixed"
            self.run("search_mixed", [*common, "--attr_json", work / "label_attributes.json",
                     "--query_json", work / "mixed_queries.json", "--groundtruth_json", work / "mixed_gt.json",
                     "--result_saveprefix", prefix, "--attr_num", 3, "--label_col", 0])
            self.check_csv(Path(str(prefix) + "_mixed.csv"), 5, "mixed", 10)
            # An absent label produces an empty candidate interval list. The
            # driver must return an empty result instead of indexing that list.
            write_json(work / "empty_queries.json", [[[99], [0, 7], [0, 7]] for _ in queries])
            write_json(work / "empty_gt.json", [[-1] for _ in queries])
            prefix = work / "mixed_empty"
            self.run("search_mixed", [*common, "--attr_json", work / "label_attributes.json",
                     "--query_json", work / "empty_queries.json", "--groundtruth_json", work / "empty_gt.json",
                     "--result_saveprefix", prefix, "--attr_num", 3, "--label_col", 0], "mixed_empty")
            self.check_csv(Path(str(prefix) + "_mixed.csv"), 5, "mixed absent label", 1, 0.0)
        else:
            print("SKIP: optional search_mixed executable was not built", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        Smoke(arguments.bin_dir, arguments.work_dir).execute()
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
        print("FAIL:", error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
