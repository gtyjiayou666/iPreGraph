// Benchmark numerical AND predicates on a Z-order segment-tree graph index.
// Vectors, attributes, and ground-truth IDs must share the same Morton-sorted order.
// Select --method irange to use probabilistic iRangeGraph traversal.
// --attr_num controls numerical attribute count and defaults to two.

#include "search_zorder.h"

std::unordered_map<std::string, std::string> paths;

int query_K = 10;
int M;
double s;
int attr_num = 2;
std::string method = "zorder";

int main(int argc, char **argv)
{
    for (int i = 1; i < argc; i += 2)
    {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h")
        {
            std::cout << "search_zorder --data_path base.bin --query_path query.bin\n"
                         "  --attribute_file attributes.json --range_saveprefix ranges.json\n"
                         "  --groundtruth_saveprefix gt.json --index_file index.bin\n"
                         "  --result_saveprefix results/run_ --M 16 [--query_K 10]\n"
                         "  [--attr_num 2] [--method zorder|irange]\n"
                         "  Base rows must be Morton-sorted and aligned with the index and ground truth.\n";
            return 0;
        }
        if (i + 1 >= argc)
            throw Exception("missing value for " + arg);
        if (arg != "--data_path" && arg != "--query_path" &&
            arg != "--range_saveprefix" && arg != "--groundtruth_saveprefix" &&
            arg != "--index_file" && arg != "--result_saveprefix" &&
            arg != "--attribute_file" && arg != "--M" && arg != "--s" &&
            arg != "--query_K" && arg != "--attr_num" && arg != "--method")
            throw Exception("unknown option: " + arg);
        if (arg == "--attr_num")
            attr_num = std::stoi(argv[i + 1]);
        if (arg == "--method")
            method = argv[i + 1];
        if (arg == "--data_path")
            paths["data_vector"] = argv[i + 1];
        if (arg == "--query_path")
            paths["query_vector"] = argv[i + 1];
        if (arg == "--range_saveprefix")
            paths["range_prefix"] = argv[i + 1];
        if (arg == "--groundtruth_saveprefix")
            paths["groundtruth_prefix"] = argv[i + 1];
        if (arg == "--index_file")
            paths["index"] = argv[i + 1];
        if (arg == "--result_saveprefix")
            paths["result_saveprefix"] = argv[i + 1];

        if (arg == "--attribute_file")
            paths["attribute_file"] = argv[i + 1];
        if (arg == "--M")
        {
            M = std::stoi(argv[i + 1]);
        }
        if (arg == "--s")
            s = std::stod(argv[i + 1]);
        if (arg == "--query_K")
            query_K = std::stoi(argv[i + 1]);
    }

    if (attr_num <= 0 || query_K <= 0 || M <= 0)
        throw Exception("attr_num, query_K and M must be positive");
    if (method != "zorder" && method != "irange")
        throw Exception("method must be zorder or irange");
    for (const auto &key : {"data_vector", "query_vector", "range_prefix",
                            "groundtruth_prefix", "index", "result_saveprefix", "attribute_file"})
        if (paths[key].empty())
            throw Exception(std::string("missing input: ") + key);

    iPreGraph_multi::DataLoader storage;
    storage.query_K = query_K;
    storage.load_query_vectors(paths["query_vector"]);
    storage.load_base_vectors(paths["data_vector"]);

    // Attribute rows follow the same ordering as vectors and indexed point IDs.
    storage.load_attributes_json(paths["attribute_file"], attr_num);

    // Use the shared multiword Morton encoder for data and query bounds.
    storage.compute_morton_codes();

    // Each query supplies one inclusive interval per attribute; combine with AND.
    storage.load_numeric_predicates_json(paths["range_prefix"], attr_num);

    // Ground-truth IDs refer to the sorted base-vector rows.
    storage.load_ground_truth_json(paths["groundtruth_prefix"]);

    iPreGraph_multi::iPreGraph_Search_Zorder<float> index(paths["index"], &storage, M);
    std::cout << "Loaded index." << std::endl;
    // Evaluate the configured beam widths in this order.
    std::vector<int> SearchEF = {230, 220, 210, 200, 190, 180, 170, 160, 150, 140, 130, 120, 110, 100, 90, 80, 70, 60, 55, 50, 45, 40, 35, 30, 25, 20, 15, 14, 13, 12, 11, 10};
    if (method == "irange")
        index.benchmark_irange(SearchEF, paths["result_saveprefix"], M);
    else
        index.benchmark_pdg(SearchEF, paths["result_saveprefix"], M);

}
