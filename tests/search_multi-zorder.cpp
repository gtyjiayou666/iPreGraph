// Benchmark numerical AND predicates on a legacy Z-order graph index.
// Vectors, attributes, and ground-truth IDs must share the same Morton-sorted order.
// This driver loads exactly two numerical attributes per base vector.

#include "iRG_search_multi.h"
#include "iRG_search_zorder.h"

std::unordered_map<std::string, std::string> paths;

int query_K = 10;
int M;
double s;

int main(int argc, char **argv)
{
    for (int i = 0; i < argc; i++)
    {
        std::string arg = argv[i];
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

    iPreGraph_multi::DataLoader storage;
    storage.query_K = query_K;
    storage.LoadQuery(paths["query_vector"]);
    storage.LoadData(paths["data_vector"]);

    // Attribute rows follow the same ordering as vectors and indexed point IDs.
    storage.LoadMultiAttributesJson(paths["attribute_file"], 2);

    // Use the shared multiword Morton encoder for data and query bounds.
    storage.CalMultiZorder();

    // Each query supplies one inclusive interval per attribute; combine with AND.
    storage.LoadMultiRangesJson(paths["range_prefix"], 2);

    // Ground-truth IDs refer to the sorted base-vector rows.
    storage.LoadGroundtruthJson(paths["groundtruth_prefix"]);

    iPreGraph_multi::iPreGraph_Search_Zorder<float> index(paths["index"], &storage, M);
    std::cout << "Loaded index." << std::endl;
    // Evaluate the configured beam widths in this order.
    std::vector<int> SearchEF = {230, 220, 210, 200, 190, 180, 170, 160, 150, 140, 130, 120, 110, 100, 90, 80, 70, 60, 55, 50, 45, 40, 35, 30, 25, 20, 15, 14, 13, 12, 11, 10};
    index.search_zorder(SearchEF, paths["result_saveprefix"], M);

}
