// Standalone benchmark for label OR and numerical range AND predicates.
// Base rows must be ordered by (label, Morton code). A query contains a label
// list followed by one [low, high] pair per numerical attribute.
// This source is not currently registered as a CMake executable target.

#include "search_mixed.h"

#include <unordered_map>
#include <sstream>
#include <iostream>
#include <string>
#include <vector>

std::unordered_map<std::string, std::string> paths;

int query_K;
int M;
int attr_num = 3;  
int label_col = 0; 

std::vector<int> ParseEfList(const std::string &s)
{
    
    std::vector<int> out;
    std::string t = s;
    for (char &c : t)
        if (c == ',')
            c = ' ';
    std::istringstream iss(t);
    int v;
    while (iss >> v)
        out.push_back(v);
    return out;
}

int main(int argc, char **argv)
{
    for (int i = 0; i < argc; i++)
    {
        std::string arg = argv[i];
        if (arg == "--data_path")
            paths["data_vector"] = argv[i + 1];
        if (arg == "--query_path")
            paths["query_vector"] = argv[i + 1];
        if (arg == "--index_file")
            paths["index"] = argv[i + 1];
        if (arg == "--attr_json")
            paths["attr_json"] = argv[i + 1];
        if (arg == "--query_json")
            paths["query_json"] = argv[i + 1];
        if (arg == "--groundtruth_json")
            paths["groundtruth_json"] = argv[i + 1];
        if (arg == "--result_saveprefix")
            paths["result_saveprefix"] = argv[i + 1];
        if (arg == "--M")
            M = std::stoi(argv[i + 1]);
        if (arg == "--query_K")
            query_K = std::stoi(argv[i + 1]);
        if (arg == "--attr_num")
            attr_num = std::stoi(argv[i + 1]);
        if (arg == "--label_col")
            label_col = std::stoi(argv[i + 1]);
        if (arg == "--help" || arg == "-h")
        {
            std::cout << argv[0]
                      << " --data_path <bin> --query_path <bin> --index_file <bin>"
                         " --attr_json <json> --query_json <json> [--groundtruth_json <json>]"
                         " --result_saveprefix <prefix> --M <int> --query_K <int>"
                         " [--attr_num 3] [--label_col 0]"
                      << std::endl;
            return 0;
        }
    }

    if (paths["data_vector"].empty() || paths["query_vector"].empty() || paths["index"].empty() ||
        paths["attr_json"].empty() || paths["query_json"].empty() ||
        paths["result_saveprefix"].empty() || M <= 0 || query_K <= 0)
        throw Exception("please check input parameters (see --help)");

    if (attr_num <= 0 || label_col < 0 || label_col >= attr_num)
        throw Exception("--label_col must satisfy 0 <= label_col < --attr_num");

    iPreGraph_mixed::MixedDataLoader storage;
    // Load labels/numerical attributes and validate the required row ordering.
    storage.LoadAttrJson(paths["attr_json"], 0, attr_num, label_col);

    // Labels are alternatives (OR); all numerical intervals must hold (AND).
    storage.LoadQueryRangeJson(paths["query_json"]);
    storage.query_K = query_K;

    storage.LoadQuery(paths["query_vector"]);
    if ((int)storage.query_labels.size() != storage.query_nb)
        throw Exception("query_json query count != query vector count");

    std::vector<std::vector<int>> gt;
    if (!paths["groundtruth_json"].empty())
        gt = iPreGraph_mixed::MixedDataLoader::LoadGtJson(
            paths["groundtruth_json"], storage.query_nb, query_K);

    iPreGraph_mixed::iPreGraph_Search_Mixed<float> index(
        paths["data_vector"], paths["index"], &storage, M);

    // Evaluate the configured beam widths in this order.
    std::vector<int> SearchEF = {40, 35, 30, 25, 20, 18, 16, 14, 12, 10};

    index.search_mixed(SearchEF, paths["result_saveprefix"], M, gt);
    
}
