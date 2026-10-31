// Benchmark categorical label OR predicates supplied as JSON label lists.
// Base vectors and attributes must be aligned and ordered by the label column.
// See README.md for the binary vector layout and common index options.

#include "search_label.h"

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
        if (arg == "--label_json")
            paths["label_json"] = argv[i + 1];
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
                         " --attr_json <json> --label_json <json> --groundtruth_json <json>"
                         " --result_saveprefix <prefix> --M <int> --query_K <int>"
                         " [--attr_num 3] [--label_col 0]"
                      << std::endl;
            return 0;
        }
    }

    if (paths["data_vector"].empty() || paths["query_vector"].empty() || paths["index"].empty() ||
        paths["attr_json"].empty() || paths["label_json"].empty() || paths["groundtruth_json"].empty() ||
        paths["result_saveprefix"].empty() || M <= 0 || query_K <= 0)
        throw Exception("please check input parameters (see --help)");

    if (attr_num <= 0 || label_col < 0 || label_col >= attr_num)
        throw Exception("--label_col must satisfy 0 <= label_col < --attr_num");

    iPreGraph::DataLoaderOr storage(paths["attr_json"], attr_num, label_col);
    storage.query_K = query_K;
    storage.LoadQuery(paths["query_vector"]);

    iPreGraph::iPreGraph_Search_Label<float> index(paths["data_vector"], paths["index"], &storage, M);

    // Evaluate the configured beam widths in this order.
    std::vector<int> SearchEF = {70, 65, 60, 55, 50, 45, 40, 35, 30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10};

    index.label_or(SearchEF, paths["result_saveprefix"], M,
       paths["label_json"], paths["groundtruth_json"]);

}
