// Benchmark label OR predicates combined with numerical range AND predicates.
// Base rows must be ordered by (Morton code, label), matching the saved index.
// The driver uses three columns: label in column zero, then two numerical values.

#include "search_multi.h"
#include "search_zorder_label.h"

std::unordered_map<std::string, std::string> paths;

int query_K = 10;
int M;
double s;
int attr_num = 3;  
int label_col = 0; 

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

    iPreGraph_multi::ZLabelDataLoader storage;
    storage.query_K = query_K;
    storage.LoadQuery(paths["query_vector"]);

    std::cout << "Loaded data and query." << std::endl;
    
    // Load labels/numerical attributes and validate the required row ordering.
    storage.LoadAttrJson(paths["attribute_file"], 0, attr_num, label_col);

    std::cout << "Loaded attributes." << std::endl;
    
    // Labels are alternatives (OR); all numerical intervals must hold (AND).
    storage.LoadQueryRangeJson(paths["range_prefix"]);
    std::cout << "Loaded query ranges." << std::endl;   
    storage.query_K = query_K;

    // Ground-truth IDs refer to the sorted base-vector rows.
    storage.LoadGroundtruthJson(paths["groundtruth_prefix"]);

    iPreGraph_multi::iPreGraph_Search_Zorder_Label<float> index(paths["data_vector"], paths["index"], &storage, M);
    std::cout << "Loaded index." << std::endl;

    // Evaluate the configured beam widths in this order.
    std::vector<int> SearchEF = {20};

    index.search_zorder_label(SearchEF, paths["result_saveprefix"], M);

}
