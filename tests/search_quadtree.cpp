#include "quadtree_cli.h"
#include <chrono>
#include <iomanip>

using iPreGraph::QuadTreeIndex;

// Optional compatibility inputs are checked against the self-contained index;
// they must use the same row order as at construction (and as the ground truth).
static void verify_base(const std::string& path, const std::string& format, const QuadTreeIndex& index) {
    quad_cli::VectorReader reader(path, format);
    if (reader.count != index.size() || reader.dimension != index.dimension())
        throw std::runtime_error("--data_path does not match index dimensions");
    std::vector<float> row;
    for (uint32_t id = 0; id < reader.count; ++id) {
        reader.read_row(row);
        if (row != index.point_vector(id))
            throw std::runtime_error("--data_path does not match indexed vector at row " + std::to_string(id));
    }
}

static std::vector<std::vector<uint32_t>> ground_truth(const std::string& path, size_t k,
        const QuadTreeIndex& index, const std::vector<iPreGraph::QuadRect>& ranges) {
    auto values = quad_cli::json_numbers(path);
    if (values.size() / k != ranges.size() || values.size() % k)
        throw std::runtime_error("ground truth must contain query_count * query_K IDs, using -1 padding");
    std::vector<std::vector<uint32_t>> gt(ranges.size());
    for (size_t qi = 0; qi < ranges.size(); ++qi) {
        for (size_t j = 0; j < k; ++j) {
            double id = values[qi*k+j];
            if (id == -1) continue;
            if (id < 0 || id >= double(index.size()) || std::floor(id) != id)
                throw std::runtime_error("invalid ground truth point ID");
            if (!ranges[qi].contains(index.point_attributes(uint32_t(id))))
                throw std::runtime_error("ground truth violates attribute constraints; check input row order");
            gt[qi].push_back(uint32_t(id));
        }
        auto& ids = gt[qi]; std::sort(ids.begin(), ids.end());
        if (std::adjacent_find(ids.begin(), ids.end()) != ids.end())
            throw std::runtime_error("duplicate ground truth point ID");
    }
    return gt;
}

static std::vector<uint32_t> ef_values(const std::map<std::string, std::string>& args) {
    if (!args.count("--ef"))
        // return {140,130,120,110,100,90,80,70,60,55,50,45,40,35,33,30,25,23,20,15,10};
        return {70,60,55,50,45,40,35,33,30,25,23,20,15,10};
    std::vector<uint32_t> result;
    std::istringstream in(args.at("--ef")); std::string token;
    while (std::getline(in, token, ',')) {
        uint32_t ef = quad_cli::number({{"--ef", token}}, "--ef", 0);
        if (!ef) throw std::runtime_error("ef must be positive");
        result.push_back(ef);
    }
    if (result.empty() || args.at("--ef").back() == ',') throw std::runtime_error("invalid --ef list");
    return result;
}

int main(int argc, char** argv) {
    try {
        auto a = quad_cli::args(argc, argv);
        quad_cli::known(a, {"--help", "--index_file", "--query_path", "--range_saveprefix",
            "--groundtruth_saveprefix", "--result_saveprefix", "--data_path", "--attribute_file",
            "--M", "--query_K", "--ef", "--recall_stop", "--data_format", "--query_format",
            "--entry_points", "--edge_limit", "--seed"});
        if (a.count("--help")) {
            std::cout << "search_quadtree --index_file tree.qidx --query_path query.fvecs\n"
                "  --range_saveprefix ranges.json --groundtruth_saveprefix gt.json\n"
                "  --result_saveprefix results/result [--query_K 10 --ef 140,100,50,10]\n"
                "  [--recall_stop 0.9 --M 16 --data_path base.fvecs --attribute_file attr.json]\n"
                "  [--data_format auto|fvecs|bin --query_format auto|fvecs|bin]\n"
                "  [--entry_points 5 --edge_limit <index M> --seed 42]\n"
                "ANN samples distinct entries inside each query rectangle, then uses one global beam.\n"
                "Each expanded point restarts at root and follows children containing its (x,y) attributes.\n"
                "Neighbors come from edges[layer][point ID]; overlapping child MBRs use exact membership.\n"
                "ANN loop follows iRG_search.h::TopDown_nodeentries_search (bitmap, heaps, lowerBound).\n"
                "Cross-range DFS: two hops, with edge_limit / 2 additional candidates.\n"
                "JSON layouts match search_zorder: 2 attributes, AND constraints, original row IDs.\n"
                "Output: <prefix>mixed-quadtree.csv, no header, ef,recall,qps.\n"
                "M/data_path/attribute_file are optional consistency checks; the index contains the data.\n";
            return 0;
        }
        auto k = quad_cli::number(a, "--query_K", 10);
        if (!k) throw std::runtime_error("require query_K > 0");
        auto efs = ef_values(a);
        auto options = quad_cli::search_options(a);
        double recall_stop = 0.9;
        if (a.count("--recall_stop")) {
            size_t used = 0; const auto& token = a.at("--recall_stop");
            recall_stop = std::stod(token, &used);
            if (used != token.size() || !std::isfinite(recall_stop) || recall_stop < 0 || recall_stop > 1)
                throw std::runtime_error("recall_stop must be in [0,1]; 0 disables early stop");
        }
        const auto index_path = quad_cli::required(a, "--index_file");
        std::cout << "Loading quadtree index: " << index_path << '\n' << std::flush;
        auto index = QuadTreeIndex::load(index_path);
        // Resolve the default once so the reported budget is the one used by
        // every query. It is shared across all layers of each point's descent.
        if (!options.edge_limit) options.edge_limit = index.config().M;
        if (a.count("--M") && quad_cli::number(a, "--M", 0) != index.config().M)
            throw std::runtime_error("--M differs from indexed M");
        if (a.count("--data_path")) verify_base(a.at("--data_path"), quad_cli::format_arg(a, "--data_format"), index);
        if (a.count("--attribute_file")) {
            auto attrs = quad_cli::json_attributes(a.at("--attribute_file"), index.size());
            for (uint32_t id = 0; id < index.size(); ++id) {
                auto p = index.point_attributes(id);
                if (p.x != attrs[id].x || p.y != attrs[id].y)
                    throw std::runtime_error("--attribute_file differs from index at row " + std::to_string(id));
            }
        }
        auto queries = quad_cli::vectors(quad_cli::required(a, "--query_path"), quad_cli::format_arg(a, "--query_format"));
        if (queries[0].size() != index.dimension()) throw std::runtime_error("query dimension mismatch");
        auto ranges = quad_cli::json_ranges(quad_cli::required(a, "--range_saveprefix"), queries.size());
        auto gt = ground_truth(quad_cli::required(a, "--groundtruth_saveprefix"), k, index, ranges);
        auto output = quad_cli::required(a, "--result_saveprefix") + "mixed-quadtree.csv";
        auto parent = std::filesystem::path(output).parent_path();
        if (!parent.empty()) std::filesystem::create_directories(parent);
        std::ofstream out(output);
        if (!out) throw std::runtime_error("cannot create result CSV: " + output);
        out << std::setprecision(12);
        std::cout << "Loaded " << index.size() << " vectors, " << index.nodes().size() << " quadtree nodes, "
                  << index.level_count() << " layers (0=root).\n";
        std::cout << "Mode: TopDown_nodeentries_search, random in-range entries -> root-to-leaf quadtree edges.\n"
                      << "entry_points=" << options.entry_points << " edge_limit=" << options.edge_limit
                      << " seed=" << options.seed << " (query seed = seed + query ID, reused across EF values).\n"
                      << "Cross-range DFS: hop=2, cross_edge_limit=" << options.edge_limit / 2 << '\n';
        std::cout << "Results: " << output << '\n' << std::flush;
        for (uint32_t ef : efs) {
            uint64_t tp = 0;
            float searchtime = 0;
            std::cout << "Searching " << queries.size() << " queries, ef=" << ef << '\n' << std::flush;
            for (size_t qi = queries.size(); qi-- > 0;) {
                auto per_query = options; per_query.seed += static_cast<uint32_t>(qi);
                // Predicate preprocessing is excluded, as in search_zorder.
                const auto flags = index.prepare_flags(ranges[qi]);
                // auto begin = std::chrono::steady_clock::now();
                // Direct selection, DFS, and vector search are timed inside search().
                // Entry sampling and predicate preprocessing are excluded.
                auto hits = index.search(queries[qi], ranges[qi], flags, k, searchtime, ef, per_query);
                // seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
                for (auto hit : hits)
                    if (std::binary_search(gt[qi].begin(), gt[qi].end(), hit.second)) ++tp;
            }
            double recall = double(tp) / double(queries.size()) / double(k);
            double qps = double(queries.size()) / searchtime;
            out << ef << ',' << recall << ',' << qps << '\n';
            std::cout << "ef=" << ef << " recall=" << recall << " qps=" << qps << '\n';
            out.flush();
            if (!out) throw std::runtime_error("failed writing result CSV");
            std::cout << std::flush;
            if (recall < recall_stop) break;
        }
        out.flush(); if (!out) throw std::runtime_error("failed writing result CSV");
        return 0;
    } catch (const std::exception& e) { std::cerr << "error: " << e.what() << '\n'; return 1; }
}


//taskset -c 0 ./tests/search_quadtree   --data_path /guotengyue/dataset/gist/gist_base.bin   --query_path /guotengyue/dataset/gist/gist_query.bin   --range_saveprefix /guotengyue/dataset/gist/label/sel_4_1000000_random/qrangesel_10_4_1000000_random.json   --groundtruth_saveprefix /guotengyue/dataset/gist/label/sel_4_1000000_random/sif_gt_sel_10_4_1000000_random_10.json   --index_file /guotengyue/dataset/gist/quadtree_index_40.qidx   --result_saveprefix /guotengyue/gist/result_top10_hybrid/recall90_10_10/result   --attribute_file /guotengyue/dataset/gist/label/sel_4_1000000_random/attr_sel_4_1000000_random.json   --M 40   --query_K 10   --entry_points 1   --edge_limit 40   --seed 42
//taskset -c 0 ./tests/search_quadtree   --data_path /guotengyue/dataset/gist/gist_base.bin   --query_path /guotengyue/dataset/gist/gist_query.bin   --range_saveprefix /guotengyue/dataset/gist/label/sel_4_1000000_random/qrangesel_15_4_1000000_random.json   --groundtruth_saveprefix /guotengyue/dataset/gist/label/sel_4_1000000_random/sif_gt_sel_15_4_1000000_random_10.json   --index_file /guotengyue/dataset/gist/quadtree_index_40.qidx   --result_saveprefix /guotengyue/gist/result_top10_hybrid/recall90_10_15/result   --attribute_file /guotengyue/dataset/gist/label/sel_4_1000000_random/attr_sel_4_1000000_random.json   --M 40   --query_K 10   --entry_points 1   --edge_limit 40   --seed 42
//taskset -c 0 ./tests/search_quadtree   --data_path /guotengyue/dataset/gist/gist_base.bin   --query_path /guotengyue/dataset/gist/gist_query.bin   --range_saveprefix /guotengyue/dataset/gist/label/sel_4_1000000_random/qrangesel_19_4_1000000_random.json   --groundtruth_saveprefix /guotengyue/dataset/gist/label/sel_4_1000000_random/sif_gt_sel_19_4_1000000_random_10.json   --index_file /guotengyue/dataset/gist/quadtree_index_40.qidx   --result_saveprefix /guotengyue/gist/result_top10_hybrid/recall90_10_19/result   --attribute_file /guotengyue/dataset/gist/label/sel_4_1000000_random/attr_sel_4_1000000_random.json   --M 40   --query_K 10   --entry_points 1   --edge_limit 40   --seed 42
//taskset -c 0 ./tests/search_quadtree   --data_path /guotengyue/dataset/gist/gist_base.bin   --query_path /guotengyue/dataset/gist/gist_query.bin   --range_saveprefix /guotengyue/dataset/gist/label/sel_4_1000000_random/qrangesel_20_4_1000000_random.json   --groundtruth_saveprefix /guotengyue/dataset/gist/label/sel_4_1000000_random/sif_gt_sel_20_4_1000000_random_10.json   --index_file /guotengyue/dataset/gist/quadtree_index_40.qidx   --result_saveprefix /guotengyue/gist/result_top10_hybrid/recall90_10_20/result   --attribute_file /guotengyue/dataset/gist/label/sel_4_1000000_random/attr_sel_4_1000000_random.json   --M 40   --query_K 10   --entry_points 1   --edge_limit 40   --seed 42
//taskset -c 0 ./tests/search_quadtree   --data_path /guotengyue/dataset/gist/gist_base.bin   --query_path /guotengyue/dataset/gist/gist_query.bin   --range_saveprefix /guotengyue/dataset/gist/label/sel_4_1000000_random/qrangesel_21_4_1000000_random.json   --groundtruth_saveprefix /guotengyue/dataset/gist/label/sel_4_1000000_random/sif_gt_sel_21_4_1000000_random_10.json   --index_file /guotengyue/dataset/gist/quadtree_index_40.qidx   --result_saveprefix /guotengyue/gist/result_top10_hybrid/recall90_10_21/result   --attribute_file /guotengyue/dataset/gist/label/sel_4_1000000_random/attr_sel_4_1000000_random.json   --M 40   --query_K 10   --entry_points 1   --edge_limit 40   --seed 42