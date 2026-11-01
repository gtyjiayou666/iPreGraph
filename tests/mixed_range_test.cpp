#include "search_mixed.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <stdexcept>

using Loader = iPreGraph_mixed::MixedDataLoader;
using Index = iPreGraph_mixed::iPreGraph_Search_Mixed<float>;

static void require(bool condition, const char *message)
{
    if (!condition)
        throw std::runtime_error(message);
}

static void set_edges(Index &index, int object, int depth,
                      std::initializer_list<unsigned> neighbors)
{
    auto *row = index.get_adjacency_list(object, depth);
    row[0] = static_cast<unsigned>(neighbors.size());
    std::copy(neighbors.begin(), neighbors.end(), row + 1);
}

int main(int argc, char **argv)
{
    try
    {
        const std::filesystem::path directory = argc > 1 ? argv[1] : "mixed-range-fixtures";
        std::filesystem::create_directories(directory);
        const auto vector_path = (directory / "vectors.bin").string();
        const auto graph_path = (directory / "graph.bin").string();
        constexpr int count = 8;
        constexpr int dimensions = 1;
        {
            std::ofstream file(vector_path, std::ios::binary);
            const float values[count] = {9, 3, 2, 1, 0, 5, 6, 7};
            file.write(reinterpret_cast<const char *>(&count), sizeof(count));
            file.write(reinterpret_cast<const char *>(&dimensions), sizeof(dimensions));
            file.write(reinterpret_cast<const char *>(values), sizeof(values));
            require(file.good(), "Cannot write test vectors");
        }
        {
            std::ofstream file(graph_path, std::ios::binary);
            const unsigned zero = 0;
            for (int object = 0; object < count; ++object)
                for (int depth = 0; depth < 4; ++depth)
                    file.write(reinterpret_cast<const char *>(&zero), sizeof(zero));
            require(file.good(), "Cannot write test graph");
        }

        Loader data;
        data.data_nb = count;
        data.Dim = dimensions;
        data.query_nb = 1;
        data.query_K = 4;
        for (int object = 0; object < count; ++object)
        {
            data.labels.push_back(object < 3 ? 0 : 1);
            data.z_attrs.push_back({object, object});
            data.z_codes.push_back(iPreGraph_multi::encode_morton_key(data.z_attrs.back()));
        }
        Index index(vector_path, graph_path, &data, 4);
        set_edges(index, 1, 1, {2, 3});
        set_edges(index, 3, 0, {1, 2, 4});

        std::vector<int> left = {1, 3}, right = {2, 4};
        std::vector<bool> flags(count, false);
        for (int object = 1; object <= 4; ++object)
            flags[object] = true;
        iPreGraph_multi::MBR box(2);
        box.min_vals = {1, 1};
        box.max_vals = {4, 4};
        iPreGraph::Bitmap labels;
        labels.set(0);
        labels.set(1);
        std::vector<uint8_t> cache(index.tree->node_count, 0);

        searcher::Bitset<uint64_t> visited(count);
        visited.set(1);
        const auto first = index.select_direct_edges(1, 1, 2, 4, visited,
                                                     labels, box, cache, flags);
        require(first == std::vector<unsigned>{2},
                "Direct selection must retain the right endpoint and exclude other covers");

        searcher::Bitset<uint64_t> second_visited(count);
        second_visited.set(3);
        const auto second = index.select_direct_edges(3, 3, 4, 4, second_visited,
                                                      labels, box, cache, flags);
        require(second == std::vector<unsigned>{4},
                "Direct selection must exclude neighbors below the left endpoint");

        std::vector<unsigned> cross;
        index.select_cross_range_edges(left, right, 1, 2, 2, cross,
                                       labels, box, cache, flags, false);
        require(std::find(cross.begin(), cross.end(), 3) != cross.end(),
                "Cross-range selection must still reach the second cover");

        // Singleton seed segments make the entry selection deterministic.
        auto seeds = index.tree->find_covering_nodes(index.tree->root, 1, 2);
        const float query = 0;
        auto results = index.search_pdg(seeds, &query, left, right, 8, 4, 4, 0,
                                        labels, box, cache, flags);
        std::set<int> ids;
        while (!results.empty())
        {
            ids.insert(results.top().second);
            results.pop();
        }
        require(ids == std::set<int>({1, 2, 3, 4}),
                "Search must expand cross-range candidates using their own cover");
        std::cout << "PASS: direct range bounds, cross-range selection, and mixed search\n";
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
