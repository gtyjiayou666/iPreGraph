#include "quadtree_cli.h"
#include <chrono>

int main(int argc, char **argv)
{
    try
    {
        auto a = quad_cli::args(argc, argv);
        quad_cli::known(a, {"--help", "--data_path", "--data_format", "--attr_path", "--attribute_file", "--index_file", "--M",
                            "--ef_construction", "--leaf_size", "--max_depth", "--threads", "--progress"});
        if (a.count("--help"))
        {
            std::cout << "build_quadtree_index --data_path base.fvecs --attribute_file attributes.json --index_file tree.qidx\n"
                         "  attributes.json: [x0,y0,x1,y1,...], same row order as base.fvecs.\n"
                         "  Balanced by point count; leaf RNG graphs are merged bottom-up in parallel.\n"
                         "  Each root/internal/leaf node and its graph are saved in the index.\n"
                         "  [--M 16 --ef_construction 100 --leaf_size 32 --max_depth 32 --threads 1]\n"
                         "  [--progress 1]; set 0 to disable construction progress.\n"
                         "  [--data_format auto|fvecs|bin]; auto uses fvecs for .fvecs, otherwise legacy bin.\n"
                         "  CSV alternative: --attr_path xy.csv (no header, x,y per vector).\n";
            return 0;
        }
        const auto progress_flag = quad_cli::number(a, "--progress", 1);
        if (progress_flag > 1) throw std::runtime_error("--progress must be 0 or 1");
        std::ostream* progress = progress_flag ? &std::cout : nullptr;
        const auto total_start = std::chrono::steady_clock::now();
        const auto data_path = quad_cli::required(a, "--data_path");
        if (progress) *progress << "[quadtree] loading vectors: " << data_path << '\n' << std::flush;
        auto v = quad_cli::vectors(data_path, quad_cli::format_arg(a, "--data_format"));
        if (progress) *progress << "[quadtree] loaded " << v.size() << " vectors, dim=" << v.front().size()
            << "; loading attributes\n" << std::flush;
        if (a.count("--attr_path") + a.count("--attribute_file") != 1)
            throw std::runtime_error("specify exactly one of --attr_path or --attribute_file");
        std::vector<iPreGraph::QuadPoint> xy;
        if (a.count("--attribute_file"))
            xy = quad_cli::json_attributes(a.at("--attribute_file"), v.size());
        else
            for (auto r : quad_cli::rows<2>(a.at("--attr_path")))
                xy.push_back({r[0], r[1]});
        iPreGraph::QuadConfig c;
        c.M = quad_cli::number(a, "--M", c.M);
        c.ef_construction = quad_cli::number(a, "--ef_construction", c.ef_construction);
        c.leaf_size = quad_cli::number(a, "--leaf_size", c.leaf_size);
        c.max_depth = quad_cli::number(a, "--max_depth", c.max_depth);
        c.threads = quad_cli::number(a, "--threads", c.threads);
        const std::string path = quad_cli::required(a, "--index_file");
        auto start = std::chrono::steady_clock::now();
        iPreGraph::QuadTreeIndex index;
        index.build(std::move(v), std::move(xy), c, progress);
        if (progress) *progress << "[quadtree] graph construction finished, "
            << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count()
            << " s; saving index: " << path << '\n' << std::flush;
        index.save(path, progress);
        std::cout << "saved " << index.size() << " points, " << index.nodes().size() << " nodes, "
                  << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() << " seconds\n";
        if (progress) *progress << "[quadtree] complete: " << std::filesystem::file_size(path)
            << " bytes, total elapsed "
            << std::chrono::duration<double>(std::chrono::steady_clock::now() - total_start).count()
            << " s (including input loading)\n" << std::flush;
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
}
