// core_split.h
//
// Multi-core helpers shared by the multi_tile stage hosts.
//
// Every stage is independent across tile-rows (32 vectors each), so the Mt
// tile-rows are cut into runs of ceil(Mt / cores), one run per Tensix core,
// taking cores row by row from (0,0). Every kernel gets its run as runtime
// args: the run length in place of Mt and, for readers and writers, the run's
// first tile-row (mt_start) as the last arg.

#pragma once

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/distributed.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <vector>

namespace turboquant::multi_tile {

struct CoreRun {
    tt::tt_metal::CoreCoord core;
    uint32_t mt_start;
    uint32_t mt_count;
};

struct CoreSplit {
    tt::tt_metal::CoreRangeSet cores;  // the cores that got a run
    std::vector<CoreRun> runs;
};

// Uses at most min(num_cores, grid size) cores, and fewer when Mt is small.
inline CoreSplit split_tile_rows(uint32_t Mt, uint32_t num_cores, tt::tt_metal::CoreCoord grid) {
    using tt::tt_metal::CoreCoord;
    const uint32_t gx = grid.x;
    const uint32_t n = std::clamp<uint32_t>(num_cores, 1, gx * grid.y);
    const uint32_t per = (Mt + n - 1) / n;

    CoreSplit s;
    for (uint32_t mt = 0; mt < Mt; mt += per) {
        const uint32_t i = s.runs.size();
        s.runs.push_back({CoreCoord{i % gx, i / gx}, mt, std::min(per, Mt - mt)});
    }
    // The cores used: the first full rows, then the start of the next row.
    const uint32_t rows = s.runs.size() / gx, rest = s.runs.size() % gx;
    std::vector<tt::tt_metal::CoreRange> ranges;
    if (rows > 0) ranges.emplace_back(CoreCoord{0, 0}, CoreCoord{gx - 1, rows - 1});
    if (rest > 0) ranges.emplace_back(CoreCoord{0, rows}, CoreCoord{rest - 1, rows});
    s.cores = tt::tt_metal::CoreRangeSet(std::move(ranges));
    return s;
}

// Runs the workload once (this run also loads the kernels), then `iters` more
// times, and prints the first time and the best of the warm times.
inline void run_timed(tt::tt_metal::distributed::MeshCommandQueue& cq,
                      tt::tt_metal::distributed::MeshWorkload& wl,
                      const char* label, uint32_t iters) {
    double best_ms = 0.0;
    for (uint32_t it = 0; it <= iters; ++it) {
        auto t0 = std::chrono::steady_clock::now();
        tt::tt_metal::distributed::EnqueueMeshWorkload(cq, wl, /*blocking=*/true);
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (it == 0) std::cout << "  " << label << ": " << ms << " ms\n";
        else if (it == 1 || ms < best_ms) best_ms = ms;
    }
    if (iters > 0)
        std::cout << "  " << label << ", best of " << iters << " warm: " << best_ms << " ms\n";
}

}  // namespace turboquant::multi_tile
