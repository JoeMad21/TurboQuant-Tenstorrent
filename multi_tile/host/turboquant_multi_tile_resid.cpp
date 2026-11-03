// turboquant_multi_tile_resid.cpp
//
// Standalone Stage 2b1 (residual) host: reads dump_input.bin and
// dump_xhat_tile.bin, computes r = x - x_hat in tile form, writes
// dump_r_tile.bin.

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/tilize_utils.hpp>

#include <cstdint>
#include <cstring>
#include <vector>
#include <fstream>
#include <iostream>
#include <chrono>

#include "tile_layout.h"
#include "core_split.h"
#include "graceful_exit.h"

namespace tq = turboquant::multi_tile;
using namespace tt::tt_metal;
using namespace tt::tt_metal::distributed;

static std::vector<uint16_t> load_flat_as_tiles(
    const std::string& path, uint32_t n_rows, uint32_t n_cols)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); std::abort(); }
    std::vector<uint16_t> raw(n_rows * n_cols);
    f.read(reinterpret_cast<char*>(raw.data()), n_rows * n_cols * 2);
    std::vector<bfloat16> as_bf(raw.size());
    static_assert(sizeof(bfloat16) == sizeof(uint16_t));
    std::memcpy(as_bf.data(), raw.data(), raw.size() * 2);
    std::vector<bfloat16> tiled = tilize_nfaces(as_bf, n_rows, n_cols);
    std::vector<uint16_t> out(tiled.size());
    std::memcpy(out.data(), tiled.data(), tiled.size() * 2);
    return out;
}

static std::shared_ptr<MeshBuffer> make_mesh_buf(
    MeshDevice* dev, uint32_t total_bytes, uint32_t page_bytes)
{
    ReplicatedBufferConfig rep{ .size = total_bytes };
    DeviceLocalBufferConfig local{ .page_size = page_bytes, .buffer_type = BufferType::DRAM };
    return MeshBuffer::create(rep, local, dev);
}

static int run(int argc, char** argv) {
    uint32_t num_vectors = 32, num_cores = 1, iters = 0;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--num-vectors" && i + 1 < argc) {
            num_vectors = std::atoi(argv[++i]);
        } else if (std::string(argv[i]) == "--cores" && i + 1 < argc) {
            num_cores = std::atoi(argv[++i]);
        } else if (std::string(argv[i]) == "--iters" && i + 1 < argc) {
            iters = std::atoi(argv[++i]);
        }
    }
    constexpr uint32_t d = tq::D;
    if (num_vectors % tq::TILE_HEIGHT != 0) {
        std::fprintf(stderr, "num_vectors must be multiple of 32\n"); return 1;
    }
    const uint32_t Mt = num_vectors / tq::TILE_HEIGHT;
    const uint32_t Dt = d / tq::TILE_WIDTH;

    std::cout << "[multi_tile Stage 2b1] N=" << num_vectors
              << " Mt=" << Mt << " Dt=" << Dt << "\n";

    auto x_tiles    = load_flat_as_tiles("dump_input.bin",     num_vectors, d);
    auto xhat_tiles = load_flat_as_tiles("dump_xhat_tile.bin", num_vectors, d);
    std::cout << "  Loaded x and xhat tiles.\n";

    auto mesh_dev = MeshDevice::create_unit_mesh(0);
    auto& cq = mesh_dev->mesh_command_queue();
    MeshCoordinateRange dev_range(MeshCoordinate{0, 0}, MeshCoordinate{0, 0});

    const uint32_t tile_bytes = tq::TILE_BYTES_BF16;
    auto x_buf    = make_mesh_buf(mesh_dev.get(), Mt * Dt * tile_bytes, tile_bytes);
    auto xhat_buf = make_mesh_buf(mesh_dev.get(), Mt * Dt * tile_bytes, tile_bytes);
    auto r_buf    = make_mesh_buf(mesh_dev.get(), Mt * Dt * tile_bytes, tile_bytes);

    EnqueueWriteMeshBuffer(cq, x_buf,    x_tiles,    true);
    EnqueueWriteMeshBuffer(cq, xhat_buf, xhat_tiles, true);

    Program prog = CreateProgram();
    const auto split = tq::split_tile_rows(Mt, num_cores, mesh_dev->compute_with_storage_grid_size());
    const auto& core_range = split.cores;
    std::cout << "  Cores: " << split.runs.size() << "\n";

    auto make_cb = [&](tt::CB id, uint32_t n_pages) {
        std::map<uint8_t, tt::DataFormat> spec = {
            {static_cast<uint8_t>(id), tt::DataFormat::Float16_b}
        };
        CircularBufferConfig cfg(n_pages * tile_bytes, spec);
        cfg.set_page_size(static_cast<uint8_t>(id), tile_bytes);
        CreateCircularBuffer(prog, core_range, cfg);
    };
    make_cb(tt::CB::c_in0,  2);
    make_cb(tt::CB::c_in1,  2);
    make_cb(tt::CB::c_out0, 2);

    auto reader_id = CreateKernel(prog,
        "multi_tile/kernels/residual_reader.cpp", core_range,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc       = NOC::RISCV_0_default,
        });
    auto writer_id = CreateKernel(prog,
        "multi_tile/kernels/residual_writer.cpp", core_range,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc       = NOC::RISCV_1_default,
        });
    std::vector<uint32_t> compute_args = {Dt};
    auto compute_id = CreateKernel(prog,
        "multi_tile/kernels/residual_compute.cpp", core_range,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .compile_args  = compute_args,
        });

    for (const auto& r : split.runs) {
        SetRuntimeArgs(prog, reader_id, r.core,
            {static_cast<uint32_t>(x_buf->address()),
             static_cast<uint32_t>(xhat_buf->address()),
             r.mt_count, Dt, r.mt_start});
        SetRuntimeArgs(prog, writer_id, r.core,
            {static_cast<uint32_t>(r_buf->address()), r.mt_count, Dt, r.mt_start});
        SetRuntimeArgs(prog, compute_id, r.core, {r.mt_count});
    }

    MeshWorkload wl;
    wl.add_program(dev_range, std::move(prog));

    std::cout << "  Dispatching workload...\n";
    tq::run_timed(cq, wl, "Dispatch+execute time", iters);

    std::vector<uint16_t> r_tiles(Mt * Dt * tq::TILE_ELEMENTS);
    EnqueueReadMeshBuffer(cq, r_tiles, r_buf, true);

    std::vector<bfloat16> as_bf(r_tiles.size());
    std::memcpy(as_bf.data(), r_tiles.data(), r_tiles.size() * 2);
    std::vector<bfloat16> flat = untilize_nfaces(as_bf, num_vectors, d);
    std::vector<uint16_t> out(flat.size());
    std::memcpy(out.data(), flat.data(), flat.size() * 2);

    std::ofstream of("dump_r_tile.bin", std::ios::binary);
    of.write(reinterpret_cast<const char*>(out.data()), out.size() * 2);
    std::cout << "  Wrote " << (out.size() * 2)
              << " bytes -> dump_r_tile.bin (flat row-major)\n";
    return 0;
}

int main(int argc, char** argv) { return turboquant::run_guarded(run, argc, argv); }
