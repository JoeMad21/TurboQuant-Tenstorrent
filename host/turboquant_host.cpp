// host/turboquant_host.cpp
//
// Host-side TurboQuant driver for Tenstorrent Wormhole.
// Uses the TT-Metalium Mesh API (MeshDevice / MeshBuffer / MeshWorkload).

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <map>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <optional>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/system_mesh.hpp>

#include "turboquant_layout.h"

#include <tt-metalium/hal.hpp>
using namespace tt::tt_metal;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

static constexpr uint32_t kD = turboquant::kDim;
static constexpr uint32_t kB = turboquant::kBits;
static constexpr uint32_t kK = turboquant::kQjlDim;
static constexpr uint32_t kN = turboquant::kTileVectors;

static constexpr uint32_t kQjlCoresX = 8u;
static constexpr uint32_t kQjlCoresY = 4u;
static constexpr uint32_t kQjlMaxVecsPerDevice = kQjlCoresX * kQjlCoresY;

// ---------------------------------------------------------------------------
// BF16 conversion
// bfloat16 is in the global namespace (not tt:: or tt::tt_metal::)

// ---------------------------------------------------------------------------

static uint16_t float_to_bf16(float v) {
    // BF16 = upper 16 bits of float32
    uint32_t bits; memcpy(&bits, &v, 4); return static_cast<uint16_t>(bits >> 16);
}

static float bf16_to_float(uint16_t u) {
    // Expand BF16 to float32 by zero-extending lower 16 bits
    uint32_t bits = static_cast<uint32_t>(u) << 16; float f; memcpy(&f, &bits, 4); return f;
}

// ---------------------------------------------------------------------------
// Test vector generation
// ---------------------------------------------------------------------------

static std::vector<std::vector<float>> gen_test_vectors(
    uint32_t n, uint32_t d, uint32_t seed = 42)
{
    uint64_t state = seed;
    auto rnd = [&]() -> float {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        uint32_t bits = static_cast<uint32_t>(state >> 33);
        bits = (bits & 0x7FFFFFu) | 0x3F800000u;
        float f; memcpy(&f, &bits, 4); return f - 1.5f;
    };
    std::vector<std::vector<float>> vs(n, std::vector<float>(d));
    for (uint32_t i = 0; i < n; ++i) {
        float nm = 0.0f;
        for (uint32_t j = 0; j < d; ++j) { vs[i][j] = rnd(); nm += vs[i][j] * vs[i][j]; }
        nm = std::sqrt(nm);
        for (uint32_t j = 0; j < d; ++j) vs[i][j] /= nm;
    }
    return vs;
}

// ---------------------------------------------------------------------------
// Binary file I/O
// ---------------------------------------------------------------------------

static void write_bin(const std::string& path, const void* data, size_t bytes) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open for writing: " + path);
    f.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(bytes));
    std::cout << "  wrote " << bytes << " bytes -> " << path << "\n";
}

static std::vector<uint8_t> read_bin(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("Cannot open: " + path);
    size_t sz = static_cast<size_t>(f.tellg()); f.seekg(0);
    std::vector<uint8_t> buf(sz);
    f.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(sz));
    return buf;
}

// ---------------------------------------------------------------------------
// MeshBuffer helper
// ---------------------------------------------------------------------------

static std::shared_ptr<distributed::MeshBuffer> make_mesh_buf(
    distributed::MeshDevice* dev,
    uint32_t total_bytes,
    uint32_t page_bytes)
{
    distributed::ReplicatedBufferConfig rep_cfg{ .size = total_bytes };
    distributed::DeviceLocalBufferConfig local_cfg{
        .page_size   = page_bytes,
        .buffer_type = BufferType::DRAM,
    };
    return distributed::MeshBuffer::create(rep_cfg, local_cfg, dev);
}

// ---------------------------------------------------------------------------
// Device-chunk planning
// ---------------------------------------------------------------------------

struct DeviceChunk {
    distributed::MeshCoordinate coord;
    uint32_t vec_offset;
    uint32_t vec_count;
};

static std::vector<DeviceChunk> plan_device_chunks(
    distributed::MeshDevice* mesh_dev, uint32_t num_vectors)
{
    std::vector<distributed::MeshCoordinate> coords;
    for (const auto& c : distributed::MeshCoordinateRange(mesh_dev->shape())) {
        coords.push_back(c);
    }

    const uint32_t min_split_vecs = 2 * kN;
    if (coords.size() < 2 || num_vectors < min_split_vecs) {
        distributed::MeshCoordinate c0 = coords.empty() ? distributed::MeshCoordinate(0, 0) : coords[0];
        return { DeviceChunk{ c0, 0u, num_vectors } };
    }

    uint32_t half0 = (num_vectors / 2 / kN) * kN;
    if (half0 == 0) half0 = kN;
    const uint32_t half1 = num_vectors - half0;

    return {
        DeviceChunk{ coords[0], 0u,     half0 },
        DeviceChunk{ coords[1], half0,  half1 },
    };
}

static std::pair<size_t, size_t> chunk_byte_range(const DeviceChunk& chunk, uint32_t page_bytes) {
    const uint32_t tile_offset = chunk.vec_offset / kN;
    const uint32_t num_tiles   = (chunk.vec_count + kN - 1) / kN;
    return { static_cast<size_t>(tile_offset) * page_bytes,
             static_cast<size_t>(num_tiles) * page_bytes };
}

static void write_chunks_to_shards(
    distributed::MeshCommandQueue&                  cq,
    const std::shared_ptr<distributed::MeshBuffer>& buf,
    const std::vector<DeviceChunk>&                 chunks,
    const std::vector<uint16_t>&                    full_data_u16,
    uint32_t                                        page_bytes)
{
    for (const auto& chunk : chunks) {
        auto [byte_off, byte_len] = chunk_byte_range(chunk, page_bytes);
        const size_t elem_off = byte_off / 2;
        const size_t elem_len = byte_len / 2;
        std::vector<uint16_t> shard_data(
            full_data_u16.begin() + elem_off,
            full_data_u16.begin() + elem_off + elem_len);

        distributed::ShardDataTransfer transfer{chunk.coord};
        transfer.host_data(shard_data.data())
                .region(BufferRegion(0, static_cast<DeviceAddr>(shard_data.size() * sizeof(uint16_t))));
        cq.enqueue_write_shards(buf, {transfer}, /*blocking=*/true);
    }
}

static std::vector<uint8_t> concat_chunks(const std::vector<std::vector<uint8_t>>& parts) {
    size_t total = 0;
    for (auto& p : parts) total += p.size();
    std::vector<uint8_t> out;
    out.reserve(total);
    for (auto& p : parts) out.insert(out.end(), p.begin(), p.end());
    return out;
}

// ---------------------------------------------------------------------------
// Isolated stage runner
// ---------------------------------------------------------------------------

static std::vector<std::vector<uint8_t>> run_isolated_stage(
    distributed::MeshDevice*                 mesh_dev,
    distributed::MeshCommandQueue&           cq,
    const std::vector<DeviceChunk>&          chunks,
    uint64_t                                 src_addr,   // DeviceAddr is uint64
    int                                      last_stage,
    tt::CB                                   dump_cb,
    uint32_t                                 dump_page_bytes,
    uint64_t                                 cent_addr = 0)
{
    distributed::MeshWorkload workload;
    std::vector<std::shared_ptr<distributed::MeshBuffer>> dump_bufs(chunks.size());

    for (size_t ci = 0; ci < chunks.size(); ++ci) {
        const DeviceChunk& chunk  = chunks[ci];
        const uint32_t num_tiles  = (chunk.vec_count + kN - 1) / kN;
        const uint32_t bytes_fp16 = kN * kD * 2;
        const uint32_t bytes_pq   = kN * turboquant::kPolarQuantBytes;
        const uint32_t bytes_qjl  = kN * (turboquant::kQjlBytes + turboquant::kRNormBytes);

        dump_bufs[ci] = make_mesh_buf(mesh_dev,
                                      num_tiles * dump_page_bytes, dump_page_bytes);

        Program prog = CreateProgram();
        tt::tt_metal::CoreCoord core{0, 0};
        tt::tt_metal::CoreRange core_range(core, core);

        auto make_cb = [&](tt::CB id, uint32_t page, tt::DataFormat fmt, uint32_t depth_pages = 2u) {
            std::map<uint8_t, tt::DataFormat> data_format_spec = {{static_cast<uint8_t>(id), fmt}};
            CircularBufferConfig cfg(depth_pages * page, data_format_spec);
            cfg.set_page_size(static_cast<uint8_t>(id), page);
            CreateCircularBuffer(prog, core_range, cfg);
        };

        make_cb(tt::CB::c_in0,  bytes_fp16, tt::DataFormat::Float16_b);
        make_cb(tt::CB::c_in1,  bytes_fp16, tt::DataFormat::Float16_b);
        
        const uint32_t c_in23_depth = (last_stage == 1) ? num_tiles : 2u;
        make_cb(tt::CB::c_in2,  bytes_pq,   tt::DataFormat::RawUInt8,  c_in23_depth);
        make_cb(tt::CB::c_in3,  bytes_fp16, tt::DataFormat::Float16_b, c_in23_depth);
        make_cb(tt::CB::c_out0, bytes_qjl,  tt::DataFormat::RawUInt8);

        std::map<std::string, std::string> kdefines = {
            {"TQ_DIM",           std::to_string(kD)},
            {"TQ_ROTATION_SEED", std::to_string(turboquant::kRotationSeed)},
            {"TQ_BITS",          std::to_string(kB)},
            {"TQ_K_VECS",        std::to_string(kN)},
            {"TQ_QJL_DIM",       std::to_string(kK)},
            {"TQ_QJL_SEED",      std::to_string(turboquant::kQjlSeed)},
        };

        if (last_stage == 0) {
            auto rk = CreateKernel(prog,
                "kernels/rotation_kernel.cpp", core_range,
                DataMovementConfig{
                    .processor = DataMovementProcessor::RISCV_0,
                    .noc       = NOC::RISCV_0_default,
                    .defines   = kdefines,
                });
            SetRuntimeArgs(prog, rk, core,
                           {static_cast<uint32_t>(src_addr), chunk.vec_count, kD, kN});
        } else if (last_stage == 1) {
            auto rk = CreateKernel(prog,
                "kernels/polarquant_kernel.cpp", core_range,
                DataMovementConfig{
                    .processor = DataMovementProcessor::RISCV_0,
                    .noc       = NOC::RISCV_0_default,
                    .defines   = kdefines,
                });
            SetRuntimeArgs(prog, rk, core,
                           {static_cast<uint32_t>(src_addr), chunk.vec_count, kD, kN});
        } else if (last_stage == 2) {
            if (chunk.vec_count > kQjlMaxVecsPerDevice) {
                throw std::runtime_error("run_isolated_stage: chunk vec_count exceeds capacity");
            }
            tt::tt_metal::CoreRange multi_core_range(
                tt::tt_metal::CoreCoord{0, 0},
                tt::tt_metal::CoreCoord{kQjlCoresX - 1, kQjlCoresY - 1});
            auto rk = CreateKernel(prog,
                "kernels/qjl_kernel.cpp", multi_core_range,
                DataMovementConfig{
                    .processor = DataMovementProcessor::RISCV_0,
                    .noc       = NOC::RISCV_0_default,
                    .defines   = kdefines,
                });
            for (uint32_t cy = 0u; cy < kQjlCoresY; ++cy) {
                for (uint32_t cx = 0u; cx < kQjlCoresX; ++cx) {
                    uint32_t vec_idx = cy * kQjlCoresX + cx;
                    if (vec_idx >= chunk.vec_count) continue;
                    tt::tt_metal::CoreCoord c{cx, cy};
                    SetRuntimeArgs(prog, rk, c,
                        {static_cast<uint32_t>(src_addr), 1u, kD, 1u,
                         static_cast<uint32_t>(cent_addr), vec_idx});
                }
            }
        }

        std::map<std::string, std::string> dump_defines = {
            {"TQ_DUMP_CB_ID",      std::to_string(static_cast<uint32_t>(dump_cb))},
            {"TQ_DUMP_PAGE_BYTES", std::to_string(dump_page_bytes)},
        };
        auto wk = CreateKernel(prog,
            "kernels/dataflow/stage_dump_writer.cpp", core_range,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_1,
                .noc       = NOC::RISCV_1_default,
                .defines   = dump_defines,
            });
        SetRuntimeArgs(prog, wk, core,
                       {static_cast<uint32_t>(dump_bufs[ci]->address()),
                        num_tiles, dump_page_bytes});

        distributed::MeshCoordinateRange chunk_range(chunk.coord, chunk.coord);
        workload.add_program(chunk_range, std::move(prog));
    }

    distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/false);

    std::vector<std::vector<uint8_t>> outs(chunks.size());
    for (size_t ci = 0; ci < chunks.size(); ++ci) {
        distributed::ReadShard(cq, outs[ci], dump_bufs[ci], chunks[ci].coord, /*blocking=*/true);
    }
    return outs;
}

static constexpr uint32_t kMaxPolarquantTilesPerBatch = 8u;

static std::vector<std::vector<uint8_t>> run_isolated_stage_batched(
    distributed::MeshDevice*                 mesh_dev,
    distributed::MeshCommandQueue&           cq,
    const std::vector<DeviceChunk>&          chunks,
    uint64_t                                 src_addr,
    int                                      last_stage,
    tt::CB                                   dump_cb,
    uint32_t                                 dump_page_bytes,
    uint64_t                                 cent_addr = 0)
{
    if (last_stage != 1) {
        return run_isolated_stage(mesh_dev, cq, chunks, src_addr, last_stage,
                                  dump_cb, dump_page_bytes, cent_addr);
    }

    const uint32_t max_vecs_per_batch = kMaxPolarquantTilesPerBatch * kN;
    uint32_t max_rounds = 1;
    for (const auto& c : chunks) {
        const uint32_t nb = (c.vec_count + max_vecs_per_batch - 1) / max_vecs_per_batch;
        max_rounds = std::max(max_rounds, std::max(nb, 1u));
    }

    std::vector<std::vector<uint8_t>> per_chunk_out(chunks.size());
    for (uint32_t round = 0; round < max_rounds; ++round) {
        std::vector<DeviceChunk> round_chunks;
        std::vector<size_t>      owner_idx;
        for (size_t ci = 0; ci < chunks.size(); ++ci) {
            const uint32_t round_start = round * max_vecs_per_batch;
            if (round_start >= chunks[ci].vec_count) continue;
            const uint32_t round_count = std::min(max_vecs_per_batch, chunks[ci].vec_count - round_start);
            round_chunks.push_back(DeviceChunk{ chunks[ci].coord, round_start, round_count });
            owner_idx.push_back(ci);
        }
        if (round_chunks.empty()) continue;

        const uint64_t round_src_addr = src_addr +
            static_cast<uint64_t>(round) * max_vecs_per_batch * kD * 2;

        auto round_out = run_isolated_stage(mesh_dev, cq, round_chunks,
                                            round_src_addr, last_stage,
                                            dump_cb, dump_page_bytes, cent_addr);

        for (size_t i = 0; i < round_chunks.size(); ++i) {
            auto& dst = per_chunk_out[owner_idx[i]];
            dst.insert(dst.end(), round_out[i].begin(), round_out[i].end());
        }
    }
    return per_chunk_out;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    uint32_t num_vectors = 64u;
    std::string mode = "full";
    uint32_t requested_mesh_devices = 2u;

    for (int i = 1; i < argc; ++i) {
        std::string a(argv[i]);
        if      (a == "--num-vectors" && i + 1 < argc) num_vectors = std::atoi(argv[++i]);
        else if (a == "--mode"        && i + 1 < argc) mode        = argv[++i];
        else if (a == "--num-devices" && i + 1 < argc) requested_mesh_devices = std::atoi(argv[++i]);
    }

    std::cout << "[TurboQuant] d=" << kD << " b=" << kB
              << " k=" << kK << " N=" << num_vectors << "\n\n";

    auto test_vecs = gen_test_vectors(num_vectors, kD);
    std::vector<uint16_t> input_bf16(num_vectors * kD);
    for (uint32_t i = 0; i < num_vectors; ++i)
        for (uint32_t j = 0; j < kD; ++j)
            input_bf16[i * kD + j] = float_to_bf16(test_vecs[i][j]);
    write_bin("dump_input.bin", input_bf16.data(), input_bf16.size() * 2);

    std::optional<distributed::MeshShape> requested_shape;
    if (requested_mesh_devices == 1) {
        requested_shape = distributed::MeshShape{1, 1};
    } else {
        if (requested_mesh_devices != 2) {
            std::cerr << "[TurboQuant] Note: --num-devices=" << requested_mesh_devices
                      << " requested; auto-detecting system mesh shape.\n";
        }
        requested_shape = std::nullopt;
    }

    auto mesh_dev = distributed::MeshDevice::create(distributed::MeshDeviceConfig(
        /*mesh_shape=*/requested_shape,
        /*offset=*/std::nullopt,
        /*physical_device_ids=*/{}));
    distributed::MeshCommandQueue& cq = mesh_dev->mesh_command_queue();

    auto chunks = plan_device_chunks(mesh_dev.get(), num_vectors);
    const bool sharded = chunks.size() > 1;
    
    auto t0 = std::chrono::high_resolution_clock::now();

    const uint32_t bytes_fp16 = kN * kD * 2;
    const uint32_t bytes_pq   = kN * turboquant::kPolarQuantBytes;
    const uint32_t bytes_qjl  = kN * (turboquant::kQjlBytes + turboquant::kRNormBytes);

    auto input_buf = make_mesh_buf(mesh_dev.get(), num_vectors * kD * 2, kD * 2);
    if (sharded) {
        write_chunks_to_shards(cq, input_buf, chunks, input_bf16, bytes_fp16);
    } else {
        distributed::EnqueueWriteMeshBuffer(cq, input_buf, input_bf16, /*blocking=*/true);
    }
    const uint64_t src_addr = input_buf->address();

    std::cout << "[Stage 0] rotation_kernel -> dump_rotated.bin\n";
    {
        auto raw_parts = run_isolated_stage_batched(mesh_dev.get(), cq, chunks,
                                                    src_addr, 0,
                                                    tt::CB::c_in1, bytes_fp16);
        auto raw = concat_chunks(raw_parts);
        write_bin("dump_rotated.bin", raw.data(), raw.size());
    }

    auto rot_raw = read_bin("dump_rotated.bin");
    auto rot_buf = make_mesh_buf(mesh_dev.get(), rot_raw.size(), kD * 2);
    {
        std::vector<uint16_t> rot_raw_u16(
            reinterpret_cast<uint16_t*>(rot_raw.data()),
            reinterpret_cast<uint16_t*>(rot_raw.data() + rot_raw.size()));
        if (sharded) {
            write_chunks_to_shards(cq, rot_buf, chunks, rot_raw_u16, bytes_fp16);
        } else {
            distributed::EnqueueWriteMeshBuffer(cq, rot_buf, rot_raw_u16, /*blocking=*/true);
        }
    }
    const uint64_t rot_addr = rot_buf->address();
    
    std::cout << "\n[Stage 1a] polarquant_kernel -> dump_quant_indices.bin\n";
    {
        auto raw_parts = run_isolated_stage_batched(mesh_dev.get(), cq, chunks,
                                                    rot_addr, 1,
                                                    tt::CB::c_in2, bytes_pq);
        auto raw = concat_chunks(raw_parts);
        write_bin("dump_quant_indices.bin", raw.data(), raw.size());
    }

    std::cout << "\n[Stage 1b] polarquant_kernel -> dump_quant_centroids.bin\n";
    {
        auto raw_parts = run_isolated_stage_batched(mesh_dev.get(), cq, chunks,
                                                    rot_addr, 1,
                                                    tt::CB::c_in3, bytes_fp16);
        auto raw = concat_chunks(raw_parts);
        write_bin("dump_quant_centroids.bin", raw.data(), raw.size());
    }

    std::cout << "\n[Stage 2] qjl_kernel -> dump_qjl.bin\n";
    {
        auto cent_raw = read_bin("dump_quant_centroids.bin");
        auto cent_buf = make_mesh_buf(mesh_dev.get(), cent_raw.size(), kD * 2);
        {
            std::vector<uint16_t> cent_raw_u16(
                reinterpret_cast<uint16_t*>(cent_raw.data()),
                reinterpret_cast<uint16_t*>(cent_raw.data() + cent_raw.size()));
            if (sharded) {
                write_chunks_to_shards(cq, cent_buf, chunks, cent_raw_u16, bytes_fp16);
            } else {
                distributed::EnqueueWriteMeshBuffer(cq, cent_buf, cent_raw_u16, /*blocking=*/true);
            }
        }
        const uint64_t cent_addr = cent_buf->address();

        const uint32_t kRecBytes       = (kK + 7) / 8 + 2;
        const uint32_t kDramAlign      = tt::tt_metal::hal::get_dram_alignment();
        const uint32_t kRecBytesPadded = (kRecBytes + kDramAlign - 1) & ~(kDramAlign - 1);

        auto grid = mesh_dev->compute_with_storage_grid_size();

        std::vector<std::shared_ptr<distributed::MeshBuffer>> qjl_out_bufs(chunks.size());
        for (size_t ci = 0; ci < chunks.size(); ++ci) {
            qjl_out_bufs[ci] = make_mesh_buf(mesh_dev.get(),
                                             chunks[ci].vec_count * kRecBytesPadded,
                                             kRecBytesPadded);
        }

        std::map<std::string, std::string> qjl_defines = {
            {"TQ_DIM",           std::to_string(kD)},
            {"TQ_ROTATION_SEED", std::to_string(turboquant::kRotationSeed)},
            {"TQ_BITS",          std::to_string(kB)},
            {"TQ_K_VECS",        std::to_string(kN)},
            {"TQ_QJL_DIM",       std::to_string(kK)},
            {"TQ_QJL_SEED",      std::to_string(turboquant::kQjlSeed)},
        };

        uint32_t max_batches = 0;
        for (const auto& chunk : chunks) {
            const uint32_t nb = (chunk.vec_count + kQjlMaxVecsPerDevice - 1) / kQjlMaxVecsPerDevice;
            max_batches = std::max(max_batches, nb);
        }

        for (uint32_t batch = 0; batch < max_batches; ++batch) {
            distributed::MeshWorkload wl2;
            bool any_work = false;

            for (size_t ci = 0; ci < chunks.size(); ++ci) {
                const DeviceChunk& chunk = chunks[ci];
                const uint32_t batch_start = batch * kQjlMaxVecsPerDevice;
                if (batch_start >= chunk.vec_count) continue;
                const uint32_t batch_count = std::min(kQjlMaxVecsPerDevice, chunk.vec_count - batch_start);
                any_work = true;

                const uint64_t qjl_out_addr = qjl_out_bufs[ci]->address();

                Program prog2 = CreateProgram();
                tt::tt_metal::CoreRange qjl_range(
                    tt::tt_metal::CoreCoord{0, 2},
                    tt::tt_metal::CoreCoord{7, 5});

                auto make_cb2 = [&](tt::CB id, uint32_t page, tt::DataFormat fmt) {
                    std::map<uint8_t, tt::DataFormat> spec = {{static_cast<uint8_t>(id), fmt}};
                    CircularBufferConfig cfg(2 * page, spec);
                    cfg.set_page_size(static_cast<uint8_t>(id), page);
                    CreateCircularBuffer(prog2, qjl_range, cfg);
                };
                make_cb2(tt::CB::c_in0,  kD * 2,    tt::DataFormat::Float16_b);
                make_cb2(tt::CB::c_in2,  kD * 2,    tt::DataFormat::Float16_b);
                make_cb2(tt::CB::c_in3,  kD * 2,    tt::DataFormat::Float16_b);
                make_cb2(tt::CB::c_out0, kRecBytes, tt::DataFormat::RawUInt8);

                auto qjl_k = CreateKernel(prog2, "kernels/qjl_kernel.cpp", qjl_range,
                    DataMovementConfig{
                        .processor = DataMovementProcessor::RISCV_0,
                        .noc       = NOC::RISCV_0_default,
                        .defines   = qjl_defines,
                    });

                for (uint32_t cy = 0u; cy < kQjlCoresY; ++cy) {
                    for (uint32_t cx = 0u; cx < kQjlCoresX; ++cx) {
                        const uint32_t local_vi = cy * kQjlCoresX + cx;
                        if (local_vi >= batch_count) continue;
                        const uint32_t vi = batch_start + local_vi;
                        tt::tt_metal::CoreCoord c{cx, cy + 2};
                        SetRuntimeArgs(prog2, qjl_k, c,
                            {static_cast<uint32_t>(src_addr), 1u, kD, 1u,
                             static_cast<uint32_t>(cent_addr), vi,
                             static_cast<uint32_t>(qjl_out_addr),
                             kRecBytesPadded});
                    }
                }

                distributed::MeshCoordinateRange chunk_range(chunk.coord, chunk.coord);
                wl2.add_program(chunk_range, std::move(prog2));
            }

            if (!any_work) continue;
            distributed::EnqueueMeshWorkload(cq, wl2, /*blocking=*/true);
        }

        std::vector<uint8_t> raw(num_vectors * kRecBytes);
        for (size_t ci = 0; ci < chunks.size(); ++ci) {
            const DeviceChunk& chunk = chunks[ci];
            std::vector<uint8_t> raw_padded;
            distributed::ReadShard(cq, raw_padded, qjl_out_bufs[ci], chunk.coord, /*blocking=*/true);
            for (uint32_t i = 0; i < chunk.vec_count; ++i) {
                std::memcpy(raw.data() + (chunk.vec_offset + i) * kRecBytes,
                            raw_padded.data() + i * kRecBytesPadded,
                            kRecBytes);
            }
        }
        write_bin("dump_qjl.bin", raw.data(), raw.size());
    }
    {
        auto pq_raw  = read_bin("dump_quant_indices.bin");
        auto qjl_raw = read_bin("dump_qjl.bin");

        const uint32_t pq_page  = bytes_pq;
        const uint32_t qjl_page = bytes_qjl;
        const uint32_t pq_vec   = turboquant::kPolarQuantBytes;
        const uint32_t qjl_vec  = turboquant::kQjlBytes + turboquant::kRNormBytes;
        const uint32_t rec      = turboquant::kRecordBytes;

        std::vector<uint8_t> output(num_vectors * rec, 0u);
        for (uint32_t v = 0; v < num_vectors; ++v) {
            uint32_t ti = v / kN, vi = v % kN;
            const uint8_t* pq_src  = pq_raw.data()  + ti * pq_page  + vi * pq_vec;
            const uint8_t* qjl_src = qjl_raw.data() + ti * qjl_page + vi * qjl_vec;
            uint8_t* dst = output.data() + v * rec;
            memcpy(dst + turboquant::kOffsetPolarQuant, pq_src,  pq_vec);
            memcpy(dst + turboquant::kOffsetQjlBits,   qjl_src, turboquant::kQjlBytes);
            memcpy(dst + turboquant::kOffsetRNorm,
                   qjl_src + turboquant::kQjlBytes, turboquant::kRNormBytes);
        }
        write_bin("dump_output.bin", output.data(), output.size());
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    mesh_dev->close();

    std::cout << "\nTotal time: " << std::fixed << std::setprecision(1)
              << ms << " ms\n\n"
              << "Validate with:\n"
              << "  python tests/test_rotation.py   --device\n"
              << "  python tests/test_polarquant.py --device\n"
              << "  python tests/test_qjl.py        --device\n"
              << "  python tests/test_roundtrip.py  --device\n";
    return 0;
}