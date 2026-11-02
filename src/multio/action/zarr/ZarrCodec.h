/*
 * (C) Copyright 2026- ECMWF.
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 * In applying this licence, ECMWF does not waive the privileges and immunities
 * granted to it by virtue of its status as an intergovernmental organisation nor
 * does it submit to any jurisdiction.
 */

/// Chunk compression of the zarr action (zarr v3 codec "zstd") and the worker pool that runs it off the
/// message-receiving thread.
///
/// Threading model: the action owns one WorkerPool (compression-threads workers, a bounded queue). For every
/// field the store hands a job (compress the raw values, then write the chunk file or keep the compressed bytes
/// for the shard) to the pool and returns; the job is counted in the JobGroup of its output step. Closing a step
/// waits for that group only, then renames / writes shards / metadata on the calling thread as without
/// compression. A full queue blocks the submitter (backpressure), so at most queue + threads raw fields are
/// buffered. An exception of a job is kept in its group and rethrown by JobGroup::wait.

#pragma once

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace multio::action::zarr {

//----------------------------------------------------------------------------------------------------------------------
// Codec

enum class Codec
{
    None,  // bytes codec only: output identical to the uncompressed action
    Zstd,
};

struct Compression {
    Codec codec = Codec::None;
    int level = 3;

    bool enabled() const { return codec != Codec::None; }

    /// JSON of the codecs after the bytes codec, e.g. ,{"name":"zstd",...}; empty without compression
    std::string codecsJsonTail() const;

    /// Short description for logs, e.g. "zstd level 3" or "none"
    std::string describe() const;
};

/// True if the action was built with libzstd
bool zstdAvailable();

/// Range of zstd levels accepted (ZSTD_minCLevel() .. ZSTD_maxCLevel()); {0, 0} without zstd
std::pair<int, int> zstdLevelRange();

/// Compress `nbytes` bytes with `c` (must be enabled) into `out` (resized to the compressed size; the capacity
/// is not shrunk, see compressExact). Uses one compression context per thread.
void compressInto(const Compression& c, const void* data, std::size_t nbytes, std::vector<unsigned char>& out);

/// As compressInto, but `out` is allocated with exactly the compressed size (for buffers kept until the flush)
std::vector<unsigned char> compressExact(const Compression& c, const void* data, std::size_t nbytes);

/// Decompress a zstd frame of known content size (tests)
std::vector<unsigned char> decompressZstd(const void* data, std::size_t nbytes);

//----------------------------------------------------------------------------------------------------------------------
// Jobs

/// Counter of the outstanding jobs of one output step, with the first error of any of them
class JobGroup {
public:
    void add();
    void done(std::exception_ptr error) noexcept;

    /// Wait for all jobs; rethrows the first error (once: the error is cleared)
    void wait();

    /// Wait for all jobs, ignore errors (destructor, discarded steps); returns true if one failed
    bool waitNoThrow() noexcept;

private:
    std::mutex mutex_;
    std::condition_variable cv_;
    std::size_t pending_ = 0;
    std::exception_ptr error_;
};

/// Fixed number of worker threads and a bounded FIFO of jobs
class WorkerPool {
public:
    /// `threads` >= 1; at most `maxQueued` jobs wait (submit blocks beyond that)
    WorkerPool(std::size_t threads, std::size_t maxQueued);
    ~WorkerPool();  // runs the remaining jobs, joins the threads

    WorkerPool(const WorkerPool&) = delete;
    WorkerPool& operator=(const WorkerPool&) = delete;

    /// Run `job` on a worker; it is counted in `group` until it has finished. Blocks while the queue is full.
    void submit(const std::shared_ptr<JobGroup>& group, std::function<void()> job);

    std::size_t threads() const { return workers_.size(); }
    std::size_t maxQueued() const { return maxQueued_; }

    /// Number of submit calls that had to wait for a free queue slot
    std::size_t blockedSubmits() const;

private:
    struct Job {
        std::shared_ptr<JobGroup> group;
        std::function<void()> run;
    };
    void work();

    std::size_t maxQueued_;
    mutable std::mutex mutex_;
    std::condition_variable notEmpty_;
    std::condition_variable notFull_;
    std::deque<Job> queue_;
    bool stop_ = false;
    std::size_t blocked_ = 0;
    std::vector<std::thread> workers_;
};

}  // namespace multio::action::zarr
