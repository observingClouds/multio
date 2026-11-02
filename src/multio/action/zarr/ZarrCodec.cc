/*
 * (C) Copyright 2026- ECMWF.
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 * In applying this licence, ECMWF does not waive the privileges and immunities
 * granted to it by virtue of its status as an intergovernmental organisation nor
 * does it submit to any jurisdiction.
 */

#include "ZarrCodec.h"

#include <algorithm>
#include <cstring>

#include "eckit/exception/Exceptions.h"

#ifdef MULTIO_ZARR_HAVE_ZSTD
#include <zstd.h>
#endif

namespace multio::action::zarr {

//----------------------------------------------------------------------------------------------------------------------

std::string Compression::codecsJsonTail() const {
    switch (codec) {
        case Codec::None:
            return std::string();
        case Codec::Zstd:
            return ",{\"name\":\"zstd\",\"configuration\":{\"level\":" + std::to_string(level)
                 + ",\"checksum\":false}}";
    }
    return std::string();
}

std::string Compression::describe() const {
    switch (codec) {
        case Codec::None:
            return "none";
        case Codec::Zstd:
            return "zstd level " + std::to_string(level);
    }
    return "?";
}

#ifdef MULTIO_ZARR_HAVE_ZSTD

namespace {

/// One compression context per thread (workers and, with compression-threads 0, the receiving thread)
struct ThreadContext {
    ZSTD_CCtx* cctx = nullptr;
    std::vector<unsigned char> scratch;
    ~ThreadContext() {
        if (cctx) {
            ZSTD_freeCCtx(cctx);
        }
    }
};

ThreadContext& threadContext() {
    thread_local ThreadContext ctx;
    if (!ctx.cctx) {
        ctx.cctx = ZSTD_createCCtx();
        if (!ctx.cctx) {
            throw eckit::SeriousBug("zarr action: ZSTD_createCCtx failed", Here());
        }
    }
    return ctx;
}

std::size_t zstdCompress(const Compression& c, const void* data, std::size_t nbytes, std::vector<unsigned char>& out) {
    ThreadContext& ctx = threadContext();
    out.resize(ZSTD_compressBound(nbytes));
    // the one-shot API without parameters: content size in the frame header, no checksum (checksum: false)
    const std::size_t r = ZSTD_compressCCtx(ctx.cctx, out.data(), out.size(), data, nbytes, c.level);
    if (ZSTD_isError(r)) {
        throw eckit::SeriousBug(std::string("zarr action: zstd compression failed: ") + ZSTD_getErrorName(r), Here());
    }
    return r;
}

}  // namespace

bool zstdAvailable() {
    return true;
}

std::pair<int, int> zstdLevelRange() {
    return {ZSTD_minCLevel(), ZSTD_maxCLevel()};
}

void compressInto(const Compression& c, const void* data, std::size_t nbytes, std::vector<unsigned char>& out) {
    ASSERT(c.codec == Codec::Zstd);
    out.resize(zstdCompress(c, data, nbytes, out));
}

std::vector<unsigned char> compressExact(const Compression& c, const void* data, std::size_t nbytes) {
    ASSERT(c.codec == Codec::Zstd);
    std::vector<unsigned char>& scratch = threadContext().scratch;
    const std::size_t n = zstdCompress(c, data, nbytes, scratch);
    return std::vector<unsigned char>(scratch.begin(), scratch.begin() + static_cast<std::ptrdiff_t>(n));
}

std::vector<unsigned char> decompressZstd(const void* data, std::size_t nbytes) {
    const unsigned long long size = ZSTD_getFrameContentSize(data, nbytes);
    if (size == ZSTD_CONTENTSIZE_ERROR || size == ZSTD_CONTENTSIZE_UNKNOWN) {
        throw eckit::BadValue("zarr action: not a zstd frame with content size", Here());
    }
    std::vector<unsigned char> out(static_cast<std::size_t>(size));
    const std::size_t r = ZSTD_decompress(out.data(), out.size(), data, nbytes);
    if (ZSTD_isError(r) || r != out.size()) {
        throw eckit::BadValue("zarr action: zstd decompression failed", Here());
    }
    return out;
}

#else

bool zstdAvailable() {
    return false;
}

std::pair<int, int> zstdLevelRange() {
    return {0, 0};
}

void compressInto(const Compression&, const void*, std::size_t, std::vector<unsigned char>&) {
    throw eckit::UserError("zarr action: built without zstd", Here());
}

std::vector<unsigned char> compressExact(const Compression&, const void*, std::size_t) {
    throw eckit::UserError("zarr action: built without zstd", Here());
}

std::vector<unsigned char> decompressZstd(const void*, std::size_t) {
    throw eckit::UserError("zarr action: built without zstd", Here());
}

#endif

//----------------------------------------------------------------------------------------------------------------------

void JobGroup::add() {
    std::lock_guard<std::mutex> lock(mutex_);
    ++pending_;
}

void JobGroup::done(std::exception_ptr error) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    if (error && !error_) {
        error_ = error;
    }
    if (--pending_ == 0) {
        cv_.notify_all();
    }
}

void JobGroup::wait() {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this] { return pending_ == 0; });
    if (error_) {
        std::exception_ptr e = error_;
        error_ = nullptr;
        std::rethrow_exception(e);
    }
}

bool JobGroup::waitNoThrow() noexcept {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this] { return pending_ == 0; });
    const bool failed = static_cast<bool>(error_);
    error_ = nullptr;
    return failed;
}

//----------------------------------------------------------------------------------------------------------------------

WorkerPool::WorkerPool(std::size_t threads, std::size_t maxQueued) : maxQueued_{std::max<std::size_t>(1, maxQueued)} {
    ASSERT(threads >= 1);
    workers_.reserve(threads);
    for (std::size_t i = 0; i < threads; ++i) {
        workers_.emplace_back([this] { work(); });
    }
}

WorkerPool::~WorkerPool() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_ = true;
    }
    notEmpty_.notify_all();
    for (auto& t : workers_) {
        t.join();
    }
}

void WorkerPool::submit(const std::shared_ptr<JobGroup>& group, std::function<void()> job) {
    group->add();
    {
        std::unique_lock<std::mutex> lock(mutex_);
        if (queue_.size() >= maxQueued_) {
            ++blocked_;
            notFull_.wait(lock, [this] { return queue_.size() < maxQueued_; });
        }
        queue_.push_back(Job{group, std::move(job)});
    }
    notEmpty_.notify_one();
}

std::size_t WorkerPool::blockedSubmits() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return blocked_;
}

void WorkerPool::work() {
    while (true) {
        Job job;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            notEmpty_.wait(lock, [this] { return stop_ || !queue_.empty(); });
            if (queue_.empty()) {
                return;  // stop_ and drained
            }
            job = std::move(queue_.front());
            queue_.pop_front();
        }
        notFull_.notify_one();
        std::exception_ptr error;
        try {
            job.run();
        }
        catch (...) {
            error = std::current_exception();
        }
        job.run = nullptr;  // free the job's buffers before the group sees it as done
        job.group->done(error);
    }
}

}  // namespace multio::action::zarr
