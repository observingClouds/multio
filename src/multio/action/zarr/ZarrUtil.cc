/*
 * (C) Copyright 2026- ECMWF.
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 * In applying this licence, ECMWF does not waive the privileges and immunities
 * granted to it by virtue of its status as an intergovernmental organisation nor
 * does it submit to any jurisdiction.
 */

#include "ZarrUtil.h"

#include <atomic>
#include <cctype>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>

#include <dirent.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "eckit/exception/Exceptions.h"

namespace multio::action::zarr {

namespace {

[[noreturn]] void ioError(const std::string& what, const std::string& path, int err) {
    std::ostringstream oss;
    oss << "zarr action: " << what << " '" << path << "': " << std::strerror(err);
    throw eckit::WriteError(oss.str(), Here());
}

[[noreturn]] void jsonError(const std::string& what, std::size_t pos) {
    std::ostringstream oss;
    oss << "zarr action: malformed JSON (" << what << ") at offset " << pos;
    throw eckit::BadValue(oss.str(), Here());
}

std::size_t skipWs(std::string_view s, std::size_t pos) {
    while (pos < s.size() && (s[pos] == ' ' || s[pos] == '\n' || s[pos] == '\r' || s[pos] == '\t')) {
        ++pos;
    }
    return pos;
}

std::size_t skipString(std::string_view s, std::size_t pos) {
    // s[pos] == '"'
    ++pos;
    while (pos < s.size()) {
        if (s[pos] == '\\') {
            pos += 2;
            continue;
        }
        if (s[pos] == '"') {
            return pos + 1;
        }
        ++pos;
    }
    jsonError("unterminated string", pos);
}

}  // namespace

//----------------------------------------------------------------------------------------------------------------------

std::string jsonString(std::string_view s) {
    std::string out;
    out.reserve(s.size() + 2);
    out.push_back('"');
    for (char c : s) {
        switch (c) {
            case '"':
                out += "\\\"";
                break;
            case '\\':
                out += "\\\\";
                break;
            case '\n':
                out += "\\n";
                break;
            case '\t':
                out += "\\t";
                break;
            case '\r':
                out += "\\r";
                break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned>(static_cast<unsigned char>(c)));
                    out += buf;
                }
                else {
                    out.push_back(c);
                }
        }
    }
    out.push_back('"');
    return out;
}

std::string jsonIntArray(const std::vector<std::int64_t>& v) {
    std::string out = "[";
    for (std::size_t i = 0; i < v.size(); ++i) {
        if (i > 0) {
            out.push_back(',');
        }
        out += std::to_string(v[i]);
    }
    out.push_back(']');
    return out;
}

std::string jsonStringArray(const std::vector<std::string>& v) {
    std::string out = "[";
    for (std::size_t i = 0; i < v.size(); ++i) {
        if (i > 0) {
            out.push_back(',');
        }
        out += jsonString(v[i]);
    }
    out.push_back(']');
    return out;
}

//----------------------------------------------------------------------------------------------------------------------

std::size_t jsonSkipValue(std::string_view s, std::size_t pos) {
    pos = skipWs(s, pos);
    if (pos >= s.size()) {
        jsonError("unexpected end", pos);
    }
    char c = s[pos];
    if (c == '"') {
        return skipString(s, pos);
    }
    if (c == '{' || c == '[') {
        int depth = 0;
        while (pos < s.size()) {
            char d = s[pos];
            if (d == '"') {
                pos = skipString(s, pos);
                continue;
            }
            if (d == '{' || d == '[') {
                ++depth;
            }
            else if (d == '}' || d == ']') {
                --depth;
                if (depth == 0) {
                    return pos + 1;
                }
            }
            ++pos;
        }
        jsonError("unterminated container", pos);
    }
    // number, true, false, null
    while (pos < s.size() && s[pos] != ',' && s[pos] != '}' && s[pos] != ']' && s[pos] != ' ' && s[pos] != '\n'
           && s[pos] != '\r' && s[pos] != '\t') {
        ++pos;
    }
    return pos;
}

std::vector<std::pair<std::string, std::string>> jsonObjectMembers(std::string_view raw) {
    std::vector<std::pair<std::string, std::string>> members;
    std::size_t pos = skipWs(raw, 0);
    if (pos >= raw.size() || raw[pos] != '{') {
        jsonError("object expected", pos);
    }
    pos = skipWs(raw, pos + 1);
    if (pos < raw.size() && raw[pos] == '}') {
        return members;
    }
    while (pos < raw.size()) {
        pos = skipWs(raw, pos);
        if (raw[pos] != '"') {
            jsonError("key expected", pos);
        }
        std::size_t kend = skipString(raw, pos);
        std::string key = jsonUnquote(raw.substr(pos, kend - pos));
        pos = skipWs(raw, kend);
        if (pos >= raw.size() || raw[pos] != ':') {
            jsonError("':' expected", pos);
        }
        std::size_t vbeg = skipWs(raw, pos + 1);
        std::size_t vend = jsonSkipValue(raw, vbeg);
        members.emplace_back(std::move(key), std::string(raw.substr(vbeg, vend - vbeg)));
        pos = skipWs(raw, vend);
        if (pos >= raw.size()) {
            jsonError("unterminated object", pos);
        }
        if (raw[pos] == '}') {
            return members;
        }
        if (raw[pos] != ',') {
            jsonError("',' expected", pos);
        }
        ++pos;
    }
    jsonError("unterminated object", pos);
}

std::vector<std::string> jsonArrayElements(std::string_view raw) {
    std::vector<std::string> elems;
    std::size_t pos = skipWs(raw, 0);
    if (pos >= raw.size() || raw[pos] != '[') {
        jsonError("array expected", pos);
    }
    pos = skipWs(raw, pos + 1);
    if (pos < raw.size() && raw[pos] == ']') {
        return elems;
    }
    while (pos < raw.size()) {
        std::size_t vbeg = skipWs(raw, pos);
        std::size_t vend = jsonSkipValue(raw, vbeg);
        elems.emplace_back(raw.substr(vbeg, vend - vbeg));
        pos = skipWs(raw, vend);
        if (pos >= raw.size()) {
            break;
        }
        if (raw[pos] == ']') {
            return elems;
        }
        if (raw[pos] != ',') {
            jsonError("',' expected in array", pos);
        }
        ++pos;
    }
    jsonError("unterminated array", pos);
}

std::string jsonUnquote(std::string_view raw) {
    std::size_t b = skipWs(raw, 0);
    std::size_t e = raw.size();
    while (e > b && (raw[e - 1] == ' ' || raw[e - 1] == '\n' || raw[e - 1] == '\r' || raw[e - 1] == '\t')) {
        --e;
    }
    if (e - b < 2 || raw[b] != '"' || raw[e - 1] != '"') {
        return std::string(raw.substr(b, e - b));
    }
    std::string out;
    for (std::size_t i = b + 1; i + 1 < e; ++i) {
        char c = raw[i];
        if (c == '\\' && i + 2 < e) {
            char n = raw[++i];
            switch (n) {
                case 'n':
                    out.push_back('\n');
                    break;
                case 't':
                    out.push_back('\t');
                    break;
                case 'r':
                    out.push_back('\r');
                    break;
                case 'b':
                    out.push_back('\b');
                    break;
                case 'f':
                    out.push_back('\f');
                    break;
                case 'u':
                    // keep \uXXXX for non-ASCII, decode ASCII range
                    if (i + 4 < e) {
                        unsigned v = std::stoul(std::string(raw.substr(i + 1, 4)), nullptr, 16);
                        if (v < 0x80) {
                            out.push_back(static_cast<char>(v));
                        }
                        else {
                            out += "\\u";
                            out += raw.substr(i + 1, 4);
                        }
                        i += 4;
                    }
                    break;
                default:
                    out.push_back(n);
            }
        }
        else {
            out.push_back(c);
        }
    }
    return out;
}

std::string jsonMember(std::string_view raw, std::string_view key) {
    for (auto& m : jsonObjectMembers(raw)) {
        if (m.first == key) {
            return std::move(m.second);
        }
    }
    return {};
}

namespace {

/// Recursive-descent JSON validator that copies the document without insignificant whitespace
class JsonCompactor {
public:
    JsonCompactor(std::string_view s, const std::string& what) : s_(s), what_(what) { out_.reserve(s.size()); }

    std::string object() {
        ws();
        if (pos_ >= s_.size() || s_[pos_] != '{') {
            fail("a JSON object is expected");
        }
        value(0);
        ws();
        if (pos_ != s_.size()) {
            fail("trailing characters after the object");
        }
        return std::move(out_);
    }

private:
    [[noreturn]] void fail(const std::string& why) const {
        std::ostringstream oss;
        oss << "zarr action: " << what_ << ": malformed JSON, " << why << " at offset " << pos_;
        throw eckit::UserError(oss.str(), Here());
    }

    void ws() {
        while (pos_ < s_.size() && (s_[pos_] == ' ' || s_[pos_] == '\n' || s_[pos_] == '\r' || s_[pos_] == '\t')) {
            ++pos_;
        }
    }

    bool isDig(char c) const { return c >= '0' && c <= '9'; }

    void string() {
        // s_[pos_] == '"'
        std::size_t b = pos_++;
        while (pos_ < s_.size()) {
            unsigned char c = static_cast<unsigned char>(s_[pos_]);
            if (c == '"') {
                ++pos_;
                out_.append(s_.substr(b, pos_ - b));
                return;
            }
            if (c < 0x20) {
                fail("control character in a string");
            }
            if (c == '\\') {
                if (pos_ + 1 >= s_.size()) {
                    break;
                }
                char e = s_[pos_ + 1];
                if (e == 'u') {
                    if (pos_ + 5 >= s_.size()) {
                        break;
                    }
                    for (std::size_t k = pos_ + 2; k < pos_ + 6; ++k) {
                        if (!std::isxdigit(static_cast<unsigned char>(s_[k]))) {
                            fail("bad \\u escape");
                        }
                    }
                    pos_ += 6;
                    continue;
                }
                if (std::string_view("\"\\/bfnrt").find(e) == std::string_view::npos) {
                    fail("bad escape");
                }
                pos_ += 2;
                continue;
            }
            ++pos_;
        }
        fail("unterminated string");
    }

    void number() {
        std::size_t b = pos_;
        if (s_[pos_] == '-') {
            ++pos_;
        }
        if (pos_ >= s_.size() || !isDig(s_[pos_])) {
            fail("bad number");
        }
        if (s_[pos_] == '0') {
            ++pos_;
        }
        else {
            while (pos_ < s_.size() && isDig(s_[pos_])) {
                ++pos_;
            }
        }
        if (pos_ < s_.size() && s_[pos_] == '.') {
            ++pos_;
            if (pos_ >= s_.size() || !isDig(s_[pos_])) {
                fail("bad number");
            }
            while (pos_ < s_.size() && isDig(s_[pos_])) {
                ++pos_;
            }
        }
        if (pos_ < s_.size() && (s_[pos_] == 'e' || s_[pos_] == 'E')) {
            ++pos_;
            if (pos_ < s_.size() && (s_[pos_] == '+' || s_[pos_] == '-')) {
                ++pos_;
            }
            if (pos_ >= s_.size() || !isDig(s_[pos_])) {
                fail("bad number");
            }
            while (pos_ < s_.size() && isDig(s_[pos_])) {
                ++pos_;
            }
        }
        out_.append(s_.substr(b, pos_ - b));
    }

    void literal(std::string_view lit) {
        if (s_.substr(pos_, lit.size()) != lit) {
            fail("unexpected character");
        }
        out_.append(lit);
        pos_ += lit.size();
    }

    void value(int depth) {
        if (depth > 64) {
            fail("nesting too deep");
        }
        ws();
        if (pos_ >= s_.size()) {
            fail("unexpected end");
        }
        char c = s_[pos_];
        if (c == '{' || c == '[') {
            const bool obj = c == '{';
            const char close = obj ? '}' : ']';
            out_.push_back(c);
            ++pos_;
            ws();
            if (pos_ < s_.size() && s_[pos_] == close) {
                out_.push_back(close);
                ++pos_;
                return;
            }
            while (true) {
                ws();
                if (obj) {
                    if (pos_ >= s_.size() || s_[pos_] != '"') {
                        fail("member name expected");
                    }
                    string();
                    ws();
                    if (pos_ >= s_.size() || s_[pos_] != ':') {
                        fail("':' expected");
                    }
                    out_.push_back(':');
                    ++pos_;
                }
                value(depth + 1);
                ws();
                if (pos_ >= s_.size()) {
                    fail("unterminated container");
                }
                if (s_[pos_] == ',') {
                    out_.push_back(',');
                    ++pos_;
                    continue;
                }
                if (s_[pos_] == close) {
                    out_.push_back(close);
                    ++pos_;
                    return;
                }
                fail("',' or end of container expected");
            }
        }
        if (c == '"') {
            string();
        }
        else if (c == '-' || isDig(c)) {
            number();
        }
        else if (c == 't') {
            literal("true");
        }
        else if (c == 'f') {
            literal("false");
        }
        else if (c == 'n') {
            literal("null");
        }
        else {
            fail("unexpected character");
        }
    }

    std::string_view s_;
    const std::string& what_;
    std::size_t pos_ = 0;
    std::string out_;
};

}  // namespace

std::string jsonCompactObject(std::string_view raw, const std::string& what) {
    return JsonCompactor(raw, what).object();
}

std::string jsonSetMember(std::string_view obj, const std::string& key, const std::string& value) {
    std::string out = "{";
    bool found = false;
    for (const auto& m : jsonObjectMembers(obj)) {
        if (out.size() > 1) {
            out.push_back(',');
        }
        out += jsonString(m.first);
        out.push_back(':');
        if (m.first == key) {
            out += value;
            found = true;
        }
        else {
            out += m.second;
        }
    }
    if (!found) {
        if (out.size() > 1) {
            out.push_back(',');
        }
        out += jsonString(key);
        out.push_back(':');
        out += value;
    }
    out.push_back('}');
    return out;
}

//----------------------------------------------------------------------------------------------------------------------

void makeDirs(const std::string& path) {
    if (path.empty()) {
        return;
    }
    if (::mkdir(path.c_str(), 0755) == 0) {
        return;
    }
    int err = errno;
    if (err == EEXIST) {
        struct stat st;
        if (::stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode)) {
            return;
        }
        ioError("not a directory", path, ENOTDIR);
    }
    if (err == ENOENT) {
        std::string parent = dirName(path);
        if (parent.empty() || parent == path) {
            ioError("cannot create directory", path, err);
        }
        makeDirs(parent);
        if (::mkdir(path.c_str(), 0755) == 0 || errno == EEXIST) {
            return;
        }
        err = errno;
    }
    ioError("cannot create directory", path, err);
}

void writeFileAtomic(const std::string& path, const void* data, std::size_t nbytes) {
    static std::atomic<unsigned long> counter{0};
    std::string tmp = path + ".tmp" + std::to_string(::getpid()) + "_" + std::to_string(counter++);

    int fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
        ioError("cannot create", tmp, errno);
    }
    const char* p = static_cast<const char*>(data);
    std::size_t left = nbytes;
    while (left > 0) {
        ssize_t w = ::write(fd, p, left);
        if (w < 0) {
            if (errno == EINTR) {
                continue;
            }
            int err = errno;
            ::close(fd);
            ::unlink(tmp.c_str());
            ioError("cannot write", tmp, err);
        }
        p += w;
        left -= static_cast<std::size_t>(w);
    }
    if (::close(fd) != 0) {
        int err = errno;
        ::unlink(tmp.c_str());
        ioError("cannot close", tmp, err);
    }
    if (::rename(tmp.c_str(), path.c_str()) != 0) {
        int err = errno;
        ::unlink(tmp.c_str());
        ioError("cannot rename to", path, err);
    }
}

std::string readFile(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        ioError("cannot open", path, errno ? errno : ENOENT);
    }
    std::ostringstream oss;
    oss << in.rdbuf();
    return oss.str();
}

bool pathExists(const std::string& path) {
    struct stat st;
    return ::stat(path.c_str(), &st) == 0;
}

bool isDirectory(const std::string& path) {
    struct stat st;
    return ::stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

bool isRegularFile(const std::string& path) {
    struct stat st;
    return ::stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

std::vector<std::string> listDirectory(const std::string& path) {
    DIR* d = ::opendir(path.c_str());
    if (d == nullptr) {
        ioError("cannot open directory", path, errno);
    }
    std::vector<std::string> names;
    for (;;) {
        errno = 0;
        struct dirent* e = ::readdir(d);
        if (e == nullptr) {
            int err = errno;
            ::closedir(d);
            if (err != 0) {
                ioError("cannot read directory", path, err);
            }
            break;
        }
        if (std::strcmp(e->d_name, ".") == 0 || std::strcmp(e->d_name, "..") == 0) {
            continue;
        }
        names.emplace_back(e->d_name);
    }
    return names;
}

namespace {

void listFilesBelow(const std::string& dir, const std::string& rel, std::vector<std::string>& out) {
    for (const auto& name : listDirectory(dir)) {
        std::string path = dir + "/" + name;
        std::string r = rel.empty() ? name : rel + "/" + name;
        struct stat st;
        if (::lstat(path.c_str(), &st) != 0) {
            continue;
        }
        if (S_ISDIR(st.st_mode)) {
            listFilesBelow(path, r, out);
        }
        else if (S_ISREG(st.st_mode) || (S_ISLNK(st.st_mode) && isRegularFile(path))) {
            out.push_back(std::move(r));
        }
    }
}

}  // namespace

std::vector<std::string> listFilesRecursive(const std::string& dir) {
    std::vector<std::string> out;
    listFilesBelow(dir, std::string(), out);
    return out;
}

void renamePath(const std::string& from, const std::string& to) {
    if (::rename(from.c_str(), to.c_str()) != 0) {
        ioError("cannot rename '" + from + "' to", to, errno);
    }
}

void removeFile(const std::string& path) noexcept {
    ::unlink(path.c_str());
}

void DirectoryCache::ensure(const std::string& dir) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (known_.find(dir) != known_.end()) {
            return;
        }
    }
    makeDirs(dir);  // outside the lock: no serialisation of the mkdir calls of different directories
    std::lock_guard<std::mutex> lock(mutex_);
    known_.insert(dir);
}

std::string dirName(const std::string& path) {
    auto p = path.find_last_of('/');
    if (p == std::string::npos) {
        return std::string();
    }
    if (p == 0) {
        return std::string("/");
    }
    return path.substr(0, p);
}

}  // namespace multio::action::zarr
