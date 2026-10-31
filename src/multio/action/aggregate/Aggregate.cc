/*
 * (C) Copyright 1996- ECMWF.
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 * In applying this licence, ECMWF does not waive the privileges and immunities
 * granted to it by virtue of its status as an intergovernmental organisation nor
 * does it submit to any jurisdiction.
 */

#include "Aggregate.h"

#include <algorithm>
#include <cctype>
#include <string>
#include <type_traits>

#include "multio/LibMultio.h"
#include "multio/domain/Mappings.h"

namespace multio::action::aggregate {

using message::Peer;

namespace {

/// Metadata key `skipAggregation` true: the field is complete already (e.g. static coordinates sent by one client
/// task) and is passed on unchanged. Accepted as bool, integer (non-zero), real (non-zero) or string
/// (true/1/yes/t/.true., case-insensitive).
bool skipAggregation(const Message& msg) {
    const auto& md = msg.metadata();
    auto it = md.find(std::string("skipAggregation"));
    if (it == md.end()) {
        return false;
    }
    return it->second.visit([](const auto& v) -> bool {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, bool>) {
            return v;
        }
        else if constexpr (std::is_same_v<T, std::int64_t> || std::is_same_v<T, double> || std::is_same_v<T, float>) {
            return v != 0;
        }
        else if constexpr (std::is_same_v<T, std::string>) {
            std::string s;
            for (char c : v) {
                if (!std::isspace(static_cast<unsigned char>(c))) {
                    s.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
                }
            }
            return s == "true" || s == "1" || s == "yes" || s == "t" || s == ".true.";
        }
        else {
            return false;
        }
    });
}

}  // namespace

Aggregate::Aggregate(const ComponentConfiguration& compConf) : ChainedAction(compConf) {}

void Aggregate::executeImpl(Message msg) {

    if ((msg.tag() == Message::Tag::Field) && skipAggregation(msg)) {
        executeNext(std::move(msg));
        return;
    }

    if ((msg.tag() == Message::Tag::Field) && handleField(msg)) {
        executeNext(globalField(msg.fieldId()));
    }

    if ((msg.tag() == Message::Tag::Flush) && handleFlush(msg)) {
        executeNext(globalFlush(msg.fieldId()));
    }
}

bool Aggregate::handleField(const Message& msg) {
    util::ScopedTiming timing{statistics_.actionTiming_};
    if (not aggCatalogue_.contains(msg.fieldId())) {
        aggCatalogue_.addNew(msg);
    }
    // TODO: Perhaps call collect indices here and store it for a later call on check consistnecy
    domain::Mappings::instance()
        .get(msg.domain())
        .at(msg.source())
        ->toGlobal(msg, aggCatalogue_.getMessage(msg.fieldId()));
    aggCatalogue_.bookProcessedPart(msg.fieldId(), msg.source());
    return allPartsArrived(msg);
}

auto Aggregate::flushCount(const Message& msg) {
    auto res = flushes_[msg.fieldId()].emplace(msg.source());

    if (not res.second) {
        std::ostringstream os;
        os << "Flush message " << msg << " has already been received";
        throw eckit::UserError(os.str(), Here());
    }

    return flushes_.at(msg.fieldId()).size();
}

bool Aggregate::handleFlush(const Message& msg) {
    // Initialise if need be
    util::ScopedTiming timing{statistics_.actionTiming_};
    // to allow flushes coming without a domain to direclty pass through.
    auto domain = msg.metadata().get<std::string>("domain");
    if (domain == "global") {
        return true;
    }
    // get domain info if existant
    const auto& domainMap = domain::Mappings::instance().get(domain);

    auto flCount = flushCount(msg);

    if (domainMap.isComplete() && flCount == domainMap.size()) {
        // if complete, pass through and reset counter
        flushes_.erase(msg.fieldId());
        return true;
    }
    return false;
}

bool Aggregate::allPartsArrived(const Message& msg) const {
    LOG_DEBUG_LIB(LibMultio) << " *** Number of messages for field " << msg.fieldId() << " are "
                             << aggCatalogue_.partsCount(msg.fieldId()) << std::endl;

    const auto& domainMap = domain::Mappings::instance().get(msg.domain());

    return domainMap.isComplete() && (aggCatalogue_.partsCount(msg.fieldId()) == domainMap.size());
}

Message Aggregate::globalField(const std::string& fid) {
    util::ScopedTiming timing{statistics_.actionTiming_};

    // TODO: checking domain consistency is skipped for now...
    // domain::Mappings::instance().checkDomainConsistency(messages_.at(fid));

    return aggCatalogue_.extract(fid);
}

Message Aggregate::globalFlush(const std::string& fid) {
    util::ScopedTiming timing{statistics_.actionTiming_};

    auto flush = flushes_.extract(fid);

    return Message{{Message::Tag::Flush, Peer{}, Peer{}, std::string(fid)}};
}

void Aggregate::print(std::ostream& os) const {
    os << "Aggregate(for " << aggCatalogue_.size() << " fields = [" << aggCatalogue_ << "])";
}


static ActionBuilder<Aggregate> AggregateBuilder("aggregate");

}  // namespace multio::action::aggregate
