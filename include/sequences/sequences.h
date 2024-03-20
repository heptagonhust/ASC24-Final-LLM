#pragma once
#include "rpc/msgpack.hpp"

struct Sequence
{
    std::vector<int32_t> inputIds;
    int32_t outputLen;
    int32_t reqId;
    MSGPACK_DEFINE_ARRAY(inputIds, outputLen, reqId)
};
using Sequences = std::vector<Sequence>;