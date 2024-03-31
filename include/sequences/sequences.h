#pragma once
#include "rpc/msgpack.hpp"

struct Sequence
{
    std::vector<int32_t> inputIds;
    int32_t outputLen;
    int32_t order_id;
    MSGPACK_DEFINE_ARRAY(inputIds, outputLen, order_id)
};
using Sequences = std::vector<Sequence>;