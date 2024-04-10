#pragma once
#include "rpc/msgpack.hpp"
#include <queue>
#include <optional>
struct Sequence
{
    std::vector<int32_t> inputIds;
    int32_t outputLen;
    int32_t order_id;
    std::optional<char> answer;
    std::optional<std::string> optionA;
    std::optional<std::string> optionB;
    std::optional<std::string> optionC;
    std::optional<std::string> optionD;
    MSGPACK_DEFINE_ARRAY(inputIds, outputLen, order_id, answer, 
        optionA, optionB, optionC, optionD)
};
using Sequences = std::vector<Sequence>;
using SeqQ = std::queue<Sequence>;
using SeqV = std::vector<Sequence>;
using outputTokenIds = std::vector<int32_t>; 
using ResultV = std::vector<outputTokenIds>;