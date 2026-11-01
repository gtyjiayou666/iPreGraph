#pragma once

#include <vector>
#include <stdint.h>
#include <stdlib.h>
#include <stdexcept>
#include "memory_utils.h"

namespace searcher
{
    template <typename Block = uint64_t>
    struct Bitset
    {
    private:
        constexpr static int block_size = sizeof(Block) * 8;
        size_t nbytes;
        static size_t bytes_for(int n)
        {
            if (n < 0)
                throw std::invalid_argument("bitset size must be non-negative");
            return (static_cast<size_t>(n) + block_size - 1) / block_size * sizeof(Block);
        }
        Block *data;

    public:
        explicit Bitset(int n)
            : nbytes(bytes_for(n)),
              data(static_cast<Block *>(memory::align_mm<64>(nbytes)))
        {
            std::memset(data, 0, nbytes);
        }

        ~Bitset() { memory::aligned_free(data); }

        void set(int i)
        {
            data[i / block_size] |= (Block(1) << (i & (block_size - 1)));
        }

        bool get(int i)
        {
            return (data[i / block_size] >> (i & (block_size - 1))) & 1;
        }
    };
}
