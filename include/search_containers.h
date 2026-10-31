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

        void clear()
        {
            std::memset(data, 0, nbytes);
        }

        bool get(int i)
        {
            return (data[i / block_size] >> (i & (block_size - 1))) & 1;
        }

        void *block_address(int i) { return data + i / block_size; }
    };

    template <typename dist_t = float>
    struct Candidiate
    {
        int id;
        dist_t distance;

        Candidiate() = default;
        Candidiate(int id, dist_t distance) : id(id), distance(distance) {}

        inline friend bool operator<(const Candidiate &lhs, const Candidiate &rhs)
        {
            return lhs.distance < rhs.distance ||
                   (lhs.distance == rhs.distance && lhs.id < rhs.id);
        }
        inline friend bool operator>(const Candidiate &lhs, const Candidiate &rhs)
        {
            return !(lhs < rhs);
        }
    };

    // Keep a distance-sorted bounded candidate array. The high bit of an ID
    // marks expansion; point IDs must fit in the remaining 31 bits.
    struct LinearPool
    {
    public:
        int nb, size_ = 0, cur_ = 0, capacity_;
        std::vector<Candidiate<float>, memory::align_alloc<Candidiate<float>>> data_;
        Bitset<uint64_t> vis;
        constexpr static int kMask = 2147483647;

        LinearPool(int n, int capacity)
            : nb(n), capacity_(capacity), data_(capacity_ + 1), vis(n) {}

        bool insert(int u, float dist)
        {
            if (size_ == capacity_ && dist >= data_[size_ - 1].distance)
            {
                return false;
            }
            int lo = find_bsearch(dist);
            std::memmove(&data_[lo + 1], &data_[lo],
                         (size_ - lo) * sizeof(Candidiate<float>));
            data_[lo] = {u, dist};
            if (size_ < capacity_)
            {
                ++size_;
            }
            if (lo < cur_)
            {
                cur_ = lo;
            }
            return true;
        }

        int pop()
        {
            set_checked(data_[cur_].id);
            int pre = cur_;
            while (cur_ < size_ && is_checked(data_[cur_].id))
            {
                ++cur_;
            }
            return get_id(data_[pre].id);
        }



        int id(int i) const { return get_id(data_[i].id); }

    private:
        int find_bsearch(float dist)
        {
            int lo = 0, hi = size_;
            while (lo < hi)
            {
                int mid = (lo + hi) / 2;
                if (data_[mid].distance > dist)
                {
                    hi = mid;
                }
                else
                {
                    lo = mid + 1;
                }
            }
            return lo;
        }

        int get_id(int id) const { return id & kMask; }
        void set_checked(int &id) { id |= 1 << 31; }
        int is_checked(int id) { return id >> 31 & 1; }
    };
}