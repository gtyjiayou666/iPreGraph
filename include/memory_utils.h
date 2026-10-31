#pragma once

#include <cstdlib>
#include <cstring>
#include <stdint.h>
#include <limits>
#include <new>
#include <type_traits>
#if defined(_WIN32)
#include <malloc.h>
#elif defined(__linux__)
#include <sys/mman.h>
#endif
#if defined(__SSE2__) || defined(_M_X64) || defined(_M_IX86)
#include <xmmintrin.h>
#endif


namespace memory
{
    // Windows aligned allocations must be paired with _aligned_free. Keep the
    // platform choice here so callers and allocators always use the same pair.
    inline void *aligned_allocate(size_t alignment, size_t nbytes)
    {
        if (nbytes > std::numeric_limits<size_t>::max() - (alignment - 1))
            throw std::bad_alloc();
        const size_t size = nbytes == 0 ? alignment :
            (nbytes + alignment - 1) / alignment * alignment;
#if defined(_WIN32)
        void *p = _aligned_malloc(size, alignment);
#else
        void *p = std::aligned_alloc(alignment, size);
#endif
        if (p == nullptr)
            throw std::bad_alloc();
        return p;
    }

    inline void aligned_free(void *pointer) noexcept
    {
#if defined(_WIN32)
        _aligned_free(pointer);
#else
        std::free(pointer);
#endif
    }

    template <uint32_t alignment>
    inline void *align_mm(size_t nbytes)
    {
        static_assert(alignment >= sizeof(void *) && (alignment & (alignment - 1)) == 0,
                      "alignment must be a power of two and at least pointer-sized");
        void *p = aligned_allocate(alignment, nbytes);
        std::memset(p, 0, nbytes);
        return p;
    }

    template <typename T>
    struct align_alloc
    {
        using value_type = T;
        using is_always_equal = std::true_type;

        align_alloc() noexcept = default;
        template <typename U>
        align_alloc(const align_alloc<U> &) noexcept {}

        T *allocate(size_t n)
        {
            if (n > std::numeric_limits<size_t>::max() / sizeof(T))
                throw std::bad_array_new_length();
            // Retain cache-line alignment for small arrays and huge-page
            // alignment for large arrays, while honoring over-aligned T.
            const size_t preferred_alignment = n <= (1u << 14) ? 64 : (1u << 21);
            const size_t alignment = preferred_alignment < alignof(T) ? alignof(T) : preferred_alignment;
            const size_t nbytes = n * sizeof(T);
            void *pointer = aligned_allocate(alignment, nbytes);
#if defined(MADV_HUGEPAGE)
            if (n > (1u << 14))
                madvise(pointer, (nbytes + alignment - 1) / alignment * alignment, MADV_HUGEPAGE);
#endif
            return static_cast<T *>(pointer);
        }

        // Stateless allocators can release blocks allocated by any equal copy.
        void deallocate(T *pointer, size_t) noexcept { aligned_free(pointer); }

        template <typename U>
        struct rebind
        {
            using other = align_alloc<U>;
        };
    };

    template <typename T, typename U>
    inline bool operator==(const align_alloc<T> &, const align_alloc<U> &) noexcept { return true; }

    template <typename T, typename U>
    inline bool operator!=(const align_alloc<T> &, const align_alloc<U> &) noexcept { return false; }

    inline void prefetch_L1(const void *address)
    {
#if defined(__SSE2__) || defined(_M_X64) || defined(_M_IX86)
        _mm_prefetch((const char *)address, _MM_HINT_T0);
#elif defined(__GNUC__) || defined(__clang__)
        __builtin_prefetch(address, 0, 3);
#else
        (void)address;
#endif
    }


    inline void mem_prefetch_L1(char *ptr, const int num_lines)
    {
        switch (num_lines)
        {
        default:
            [[fallthrough]];
        case 28:
            prefetch_L1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 27:
            prefetch_L1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 26:
            prefetch_L1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 25:
            prefetch_L1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 24:
            prefetch_L1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 23:
            prefetch_L1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 22:
            prefetch_L1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 21:
            prefetch_L1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 20:
            prefetch_L1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 19:
            prefetch_L1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 18:
            prefetch_L1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 17:
            prefetch_L1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 16:
            prefetch_L1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 15:
            prefetch_L1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 14:
            prefetch_L1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 13:
            prefetch_L1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 12:
            prefetch_L1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 11:
            prefetch_L1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 10:
            prefetch_L1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 9:
            prefetch_L1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 8:
            prefetch_L1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 7:
            prefetch_L1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 6:
            prefetch_L1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 5:
            prefetch_L1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 4:
            prefetch_L1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 3:
            prefetch_L1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 2:
            prefetch_L1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 1:
            prefetch_L1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 0:
            break;
        }
    }
}