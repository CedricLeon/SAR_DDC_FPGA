// test_rans_roundtrip.cpp
// Standalone unit test: encode a known sequence with RansEncoderCxx,
// then decode it with RansDecoderCxx, assert bit-exact round-trip.
//
// Build (without CMake):
//   g++ -O2 -std=c++17 -I../src/rans test_rans_roundtrip.cpp \
//       ../src/rans/rans_interface_cxx.cpp -o test_rans
//
// Exit code: 0 on success, 1 on failure.

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "rans_interface_cxx.hpp"

// ---------------------------------------------------------------------------
// Minimal CDF for a 5-symbol uniform distribution (matches CompressAI format)
// cdf[0] = 0, cdf[last] = 65536 (= 1 << 16)
// ---------------------------------------------------------------------------
static std::vector<int32_t> make_uniform_cdf(int n_symbols) {
    std::vector<int32_t> cdf(n_symbols + 1);
    for (int i = 0; i <= n_symbols; ++i)
        cdf[i] = static_cast<int32_t>((65536LL * i) / n_symbols);
    cdf[0]         = 0;
    cdf[n_symbols] = 65536;
    return cdf;
}

int main() {
    // Test 1 — simple uniform CDF round-trip
    {
        constexpr int N_SYMBOLS = 8;
        constexpr int N_CDFS    = 4;

        std::vector<std::vector<int32_t>> cdfs;
        for (int c = 0; c < N_CDFS; ++c)
            cdfs.push_back(make_uniform_cdf(N_SYMBOLS));

        std::vector<int32_t> cdf_sizes(N_CDFS, N_SYMBOLS + 1);
        std::vector<int32_t> offsets(N_CDFS, 0);

        // symbols: cycle through 0..N_SYMBOLS-1
        std::vector<int32_t> symbols;
        std::vector<int32_t> indexes;
        for (int i = 0; i < 64; ++i) {
            symbols.push_back(i % N_SYMBOLS);
            indexes.push_back(i % N_CDFS);
        }

        ddc::RansEncoderCxx enc;
        std::vector<uint8_t> bitstring =
            enc.encode_with_indexes(symbols, indexes, cdfs, cdf_sizes, offsets);

        assert(!bitstring.empty());

        ddc::RansDecoderCxx dec;
        std::vector<int32_t> decoded =
            dec.decode_with_indexes(bitstring, indexes, cdfs, cdf_sizes, offsets);

        assert(decoded.size() == symbols.size());
        for (size_t i = 0; i < symbols.size(); ++i) {
            if (decoded[i] != symbols[i]) {
                fprintf(stderr, "FAIL test1: index %zu: expected %d got %d\n",
                        i, symbols[i], decoded[i]);
                return EXIT_FAILURE;
            }
        }
        printf("PASS test1: uniform CDF round-trip (%zu symbols, %zu bytes)\n",
               symbols.size(), bitstring.size());
    }

    // Test 2 — set_stream / decode_stream interface
    {
        constexpr int N_SYMBOLS = 5;
        auto cdf = make_uniform_cdf(N_SYMBOLS);
        std::vector<std::vector<int32_t>> cdfs = {cdf};
        std::vector<int32_t> cdf_sizes = {N_SYMBOLS + 1};
        std::vector<int32_t> offsets   = {0};

        std::vector<int32_t> symbols  = {0, 2, 4, 1, 3, 0};
        std::vector<int32_t> indexes(symbols.size(), 0);

        ddc::RansEncoderCxx enc;
        auto bitstring = enc.encode_with_indexes(symbols, indexes, cdfs, cdf_sizes, offsets);

        ddc::RansDecoderCxx dec;
        dec.set_stream(bitstring);
        auto decoded = dec.decode_stream(indexes, cdfs, cdf_sizes, offsets);

        assert(decoded.size() == symbols.size());
        for (size_t i = 0; i < symbols.size(); ++i) {
            if (decoded[i] != symbols[i]) {
                fprintf(stderr, "FAIL test2: index %zu: expected %d got %d\n",
                        i, symbols[i], decoded[i]);
                return EXIT_FAILURE;
            }
        }
        printf("PASS test2: set_stream / decode_stream round-trip\n");
    }

    printf("All tests passed.\n");
    return EXIT_SUCCESS;
}
