#include <Windows.h>
#pragma warning(disable: 4512 4244 4100)
#include "avisynth.h"
#pragma warning(default: 4512 4244 4100)
#include <emmintrin.h>

#ifdef __INTEL_COMPILER
#define SG_FORCEINLINE inline
#else
#define SG_FORCEINLINE __forceinline
#endif

#define USE_MOVPS

extern "C" {

    SG_FORCEINLINE __m128i simd_load_si128(const BYTE *ptr) {
#ifdef USE_MOVPS
        return _mm_castps_si128(_mm_load_ps(reinterpret_cast<const float*>(ptr)));
#else
        return _mm_load_si128(reinterpret_cast<const __m128i*>(ptr));
#endif
    }

    SG_FORCEINLINE __m128i simd_loadu_si128(const BYTE *ptr) {
#ifdef USE_MOVPS
        return _mm_castps_si128(_mm_loadu_ps(reinterpret_cast<const float*>(ptr)));
#else
        return _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr));
#endif
    }

    SG_FORCEINLINE void simd_store_si128(BYTE *ptr, __m128i value) {
#ifdef USE_MOVPS
        _mm_store_ps(reinterpret_cast<float*>(ptr), _mm_castsi128_ps(value));
#else
        _mm_store_si128(reinterpret_cast<__m128i*>(ptr), value);
#endif
    }

    SG_FORCEINLINE void simd_storeu_si128(BYTE *ptr, __m128i value) {
#ifdef USE_MOVPS
        _mm_storeu_ps(reinterpret_cast<float*>(ptr), _mm_castsi128_ps(value));
#else
        _mm_storeu_si128(reinterpret_cast<__m128i*>(ptr), value);
#endif
    }
}

#pragma warning(disable: 4309)

template<bool isBorder, decltype(simd_load_si128) simd_load>
static SG_FORCEINLINE __m128i simd_load_one_to_left(const BYTE *ptr) {
    if (isBorder) {
        auto mask = _mm_setr_epi8(0xFF, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00);
        auto val = simd_load(ptr);
        auto shifted = _mm_slli_si128(val, 1);
        auto andm = _mm_and_si128(val, mask);
        return _mm_or_si128(shifted, andm);
    } else {
        return simd_loadu_si128(ptr - 1);
    }
}

template<bool isBorder, decltype(simd_load_si128) simd_load>
static SG_FORCEINLINE __m128i simd_load_two_to_left(const BYTE *ptr) {
    if (isBorder) {
        auto mask = _mm_setr_epi8(0xFF, 0xFF, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00);
        auto val = simd_load(ptr);
        auto shifted = _mm_slli_si128(val, 2);
        auto unpck = _mm_unpacklo_epi8(val, val);
        auto andm = _mm_and_si128(unpck, mask);
        return _mm_or_si128(shifted, andm);
    } else {
        return simd_loadu_si128(ptr - 2);
    }
}

template<bool isBorder, decltype(simd_load_si128) simd_load>
static SG_FORCEINLINE __m128i simd_load_three_to_left(const BYTE *ptr) {
    if (isBorder) {
        auto mask = _mm_setr_epi8(0xFF, 0xFF, 0xFF, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00);
        auto val = simd_load(ptr);
        auto shifted = _mm_slli_si128(val, 3);
        auto unpck = _mm_unpacklo_epi8(val, val);
        unpck = _mm_unpacklo_epi16(unpck, unpck);
        auto andm = _mm_and_si128(unpck, mask);
        return _mm_or_si128(shifted, andm);
    } else {
        return simd_loadu_si128(ptr - 3);
    }
}

template<bool isBorder, decltype(simd_load_si128) simd_load>
static SG_FORCEINLINE __m128i simd_load_one_epi16_to_left(const BYTE *ptr) {
    if (isBorder) {
        auto mask = _mm_setr_epi16(0xFFFF, 00, 00, 00, 00, 00, 00, 00);
        auto val = simd_load(ptr);
        auto shifted = _mm_slli_si128(val, 2);
        auto unpck = _mm_unpacklo_epi16(val, val);
        auto andm = _mm_and_si128(unpck, mask);
        return _mm_or_si128(shifted, andm);
    } else {
        return simd_loadu_si128(ptr - 2);
    }
}

template<bool isBorder, decltype(simd_load_si128) simd_load>
static SG_FORCEINLINE __m128i simd_load_two_epi16_to_left(const BYTE *ptr) {
    if (isBorder) {
        auto mask = _mm_setr_epi16(0xFFFF, 0xFFFF, 00, 00, 00, 00, 00, 00);
        auto val = simd_load(ptr);
        auto shifted = _mm_slli_si128(val, 4);
        auto unpck = _mm_unpacklo_epi16(val, val);
        auto andm = _mm_and_si128(unpck, mask);
        return _mm_or_si128(shifted, andm);
    } else {
        return simd_loadu_si128(ptr - 4);
    }
}

template<bool isBorder, decltype(simd_load_si128) simd_load>
static SG_FORCEINLINE __m128i simd_load_three_epi16_to_left(const BYTE *ptr) {
    if (isBorder) {
        auto mask = _mm_setr_epi16(0xFFFF, 0xFFFF, 0xFFFF, 00, 00, 00, 00, 00);
        auto val = simd_load(ptr);
        auto shifted = _mm_slli_si128(val, 6);
        auto unpck = _mm_unpacklo_epi16(val, val);
        unpck = _mm_unpacklo_epi32(unpck, unpck);
        auto andm = _mm_and_si128(unpck, mask);
        return _mm_or_si128(shifted, andm);
    } else {
        return simd_loadu_si128(ptr - 6);
    }
}

//note the difference between set and setr for left and right loading
template<bool isBorder, decltype(simd_load_si128) simd_load>
static SG_FORCEINLINE __m128i simd_load_one_to_right(const BYTE *ptr) {
    if (isBorder) {
        auto mask = _mm_set_epi8(0xFF, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00);
        auto val = simd_load(ptr);
        auto shifted = _mm_srli_si128(val, 1);
        auto andm = _mm_and_si128(val, mask);
        return _mm_or_si128(shifted, andm);
    } else {
        return simd_loadu_si128(ptr + 1);
    }
}

template<bool isBorder, decltype(simd_load_si128) simd_load>
static SG_FORCEINLINE __m128i simd_load_two_to_right(const BYTE *ptr) {
    if (isBorder) {
        auto mask = _mm_set_epi8(0xFF, 0xFF, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00);
        auto val = simd_load(ptr);
        auto shifted = _mm_srli_si128(val, 2);
        auto unpck = _mm_unpackhi_epi8(val, val);
        auto andm = _mm_and_si128(unpck, mask);
        return _mm_or_si128(shifted, andm);
    } else {
        return simd_loadu_si128(ptr + 2);
    }
}

template<bool isBorder, decltype(simd_load_si128) simd_load>
static SG_FORCEINLINE __m128i simd_load_three_to_right(const BYTE *ptr) {
    if (isBorder) {
        auto mask = _mm_set_epi8(0xFF, 0xFF, 0xFF, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00);
        auto val = simd_load(ptr);
        auto shifted = _mm_srli_si128(val, 3);
        auto unpck = _mm_unpackhi_epi8(val, val);
        unpck = _mm_unpackhi_epi16(unpck, unpck);
        auto andm = _mm_and_si128(unpck, mask);
        return _mm_or_si128(shifted, andm);
    } else {
        return simd_loadu_si128(ptr + 3);
    }
}

template<bool isBorder, decltype(simd_load_si128) simd_load>
static SG_FORCEINLINE __m128i simd_load_one_epi16_to_right(const BYTE *ptr) {
    if (isBorder) {
        auto mask = _mm_set_epi16(0xFFFF, 00, 00, 00, 00, 00, 00, 00);
        auto val = simd_load(ptr);
        auto shifted = _mm_srli_si128(val, 2);
        auto unpck = _mm_unpackhi_epi16(val, val);
        auto andm = _mm_and_si128(unpck, mask);
        return _mm_or_si128(shifted, andm);
    } else {
        return simd_loadu_si128(ptr + 2);
    }
}


template<bool isBorder, decltype(simd_load_si128) simd_load>
static SG_FORCEINLINE __m128i simd_load_two_epi16_to_right(const BYTE *ptr) {
    if (isBorder) {
        auto mask = _mm_set_epi16(0xFFFF, 0xFFFF, 00, 00, 00, 00, 00, 00);
        auto val = simd_load(ptr);
        auto shifted = _mm_srli_si128(val, 4);
        auto unpck = _mm_unpackhi_epi16(val, val);
        auto andm = _mm_and_si128(unpck, mask);
        return _mm_or_si128(shifted, andm);
    } else {
        return simd_loadu_si128(ptr + 4);
    }
}

template<bool isBorder, decltype(simd_load_si128) simd_load>
static SG_FORCEINLINE __m128i simd_load_three_epi16_to_right(const BYTE *ptr) {
    if (isBorder) {
        auto mask = _mm_set_epi16(0xFFFF, 0xFFFF, 0xFFFF, 00, 00, 00, 00, 00);
        auto val = simd_load(ptr);
        auto shifted = _mm_srli_si128(val, 6);
        auto unpck = _mm_unpackhi_epi16(val, val);
        unpck = _mm_unpackhi_epi32(unpck, unpck);
        auto andm = _mm_and_si128(unpck, mask);
        return _mm_or_si128(shifted, andm);
    } else {
        return simd_loadu_si128(ptr + 6);
    }
}
#pragma warning(default: 4309)

enum Buffers {
    ADIFF_M3_P3 = 0,
    ADIFF_M2_P2 = 1,
    ADIFF_M1_P1 = 2,
    ADIFF_P0_M0 = 4,
    ADIFF_P1_M1 = 6,
    ADIFF_P2_M2 = 7,
    ADIFF_P3_M3 = 8,

    SG_FORWARD = 3,
    SG_REVERSE = 5
};

enum class BorderMode {
    LEFT,
    RIGHT,
    NONE
};

const unsigned int BUFFERS_COUNT = 9;


static SG_FORCEINLINE __m128i simd_abs_diff_epu8(__m128i a, __m128i b) {
    auto positive = _mm_subs_epu8(a, b);
    auto negative = _mm_subs_epu8(b, a);
    return _mm_or_si128(positive, negative);
}

static SG_FORCEINLINE __m128i calculateSangnom(const __m128i& p1, const __m128i& p2, const __m128i& p3) {
    auto zero = _mm_setzero_si128();

    auto temp_lo = _mm_unpacklo_epi8(p1, zero);
    auto temp_hi = _mm_unpackhi_epi8(p1, zero);

    temp_lo = _mm_slli_epi16(temp_lo, 2); //p1*4
    temp_hi = _mm_slli_epi16(temp_hi, 2);

    auto t2_lo = _mm_unpacklo_epi8(p2, zero);
    auto t2_hi = _mm_unpackhi_epi8(p2, zero);

    temp_lo = _mm_adds_epu16(temp_lo, t2_lo); //p1*4 + p2
    temp_hi = _mm_adds_epu16(temp_hi, t2_hi);

    t2_lo = _mm_slli_epi16(t2_lo, 2); 
    t2_hi = _mm_slli_epi16(t2_hi, 2);

    temp_lo = _mm_adds_epu16(temp_lo, t2_lo); //p1*4 + p2*4 + p2 = p1*4 + p2*5
    temp_hi = _mm_adds_epu16(temp_hi, t2_hi);

    auto t3_lo = _mm_unpacklo_epi8(p3, zero);
    auto t3_hi = _mm_unpackhi_epi8(p3, zero);

    temp_lo = _mm_subs_epu16(temp_lo, t3_lo); //p1*4 + p2*5 - p3
    temp_hi = _mm_subs_epu16(temp_hi, t3_hi);

    temp_lo = _mm_srli_epi16(temp_lo, 3); //(p1*4 + p2*5 - p3) / 8
    temp_hi = _mm_srli_epi16(temp_hi, 3);

    return _mm_packus_epi16(temp_lo, temp_hi); //(p1*4 + p2*5 - p3) / 8
}


static SG_FORCEINLINE __m128i blendAvgOnMinimalBuffer(const __m128i& a1, const __m128i& a2, const __m128i& buf, 
                                              const __m128i& minv, const __m128i& acc, const __m128i& zero) {
    auto average = _mm_avg_epu8(a1, a2);
    //buffer is minimal
    auto mask = _mm_cmpeq_epi8(buf, minv);
    //blend
    auto avgPart = _mm_and_si128(mask, average);
    auto accPart = _mm_andnot_si128(mask, acc);
    return _mm_or_si128(avgPart, accPart);   
}

template<BorderMode border, decltype(simd_load_si128) simd_load, decltype(simd_store_si128) simd_store>
static SG_FORCEINLINE void prepareBuffersLine(const BYTE* pSrc, const BYTE *pSrcn2, BYTE* pBuffers[BUFFERS_COUNT], int bufferOffset, int width) {
    for (int x = 0; x < width; x += 16) {
        auto cur_minus_3   = simd_load_three_to_left<border == BorderMode::LEFT, simd_load>(pSrc+x); 
        auto cur_minus_2   = simd_load_two_to_left<border == BorderMode::LEFT, simd_load>(pSrc+x); 
        auto cur_minus_1   = simd_load_one_to_left<border == BorderMode::LEFT, simd_load>(pSrc+x); 
        auto cur           = simd_load(pSrc+x);
        auto cur_plus_1    = simd_load_one_to_right<border == BorderMode::RIGHT, simd_load>(pSrc+x); 
        auto cur_plus_2    = simd_load_two_to_right<border == BorderMode::RIGHT, simd_load>(pSrc+x); 
        auto cur_plus_3    = simd_load_three_to_right<border == BorderMode::RIGHT, simd_load>(pSrc+x); 

        auto next_minus_3  = simd_load_three_to_left<border == BorderMode::LEFT, simd_load>(pSrcn2+x); 
        auto next_minus_2  = simd_load_two_to_left<border == BorderMode::LEFT, simd_load>(pSrcn2+x); 
        auto next_minus_1  = simd_load_one_to_left<border == BorderMode::LEFT, simd_load>(pSrcn2+x); 
        auto next          = simd_load(pSrcn2+x); 
        auto next_plus_1   = simd_load_one_to_right<border == BorderMode::RIGHT, simd_load>(pSrcn2+x); 
        auto next_plus_2   = simd_load_two_to_right<border == BorderMode::RIGHT, simd_load>(pSrcn2+x); 
        auto next_plus_3   = simd_load_three_to_right<border == BorderMode::RIGHT, simd_load>(pSrcn2+x); 

        auto adiff_m3_p3 = simd_abs_diff_epu8(cur_minus_3, next_plus_3);
        simd_store(pBuffers[ADIFF_M3_P3]+bufferOffset+x, adiff_m3_p3);

        auto adiff_m2_p2 = simd_abs_diff_epu8(cur_minus_2, next_plus_2);
        simd_store(pBuffers[ADIFF_M2_P2]+bufferOffset+x, adiff_m2_p2);

        auto adiff_m1_p1 = simd_abs_diff_epu8(cur_minus_1, next_plus_1);
        simd_store(pBuffers[ADIFF_M1_P1]+bufferOffset+x, adiff_m1_p1);

        auto adiff_0     = simd_abs_diff_epu8(cur, next);
        simd_store(pBuffers[ADIFF_P0_M0]+bufferOffset+x, adiff_0);

        auto adiff_p1_m1 = simd_abs_diff_epu8(cur_plus_1, next_minus_1);
        simd_store(pBuffers[ADIFF_P1_M1]+bufferOffset+x, adiff_p1_m1);

        auto adiff_p2_m2 = simd_abs_diff_epu8(cur_plus_2, next_minus_2);
        simd_store(pBuffers[ADIFF_P2_M2]+bufferOffset+x, adiff_p2_m2);

        auto adiff_p3_m3 = simd_abs_diff_epu8(cur_plus_3, next_minus_3);
        simd_store(pBuffers[ADIFF_P3_M3]+bufferOffset+x, adiff_p3_m3);

        //////////////////////////////////////////////////////////////////////////
        auto temp1 = calculateSangnom(cur_minus_1, cur, cur_plus_1);
        auto temp2 = calculateSangnom(next_plus_1, next, next_minus_1);

        //abs((cur_minus_1*4 + cur*5 - cur_plus_1) / 8  - (next_plus_1*4 + next*5 - next_minus_1) / 8)
        auto absdiff_p1_p2 = simd_abs_diff_epu8(temp1, temp2); 
        simd_store(pBuffers[SG_FORWARD]+bufferOffset+x, absdiff_p1_p2);
        //////////////////////////////////////////////////////////////////////////
        auto temp3 = calculateSangnom(cur_plus_1, cur, cur_minus_1);
        auto temp4 = calculateSangnom(next_minus_1, next, next_plus_1);

        //abs((cur_plus_1*4 + cur*5 - cur_minus_1) / 8  - (next_minus_1*4 + next*5 - next_plus_1) / 8)
        auto absdiff_p3_p4 = simd_abs_diff_epu8(temp3, temp4);
        simd_store(pBuffers[SG_REVERSE]+bufferOffset+x, absdiff_p3_p4);
        //////////////////////////////////////////////////////////////////////////
    }
}


template<decltype(simd_load_si128) simd_load>
static void prepareBuffers(const BYTE* pSrc, BYTE* pBuffers[BUFFERS_COUNT], int width, int height, int srcPitch, int bufferPitch) {
    auto pSrcn2 = pSrc + srcPitch*2;

    int bufferOffset = bufferPitch;
    int sse2Width = (width - 1 - 16) / 16 * 16 + 16;

    for (int y = 0; y < height / 2 - 1; y++) {
        prepareBuffersLine<BorderMode::LEFT, simd_load, simd_store_si128>(pSrc, pSrcn2, pBuffers, bufferOffset, 16);

        prepareBuffersLine<BorderMode::NONE, simd_load, simd_store_si128>(pSrc + 16, pSrcn2+16, pBuffers, bufferOffset+16, sse2Width - 16);
        
        prepareBuffersLine<BorderMode::RIGHT, simd_loadu_si128, simd_storeu_si128>(pSrc + width - 16, pSrcn2 + width - 16, pBuffers, bufferOffset + width - 16, 16);
      
        pSrc += srcPitch*2;
        pSrcn2 += srcPitch*2;
        bufferOffset += bufferPitch;
    }
}

template<BorderMode border>
static SG_FORCEINLINE void finalizeBufferProcessingBlock(BYTE* pTemp, BYTE* pSrcn, int x) {
    auto cur_minus_6_lo = simd_load_three_epi16_to_left<border == BorderMode::LEFT, simd_load_si128>(pTemp+x*2);
    auto cur_minus_4_lo = simd_load_two_epi16_to_left<border == BorderMode::LEFT, simd_load_si128>(pTemp+x*2);
    auto cur_minus_2_lo = simd_load_one_epi16_to_left<border == BorderMode::LEFT, simd_load_si128>(pTemp+x*2);
    auto cur_lo         = simd_load_si128(pTemp+x*2);
    auto cur_plus_2_lo  = simd_load_one_epi16_to_right<false, simd_load_si128>(pTemp+x*2);
    auto cur_plus_4_lo  = simd_load_two_epi16_to_right<false, simd_load_si128>(pTemp+x*2);
    auto cur_plus_6_lo  = simd_load_three_epi16_to_right<false, simd_load_si128>(pTemp+x*2);

    auto cur_minus_6_hi = simd_load_three_epi16_to_left<false, simd_load_si128>(pTemp+x*2+16);
    auto cur_minus_4_hi = simd_load_two_epi16_to_left<false, simd_load_si128>(pTemp+x*2+16);
    auto cur_minus_2_hi = simd_load_one_epi16_to_left<false, simd_load_si128>(pTemp+x*2+16);
    auto cur_hi         = simd_load_si128(pTemp+x*2+16);
    auto cur_plus_2_hi  = simd_load_one_epi16_to_right<border == BorderMode::RIGHT, simd_load_si128>(pTemp+x*2+16);
    auto cur_plus_4_hi  = simd_load_two_epi16_to_right<border == BorderMode::RIGHT, simd_load_si128>(pTemp+x*2+16);
    auto cur_plus_6_hi  = simd_load_three_epi16_to_right<border == BorderMode::RIGHT, simd_load_si128>(pTemp+x*2+16);

    auto sum_lo = _mm_adds_epu16(cur_minus_6_lo, cur_minus_4_lo);
    sum_lo = _mm_adds_epu16(sum_lo, cur_minus_2_lo);
    sum_lo = _mm_adds_epu16(sum_lo, cur_lo);
    sum_lo = _mm_adds_epu16(sum_lo, cur_plus_2_lo);
    sum_lo = _mm_adds_epu16(sum_lo, cur_plus_4_lo);
    sum_lo = _mm_adds_epu16(sum_lo, cur_plus_6_lo);

    sum_lo = _mm_srli_epi16(sum_lo, 4);

    auto sum_hi = _mm_adds_epu16(cur_minus_6_hi, cur_minus_4_hi);
    sum_hi = _mm_adds_epu16(sum_hi, cur_minus_2_hi);
    sum_hi = _mm_adds_epu16(sum_hi, cur_hi);
    sum_hi = _mm_adds_epu16(sum_hi, cur_plus_2_hi);
    sum_hi = _mm_adds_epu16(sum_hi, cur_plus_4_hi);
    sum_hi = _mm_adds_epu16(sum_hi, cur_plus_6_hi);

    sum_hi = _mm_srli_epi16(sum_hi, 4);

    auto result = _mm_packus_epi16(sum_lo, sum_hi);
    simd_store_si128(pSrcn+x, result);
}

static void processBuffers(BYTE* pBuffers[BUFFERS_COUNT], BYTE* pTemp, int pitch, int height) {
    for (int i = 0; i < BUFFERS_COUNT; ++i) {
        auto pSrc = pBuffers[i];
        auto pSrcn = pSrc + pitch;
        auto pSrcn2 = pSrcn + pitch;
        
        for (int y = 0; y < height - 1; ++y) {
            auto zero = _mm_setzero_si128();
            for(int x = 0; x < pitch; x+= 16) {
                auto src = simd_load_si128(pSrc+x);
                auto srcn = simd_load_si128(pSrcn+x);
                auto srcn2 = simd_load_si128(pSrcn2+x);

                auto src_lo = _mm_unpacklo_epi8(src, zero);
                auto srcn_lo = _mm_unpacklo_epi8(srcn, zero);
                auto srcn2_lo = _mm_unpacklo_epi8(srcn2, zero);

                auto src_hi     = _mm_unpackhi_epi8(src, zero);
                auto srcn_hi    = _mm_unpackhi_epi8(srcn, zero);
                auto srcn2_hi   = _mm_unpackhi_epi8(srcn2, zero);

                auto sum_lo = _mm_adds_epu16(src_lo, srcn_lo);
                sum_lo = _mm_adds_epu16(sum_lo, srcn2_lo);

                auto sum_hi = _mm_adds_epu16(src_hi, srcn_hi);
                sum_hi = _mm_adds_epu16(sum_hi, srcn2_hi);

                simd_store_si128(pTemp+(x*2), sum_lo);
                simd_store_si128(pTemp+(x*2)+16, sum_hi);
            }

            finalizeBufferProcessingBlock<BorderMode::LEFT>(pTemp, pSrcn, 0);

            for (int x = 16; x < pitch-16; x+= 16) {
                finalizeBufferProcessingBlock<BorderMode::NONE>(pTemp, pSrcn, x);
            }

            finalizeBufferProcessingBlock<BorderMode::RIGHT>(pTemp, pSrcn, pitch-16);

            pSrc += pitch;
            pSrcn += pitch;
            pSrcn2 += pitch;
        }
    }
}

template<BorderMode border, decltype(simd_load_si128) simd_load, decltype(simd_load_si128) simd_load_buffer, decltype(simd_store_si128) simd_store>
static SG_FORCEINLINE void finalizePlaneLine(const BYTE* pSrc, const BYTE* pSrcn2, BYTE* pDstn, BYTE* pBuffers[BUFFERS_COUNT], int bufferOffset, int width, const __m128i& aath) {
    auto zero = _mm_setzero_si128();
    for (int x = 0; x < width; x += 16) {
        auto buf0 = simd_load_buffer(pBuffers[ADIFF_M3_P3] + bufferOffset + x); 
        auto buf1 = simd_load_buffer(pBuffers[ADIFF_M2_P2] + bufferOffset + x); 
        auto buf2 = simd_load_buffer(pBuffers[ADIFF_M1_P1] + bufferOffset + x); 
        auto buf3 = simd_load_buffer(pBuffers[SG_FORWARD]  + bufferOffset + x); 
        auto buf4 = simd_load_buffer(pBuffers[ADIFF_P0_M0] + bufferOffset + x); 
        auto buf5 = simd_load_buffer(pBuffers[SG_REVERSE]  + bufferOffset + x); 
        auto buf6 = simd_load_buffer(pBuffers[ADIFF_P1_M1] + bufferOffset + x); 
        auto buf7 = simd_load_buffer(pBuffers[ADIFF_P2_M2] + bufferOffset + x); 
        auto buf8 = simd_load_buffer(pBuffers[ADIFF_P3_M3] + bufferOffset + x); 

        auto cur_minus_3   = simd_load_three_to_left<border == BorderMode::LEFT, simd_load>(pSrc+x); 
        auto cur_minus_2   = simd_load_two_to_left<border == BorderMode::LEFT, simd_load>(pSrc+x); 
        auto cur_minus_1   = simd_load_one_to_left<border == BorderMode::LEFT, simd_load>(pSrc+x); 
        auto cur           = simd_load(pSrc+x); 
        auto cur_plus_1    = simd_load_one_to_right<border == BorderMode::RIGHT, simd_load>(pSrc+x); 
        auto cur_plus_2    = simd_load_two_to_right<border == BorderMode::RIGHT, simd_load>(pSrc+x); 
        auto cur_plus_3    = simd_load_three_to_right<border == BorderMode::RIGHT, simd_load>(pSrc+x); 

        auto next_minus_3  = simd_load_three_to_left<border == BorderMode::LEFT, simd_load>(pSrcn2+x); 
        auto next_minus_2  = simd_load_two_to_left<border == BorderMode::LEFT, simd_load>(pSrcn2+x); 
        auto next_minus_1  = simd_load_one_to_left<border == BorderMode::LEFT, simd_load>(pSrcn2+x); 
        auto next          = simd_load(pSrcn2+x); 
        auto next_plus_1   = simd_load_one_to_right<border == BorderMode::RIGHT, simd_load>(pSrcn2+x); 
        auto next_plus_2   = simd_load_two_to_right<border == BorderMode::RIGHT, simd_load>(pSrcn2+x); 
        auto next_plus_3   = simd_load_three_to_right<border == BorderMode::RIGHT, simd_load>(pSrcn2+x); 

        auto minbuf = _mm_min_epu8(buf0, buf1);
        minbuf = _mm_min_epu8(minbuf, buf2);
        minbuf = _mm_min_epu8(minbuf, buf3);
        minbuf = _mm_min_epu8(minbuf, buf4);
        minbuf = _mm_min_epu8(minbuf, buf5);
        minbuf = _mm_min_epu8(minbuf, buf6);
        minbuf = _mm_min_epu8(minbuf, buf7);
        minbuf = _mm_min_epu8(minbuf, buf8);

        auto processed = _mm_setzero_si128();

        processed = blendAvgOnMinimalBuffer(cur_minus_3, next_plus_3, buf0, minbuf, processed, zero);
        processed = blendAvgOnMinimalBuffer(cur_plus_3, next_minus_3, buf8, minbuf, processed, zero);

        processed = blendAvgOnMinimalBuffer(cur_minus_2, next_plus_2, buf1, minbuf, processed, zero);
        processed = blendAvgOnMinimalBuffer(cur_plus_2, next_minus_2, buf7, minbuf, processed, zero);

        processed = blendAvgOnMinimalBuffer(cur_minus_1, next_plus_1, buf2, minbuf, processed, zero);
        processed = blendAvgOnMinimalBuffer(cur_plus_1, next_minus_1, buf6, minbuf, processed, zero);

        ////////////////////////////////////////////////////////////////////////////
        auto temp1 = calculateSangnom(cur_minus_1, cur, cur_plus_1);
        auto temp2 = calculateSangnom(next_plus_1, next, next_minus_1);

        processed = blendAvgOnMinimalBuffer(temp1, temp2, buf3, minbuf, processed, zero);
        ////////////////////////////////////////////////////////////////////////////
        auto temp3 = calculateSangnom(cur_plus_1, cur, cur_minus_1);
        auto temp4 = calculateSangnom(next_minus_1, next, next_plus_1);

        processed = blendAvgOnMinimalBuffer(temp3, temp4, buf5, minbuf, processed, zero);
        ////////////////////////////////////////////////////////////////////////////

        auto average = _mm_avg_epu8(cur, next);

        auto buf4IsMinimal = _mm_cmpeq_epi8(buf4, minbuf);
        
        auto takeAaa = _mm_subs_epu8(minbuf, aath);
        //this isn't strictly negation, don't optimize
        auto takeProcessed = _mm_cmpeq_epi8(takeAaa, zero);
        auto mask = _mm_andnot_si128(buf4IsMinimal, takeProcessed);

        //blending
        processed = _mm_and_si128(mask, processed);
        average = _mm_andnot_si128(mask, average);
        auto result = _mm_or_si128(processed, average);

        simd_store(pDstn+x, result);
    }
}

template<decltype(simd_load_si128) simd_load, decltype(simd_store_si128) simd_store>
static void finalizePlane(const BYTE* pSrc, BYTE* pDst, BYTE* pBuffers[BUFFERS_COUNT], int srcPitch, int dstPitch, int bufferPitch, int width, int height, int aa) {
    auto pDstn = pDst + dstPitch;
    auto pSrcn2 = pSrc + srcPitch*2;
    auto aav = _mm_set1_epi8(aa);
    int bufferOffset = bufferPitch;
    int sse2Width = (width - 1 - 16) / 16 * 16 + 16;

    for (int y = 0; y < height / 2 - 1; ++y) {
        finalizePlaneLine<BorderMode::LEFT, simd_load, simd_load_si128, simd_store>(pSrc, pSrcn2, pDstn, pBuffers, bufferOffset, 16, aav);

        finalizePlaneLine<BorderMode::NONE, simd_load, simd_load_si128, simd_store>(pSrc + 16, pSrcn2+16, pDstn+16, pBuffers, bufferOffset+16, sse2Width - 16, aav);

        finalizePlaneLine<BorderMode::RIGHT, simd_loadu_si128, simd_loadu_si128, simd_storeu_si128>(pSrc + width - 16, pSrcn2 + width - 16, pDstn + width - 16, pBuffers, bufferOffset + width - 16, 16, aav);

        pSrc += srcPitch * 2;
        pSrcn2 += srcPitch * 2;
        pDstn += dstPitch *2;
        bufferOffset += bufferPitch;
    }
}

inline bool is16byteAligned(const void *ptr) {
    return (((unsigned long)ptr) & 15) == 0;
}

auto prepareBuffers_sse2 = &prepareBuffers<simd_loadu_si128>;
auto prepareBuffers_asse2 = &prepareBuffers<simd_load_si128>;
auto finalizePlane_sse2 = &finalizePlane<simd_loadu_si128, simd_storeu_si128>;
auto finalizePlane_asse2 = &finalizePlane<simd_load_si128, simd_store_si128>;


class SangNom2 : public GenericVideoFilter {
public:
    SangNom2(PClip child, int order, int aa, IScriptEnvironment* env);

    PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env);

    ~SangNom2() {
        for (int i = 0; i < BUFFERS_COUNT; i++) {
            _mm_free(buffers_[i]);
        }
        _mm_free(intermediate_);
    }

private:
    int order_;
    int offset_;
    int aa_;

    BYTE *buffers_[BUFFERS_COUNT];
    int bufferPitch_;
    int bufferHeight_;
    BYTE *intermediate_;

    void processPlane(IScriptEnvironment* env, const BYTE* srcp, BYTE* dstp, int width, int height, int src_pitch, int dst_pitch, int aa);
};

SangNom2::SangNom2(PClip child, int order, int aa, IScriptEnvironment* env)
    : GenericVideoFilter(child), order_(order) {
        if(!vi.IsPlanar()) {
            env->ThrowError("SangNom2 works only with planar colorspaces");
        }

        if (!(env->GetCPUFlags() && CPUF_SSE2)) {
            env->ThrowError("Sorry, SSE2 is requried");
        }

        if (vi.width < 16) {
            env->ThrowError("Sorry, wight must be bigger or equal to 16");
        }

        bufferPitch_ = (vi.width + 15) / 16 * 16;
        bufferHeight_ = (vi.height + 1) / 2;
        for (int i = 0; i < BUFFERS_COUNT; i++) {
            buffers_[i] = reinterpret_cast<BYTE*>(_mm_malloc(bufferPitch_ * (bufferHeight_+1), 16));
            memset(buffers_[i], 0, bufferPitch_); //this is important... I think
        }
        intermediate_ = reinterpret_cast<BYTE*>(_mm_malloc(bufferPitch_*2, 16));
        
        aa = min(128, aa);
        aa_ = (21 * aa) / 16;
}

void SangNom2::processPlane(IScriptEnvironment* env, const BYTE* pSrc, BYTE* pDst, int width, int height, int srcPitch, int dstPitch, int aa) {
    env->BitBlt(pDst + offset_ * dstPitch, dstPitch * 2, pSrc + offset_ * srcPitch, srcPitch * 2, width, height / 2);
    
    if (offset_ == 1) {
        env->BitBlt(pDst, dstPitch, pSrc + srcPitch, srcPitch, width,1);
    } else {
        env->BitBlt(pDst+dstPitch * (height-1), dstPitch, pSrc + srcPitch*(height-2), srcPitch, width,1);
    }

    auto prepareBuffers_op = prepareBuffers_sse2;
    auto finalizePlane_op = finalizePlane_sse2;
    if (is16byteAligned(pSrc)) {
        prepareBuffers_op = prepareBuffers_asse2;
        finalizePlane_op = finalizePlane_asse2;
    }

    prepareBuffers_op(pSrc + offset_*srcPitch, buffers_, width, height, srcPitch, bufferPitch_);
    processBuffers(buffers_, intermediate_, bufferPitch_, bufferHeight_);
    finalizePlane_op(pSrc + offset_ * srcPitch, pDst + offset_ * dstPitch, buffers_, srcPitch, dstPitch, bufferPitch_, width, height, aa);
}


PVideoFrame SangNom2::GetFrame(int n, IScriptEnvironment* env) {
    auto srcFrame = child->GetFrame(n, env);
    auto dstFrame = env->NewVideoFrame(vi, 16);

    offset_ = order_ == 0 
        ? child->GetParity(n) ? 0 : 1
        : order_ == 1 ? 0 : 1;

    processPlane(env, srcFrame->GetReadPtr(PLANAR_Y), dstFrame->GetWritePtr(PLANAR_Y), srcFrame->GetRowSize(PLANAR_Y), 
        srcFrame->GetHeight(PLANAR_Y), srcFrame->GetPitch(PLANAR_Y), srcFrame->GetPitch(PLANAR_Y), aa_);

    if (!vi.IsY8()) {
        processPlane(env, srcFrame->GetReadPtr(PLANAR_U), dstFrame->GetWritePtr(PLANAR_U), srcFrame->GetRowSize(PLANAR_U), 
            srcFrame->GetHeight(PLANAR_U), srcFrame->GetPitch(PLANAR_U), srcFrame->GetPitch(PLANAR_U), 0);

        processPlane(env, srcFrame->GetReadPtr(PLANAR_V), dstFrame->GetWritePtr(PLANAR_V), srcFrame->GetRowSize(PLANAR_V), 
            srcFrame->GetHeight(PLANAR_V), srcFrame->GetPitch(PLANAR_V), srcFrame->GetPitch(PLANAR_V), 0);
    }

    return dstFrame;
}


AVSValue __cdecl Create_SangNom2(AVSValue args, void*, IScriptEnvironment* env) {
    enum { CLIP, ORDER, AA };
    return new SangNom2(args[CLIP].AsClip(), args[ORDER].AsInt(1), args[AA].AsInt(48), env);
}

extern "C" __declspec(dllexport) const char* __stdcall AvisynthPluginInit2(IScriptEnvironment* env) {
    env->AddFunction("SangNom2", "c[order]i[aa]i", Create_SangNom2, 0);
    return "`x' xxx";
}
