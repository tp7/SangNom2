#include <Windows.h>
#pragma warning(disable: 4512 4244 4100)
#include "avisynth.h"
#pragma warning(default: 4512 4244 4100)
#include <emmintrin.h>

#ifdef __INTEL_COMPILER
#define MT_FORCEINLINE inline
#else
#define MT_FORCEINLINE __forceinline
#endif

#define USE_MOVPS

extern "C" {

    static MT_FORCEINLINE __m128i simd_load_si128(const __m128i* ptr) {
#ifdef USE_MOVPS
        return _mm_castps_si128(_mm_load_ps(reinterpret_cast<const float*>(ptr)));
#else
        return _mm_load_si128(ptr);
#endif
    }

    static MT_FORCEINLINE __m128i simd_loadu_si128(const __m128i* ptr) {
#ifdef USE_MOVPS
        return _mm_castps_si128(_mm_loadu_ps(reinterpret_cast<const float*>(ptr)));
#else
        return _mm_loadu_si128(ptr);
#endif
    }

    static MT_FORCEINLINE void simd_store_si128(__m128i *ptr, __m128i value) {
#ifdef USE_MOVPS
        _mm_store_ps(reinterpret_cast<float*>(ptr), _mm_castsi128_ps(value));
#else
        _mm_store_si128(ptr, value);
#endif
    }

    static MT_FORCEINLINE void simd_storeu_si128(__m128i *ptr, __m128i value) {
#ifdef USE_MOVPS
        _mm_storeu_ps(reinterpret_cast<float*>(ptr), _mm_castsi128_ps(value));
#else
        _mm_storeu_si128(ptr, value);
#endif
    }
}

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

class SangNom2 : public GenericVideoFilter {
public:
    SangNom2(PClip child, int order, int aa, IScriptEnvironment* env);

    PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env);

    ~SangNom2() {
        for (int i = 0; i < 9; i++) {
            _mm_free(buffers[i]);
        }
        _mm_free(intermediate);
    }

private:
    int order_;
    int offset_;
    int aa_;

    BYTE *buffers[9];
    int bufferWidth_;
    int bufferHeight_;
    BYTE *intermediate;

    void processPlane(IScriptEnvironment* env, const BYTE* srcp, BYTE* dstp, int width, int height, int src_pitch, int dst_pitch);
    void processPlane(const BYTE* pSrc, BYTE* pDst, int width, int height, int srcPitch, int dstPitch);
    void prepareBuffers(const BYTE* pSrc, BYTE* pDst, int width, int height, int srcPitch, int dstPitch);
    void processBuffers( int width, int height, int srcPitch, int dstPitch );
};

SangNom2::SangNom2(PClip child, int order, int aa, IScriptEnvironment* env)
    : GenericVideoFilter(child), order_(order) {
        if(!vi.IsPlanar()) {
            env->ThrowError("SangNom2 works only with planar colorspaces");
        }

        if (!(env->GetCPUFlags() && CPUF_SSE2)) {
            env->ThrowError("Sorry, SSE2 is requried");
        }

        bufferWidth_ = (vi.width + 15) / 16 * 16;
        bufferHeight_ = (vi.height + 1) / 2;
        for (int i = 0; i < 9; i++) {
            buffers[i] = reinterpret_cast<BYTE*>(_mm_malloc(bufferWidth_ * bufferHeight_, 16));
            memset(buffers[i], 0,bufferWidth_ * bufferHeight_); //this is important
        }
        intermediate = reinterpret_cast<BYTE*>(_mm_malloc(bufferWidth_*2, 16));
        //int edx = aa;
        aa = min(128, aa);
        aa_ = (21 * aa) / 16;
}

static __forceinline __m128i simd_abs_diff_epu8(__m128i a, __m128i b) {
    auto positive = _mm_subs_epu8(a, b);
    auto negative = _mm_subs_epu8(b, a);
    return _mm_or_si128(positive, negative);
}

static __forceinline __m128i calculateSangnom(const __m128i& p1, const __m128i& p2, const __m128i& p3) {
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

void SangNom2::prepareBuffers(const BYTE* pSrc, BYTE* pDst, int width, int height, int srcPitch, int dstPitch) {
    pSrc += offset_ * srcPitch;
    auto pSrcn2 = pSrc + srcPitch*2;
    auto zero = _mm_setzero_si128();

    int bufferOffset = width;

    for (int y = 1; y < height / 2; y++) {
        for (int x = 0; x < width; x += 16) {
            auto cur_minus_3   = simd_loadu_si128(reinterpret_cast<const __m128i*>(pSrc+x-3)); 
            auto cur_minus_2   = simd_loadu_si128(reinterpret_cast<const __m128i*>(pSrc+x-2)); 
            auto cur_minus_1   = simd_loadu_si128(reinterpret_cast<const __m128i*>(pSrc+x-1)); 
            auto cur           = simd_loadu_si128(reinterpret_cast<const __m128i*>(pSrc+x));
            auto cur_plus_1    = simd_loadu_si128(reinterpret_cast<const __m128i*>(pSrc+x+1)); 
            auto cur_plus_2    = simd_loadu_si128(reinterpret_cast<const __m128i*>(pSrc+x+2)); 
            auto cur_plus_3    = simd_loadu_si128(reinterpret_cast<const __m128i*>(pSrc+x+3)); 

            auto next_minus_3  = simd_loadu_si128(reinterpret_cast<const __m128i*>(pSrcn2+x-3)); 
            auto next_minus_2  = simd_loadu_si128(reinterpret_cast<const __m128i*>(pSrcn2+x-2)); 
            auto next_minus_1  = simd_loadu_si128(reinterpret_cast<const __m128i*>(pSrcn2+x-1)); 
            auto next          = simd_loadu_si128(reinterpret_cast<const __m128i*>(pSrcn2+x)); 
            auto next_plus_1   = simd_loadu_si128(reinterpret_cast<const __m128i*>(pSrcn2+x+1)); 
            auto next_plus_2   = simd_loadu_si128(reinterpret_cast<const __m128i*>(pSrcn2+x+2)); 
            auto next_plus_3   = simd_loadu_si128(reinterpret_cast<const __m128i*>(pSrcn2+x+3)); 

            auto adiff_m3_p3 = simd_abs_diff_epu8(cur_minus_3, next_plus_3);
            simd_storeu_si128(reinterpret_cast<__m128i*>(buffers[ADIFF_M3_P3]+bufferOffset+x), adiff_m3_p3);

            auto adiff_m2_p2 = simd_abs_diff_epu8(cur_minus_2, next_plus_2);
            simd_storeu_si128(reinterpret_cast<__m128i*>(buffers[ADIFF_M2_P2]+bufferOffset+x), adiff_m2_p2);

            auto adiff_m1_p1 = simd_abs_diff_epu8(cur_minus_1, next_plus_1);
            simd_storeu_si128(reinterpret_cast<__m128i*>(buffers[ADIFF_M1_P1]+bufferOffset+x), adiff_m1_p1);

            auto adiff_0     = simd_abs_diff_epu8(cur, next);
            simd_storeu_si128(reinterpret_cast<__m128i*>(buffers[ADIFF_P0_M0]+bufferOffset+x), adiff_0);

            auto adiff_p1_m1 = simd_abs_diff_epu8(cur_plus_1, next_minus_1);
            simd_storeu_si128(reinterpret_cast<__m128i*>(buffers[ADIFF_P1_M1]+bufferOffset+x), adiff_p1_m1);

            auto adiff_p2_m2 = simd_abs_diff_epu8(cur_plus_2, next_minus_2);
            simd_storeu_si128(reinterpret_cast<__m128i*>(buffers[ADIFF_P2_M2]+bufferOffset+x), adiff_p2_m2);

            auto adiff_p3_m3 = simd_abs_diff_epu8(cur_plus_3, next_minus_3);
            simd_storeu_si128(reinterpret_cast<__m128i*>(buffers[ADIFF_P3_M3]+bufferOffset+x), adiff_p3_m3);

            //////////////////////////////////////////////////////////////////////////
            auto temp1 = calculateSangnom(cur_minus_1, cur, cur_plus_1);
            auto temp2 = calculateSangnom(next_plus_1, next, next_minus_1);

            //abs((cur_minus_1*4 + cur*5 - cur_plus_1) / 8  - (next_plus_1*4 + next*5 - next_minus_1) / 8)
            auto absdiff_p1_p2 = simd_abs_diff_epu8(temp1, temp2); 
            simd_storeu_si128(reinterpret_cast<__m128i*>(buffers[SG_FORWARD]+bufferOffset+x), absdiff_p1_p2);
            //////////////////////////////////////////////////////////////////////////
            auto temp3 = calculateSangnom(cur_plus_1, cur, cur_minus_1);
            auto temp4 = calculateSangnom(next_minus_1, next, next_plus_1);

            //abs((cur_plus_1*4 + cur*5 - cur_minus_1) / 8  - (next_minus_1*4 + next*5 - next_plus_1) / 8)
            auto absdiff_p3_p4 = simd_abs_diff_epu8(temp3, temp4);
            simd_storeu_si128(reinterpret_cast<__m128i*>(buffers[SG_REVERSE]+bufferOffset+x), absdiff_p3_p4);
            //////////////////////////////////////////////////////////////////////////
        }
        pSrc += srcPitch*2;
        pSrcn2 += srcPitch*2;
        bufferOffset += width;
    }
}

void SangNom2::processBuffers(int width, int height, int srcPitch, int dstPitch) {
    for (int i = 0; i < 9; ++i) {
        auto pSrc = buffers[i];
        auto pSrcn = pSrc + bufferWidth_;
        auto pSrcn2 = pSrcn + bufferWidth_;
        auto pDst = intermediate;
        auto zero = _mm_setzero_si128();
        for (int y = 0; y < bufferHeight_ - 1; ++y) {
            for(int x = 0; x < bufferWidth_; x+= 16) {
                auto src = simd_load_si128(reinterpret_cast<const __m128i*>(pSrc+x));
                auto srcn = simd_load_si128(reinterpret_cast<const __m128i*>(pSrcn+x));
                auto srcn2 = simd_load_si128(reinterpret_cast<const __m128i*>(pSrcn2+x));

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

                simd_store_si128(reinterpret_cast<__m128i*>(pDst+(x*2)), sum_lo);
                simd_store_si128(reinterpret_cast<__m128i*>(pDst+(x*2)+16), sum_hi);
            }

            for (int x = 0; x < bufferWidth_; x+= 16) {
                auto cur_minus_6_lo = simd_loadu_si128(reinterpret_cast<const __m128i*>(pDst+x*2-6));
                auto cur_minus_4_lo = simd_loadu_si128(reinterpret_cast<const __m128i*>(pDst+x*2-4));
                auto cur_minus_2_lo = simd_loadu_si128(reinterpret_cast<const __m128i*>(pDst+x*2-2));
                auto cur_lo         = simd_load_si128(reinterpret_cast<const __m128i*>(pDst+x*2));
                auto cur_plus_2_lo  = simd_loadu_si128(reinterpret_cast<const __m128i*>(pDst+x*2+2));
                auto cur_plus_4_lo  = simd_loadu_si128(reinterpret_cast<const __m128i*>(pDst+x*2+4));
                auto cur_plus_6_lo  = simd_loadu_si128(reinterpret_cast<const __m128i*>(pDst+x*2+6));

                auto cur_minus_6_hi = simd_loadu_si128(reinterpret_cast<const __m128i*>(pDst+x*2-6+16));
                auto cur_minus_4_hi = simd_loadu_si128(reinterpret_cast<const __m128i*>(pDst+x*2-4+16));
                auto cur_minus_2_hi = simd_loadu_si128(reinterpret_cast<const __m128i*>(pDst+x*2-2+16));
                auto cur_hi         = simd_load_si128(reinterpret_cast<const __m128i*>(pDst+x*2+16));
                auto cur_plus_2_hi  = simd_loadu_si128(reinterpret_cast<const __m128i*>(pDst+x*2+2+16));
                auto cur_plus_4_hi  = simd_loadu_si128(reinterpret_cast<const __m128i*>(pDst+x*2+4+16));
                auto cur_plus_6_hi  = simd_loadu_si128(reinterpret_cast<const __m128i*>(pDst+x*2+6+16));

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
                simd_store_si128(reinterpret_cast<__m128i*>(pSrcn+x), result);
            }

            pSrc += bufferWidth_;
            pSrcn += bufferWidth_;
            pSrcn2 += bufferWidth_;
        }
    }
}

static __forceinline __m128i doSomeWeirdMagic(const __m128i& a1, const __m128i& a2, const __m128i& buf, 
                                             const __m128i& minv, const __m128i& acc, const __m128i& zero) {
    auto average = _mm_avg_epu8(a1, a2);
    auto equalToMin = _mm_cmpeq_epi8(buf, minv);
    auto notEqualToMin = _mm_cmpeq_epi8(equalToMin, zero);
    auto accNotMin = _mm_and_si128(acc, notEqualToMin);
    auto idk = _mm_andnot_si128(notEqualToMin, average);
    return _mm_or_si128(accNotMin, idk);   
}

void SangNom2::processPlane(const BYTE* pSrc, BYTE* pDst, int width, int height, int srcPitch, int dstPitch) {
    pSrc += offset_ * srcPitch;
    pDst += offset_ * dstPitch;

    auto pSrcn2 = pSrc + srcPitch*2;
    auto zero = _mm_setzero_si128();
    auto aav = _mm_set1_epi8(aa_);

    for (int y = 1; y < height / 2; ++y) {
        for (int x = 0; x < width; x += 16) {
            auto buf0 = simd_loadu_si128(reinterpret_cast<const __m128i*>(buffers[ADIFF_M3_P3]+x+y*width)); 
            auto buf1 = simd_loadu_si128(reinterpret_cast<const __m128i*>(buffers[ADIFF_M2_P2]+x+y*width)); 
            auto buf2 = simd_loadu_si128(reinterpret_cast<const __m128i*>(buffers[ADIFF_M1_P1]+x+y*width)); 
            auto buf3 = simd_loadu_si128(reinterpret_cast<const __m128i*>(buffers[SG_FORWARD]+x+y*width)); 
            auto buf4 = simd_loadu_si128(reinterpret_cast<const __m128i*>(buffers[ADIFF_P0_M0]+x+y*width)); 
            auto buf5 = simd_loadu_si128(reinterpret_cast<const __m128i*>(buffers[SG_REVERSE]+x+y*width)); 
            auto buf6 = simd_loadu_si128(reinterpret_cast<const __m128i*>(buffers[ADIFF_P1_M1]+x+y*width)); 
            auto buf7 = simd_loadu_si128(reinterpret_cast<const __m128i*>(buffers[ADIFF_P2_M2]+x+y*width)); 
            auto buf8 = simd_loadu_si128(reinterpret_cast<const __m128i*>(buffers[ADIFF_P3_M3]+x+y*width)); 

            auto cur_minus_3   = simd_loadu_si128(reinterpret_cast<const __m128i*>(pSrc+x-3)); 
            auto cur_minus_2   = simd_loadu_si128(reinterpret_cast<const __m128i*>(pSrc+x-2)); 
            auto cur_minus_1   = simd_loadu_si128(reinterpret_cast<const __m128i*>(pSrc+x-1)); 
            auto cur           = simd_loadu_si128(reinterpret_cast<const __m128i*>(pSrc+x)); 
            auto cur_plus_1    = simd_loadu_si128(reinterpret_cast<const __m128i*>(pSrc+x+1)); 
            auto cur_plus_2    = simd_loadu_si128(reinterpret_cast<const __m128i*>(pSrc+x+2)); 
            auto cur_plus_3    = simd_loadu_si128(reinterpret_cast<const __m128i*>(pSrc+x+3)); 

            auto next_minus_3  = simd_loadu_si128(reinterpret_cast<const __m128i*>(pSrcn2+x-3)); 
            auto next_minus_2  = simd_loadu_si128(reinterpret_cast<const __m128i*>(pSrcn2+x-2)); 
            auto next_minus_1  = simd_loadu_si128(reinterpret_cast<const __m128i*>(pSrcn2+x-1)); 
            auto next          = simd_loadu_si128(reinterpret_cast<const __m128i*>(pSrcn2+x)); 
            auto next_plus_1   = simd_loadu_si128(reinterpret_cast<const __m128i*>(pSrcn2+x+1)); 
            auto next_plus_2   = simd_loadu_si128(reinterpret_cast<const __m128i*>(pSrcn2+x+2)); 
            auto next_plus_3   = simd_loadu_si128(reinterpret_cast<const __m128i*>(pSrcn2+x+3)); 

            auto minv = _mm_min_epu8(buf0, buf1);
            minv = _mm_min_epu8(minv, buf2);
            minv = _mm_min_epu8(minv, buf3);
            minv = _mm_min_epu8(minv, buf4);
            minv = _mm_min_epu8(minv, buf5);
            minv = _mm_min_epu8(minv, buf6);
            minv = _mm_min_epu8(minv, buf7);
            minv = _mm_min_epu8(minv, buf8);

            auto acc = _mm_setzero_si128();

            acc = doSomeWeirdMagic(cur_minus_3, next_plus_3, buf0, minv, acc, zero);
            acc = doSomeWeirdMagic(cur_plus_3, next_minus_3, buf8, minv, acc, zero);

            acc = doSomeWeirdMagic(cur_minus_2, next_plus_2, buf1, minv, acc, zero);
            acc = doSomeWeirdMagic(cur_plus_2, next_minus_2, buf7, minv, acc, zero);

            acc = doSomeWeirdMagic(cur_minus_1, next_plus_1, buf2, minv, acc, zero);
            acc = doSomeWeirdMagic(cur_plus_1, next_minus_1, buf6, minv, acc, zero);

            //////////////////////////////////////////////////////////////////////////
            auto temp1 = calculateSangnom(cur_minus_1, cur, cur_plus_1);
            auto temp2 = calculateSangnom(next_plus_1, next, next_minus_1);
            
            acc = doSomeWeirdMagic(temp1, temp2, buf3, minv, acc, zero);
            //////////////////////////////////////////////////////////////////////////
            auto temp3 = calculateSangnom(cur_plus_1, cur, cur_minus_1);
            auto temp4 = calculateSangnom(next_minus_1, next, next_plus_1);
            
            acc = doSomeWeirdMagic(temp3, temp4, buf5, minv, acc, zero);
            //////////////////////////////////////////////////////////////////////////

            auto avg = _mm_avg_epu8(cur, next);

            auto equalToMin = _mm_cmpeq_epi8(buf4, minv);
            auto notEqualToMin = _mm_cmpeq_epi8(equalToMin, zero);

            auto idk = _mm_subs_epu8(minv, aav);
            idk = _mm_cmpeq_epi8(idk, zero);
            notEqualToMin = _mm_and_si128(notEqualToMin, idk);

            acc = _mm_and_si128(acc, notEqualToMin);
            auto t2 = _mm_andnot_si128(notEqualToMin, avg);
            acc = _mm_or_si128(acc, t2);
            simd_storeu_si128(reinterpret_cast<__m128i*>(pDst+x+dstPitch), acc);
        }
        pSrc += srcPitch * 2;
        pSrcn2 += srcPitch * 2;
        pDst += dstPitch *2;
    }
}


void SangNom2::processPlane(IScriptEnvironment* env, const BYTE* pSrc, BYTE* pDst, int width, int height, int srcPitch, int dstPitch) {
    env->BitBlt(pDst + offset_ * dstPitch, dstPitch * 2, pSrc + offset_ * srcPitch, srcPitch * 2, width, height / 2);
    
    if (offset_ == 1) {
        env->BitBlt(pDst, dstPitch, pSrc + srcPitch, srcPitch, width,1);
    } else {
        env->BitBlt(pDst+dstPitch * (height-1), dstPitch, pSrc + srcPitch*(height-2), srcPitch, width,1);
    }

    prepareBuffers(pSrc, pDst, width, height, srcPitch, dstPitch);
    processBuffers(width, height, srcPitch, dstPitch);
    processPlane(pSrc, pDst, width, height, srcPitch, dstPitch);
}


PVideoFrame SangNom2::GetFrame(int n, IScriptEnvironment* env) {
    auto srcFrame = child->GetFrame(n, env);
    auto dstFrame = env->NewVideoFrame(vi);

    offset_ = order_ == 0 
        ? child->GetParity(n) ? 0 : 1
        : order_ == 1 ? 0 : 1;

    processPlane(env, srcFrame->GetReadPtr(PLANAR_Y), dstFrame->GetWritePtr(PLANAR_Y), srcFrame->GetRowSize(PLANAR_Y), 
        srcFrame->GetHeight(PLANAR_Y), srcFrame->GetPitch(PLANAR_Y), srcFrame->GetPitch(PLANAR_Y));

   /* processPlane(env, srcFrame->GetReadPtr(PLANAR_U), dstFrame->GetWritePtr(PLANAR_U), srcFrame->GetRowSize(PLANAR_U), 
        srcFrame->GetHeight(PLANAR_U), srcFrame->GetPitch(PLANAR_U), srcFrame->GetPitch(PLANAR_U));

    processPlane(env, srcFrame->GetReadPtr(PLANAR_V), dstFrame->GetWritePtr(PLANAR_V), srcFrame->GetRowSize(PLANAR_V), 
        srcFrame->GetHeight(PLANAR_V), srcFrame->GetPitch(PLANAR_V), srcFrame->GetPitch(PLANAR_V));*/

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
