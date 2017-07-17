/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Copyright (C) 2014-2015, Itseez Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/
#include <immintrin.h>

static inline void v_transpose4x4(const __m128i& a0, const __m128i& a1, const __m128i& a2, const __m128i& a3,
    __m128i& b0, __m128i& b1, __m128i& b2, __m128i& b3)
{
    __m128i t0 = _mm_unpacklo_epi32(a0, a1);
    __m128i t1 = _mm_unpacklo_epi32(a2, a3);
    __m128i t2 = _mm_unpackhi_epi32(a0, a1);
    __m128i t3 = _mm_unpackhi_epi32(a2, a3);

    b0 = _mm_unpacklo_epi64(t0, t1);
    b1 = _mm_unpackhi_epi64(t0, t1);
    b2 = _mm_unpacklo_epi64(t2, t3);
    b3 = _mm_unpackhi_epi64(t2, t3);
}

// if function uses AVX2 instruction, write it here
#if CV_AVX2
// src: 16 elements of 16bit
// dst0: 0th-7th 32bit expanded elements of src
// dst1: 8th-15th 32bit expanded elements of src
static inline void expand_u8tou16(const __m256i& src, __m256i& dst0, __m256i& dst1)
{
    const __m256i z = _mm256_setzero_si256();

    __m256i t0 = _mm256_unpacklo_epi8(src, z); // a0 .. a7,  a16.. a23
    __m256i t1 = _mm256_unpackhi_epi8(src, z); // a8 .. a15, a24.. a31

    dst0 = _mm256_permute2f128_si256(t00, t01, 0x20); // a0 .. a15
    dst1 = _mm256_permute2f128_si256(t00, t01, 0x31); // a16.. a31
}

// src: 32 elements of 8bit
// dst0: 0th-7th 32bit expanded elements of src
// dst1: 8th-15th 32bit expanded elements of src
// dst2: 16th-23rd 32bit expanded elements of src
// dst3: 24th-31st 32bit expanded elements of src
static inline void expand_u8tou32(const __m256i& src, __m256i& dst0, __m256i& dst1, __m256i& dst2, __m256i& dst3)
{
    const __m256i z = _mm256_setzero_si256();

    __m256i t0 = _mm256_unpacklo_epi8(src, z); // a0 .. a7,  a16 .. a23
    __m256i t1 = _mm256_unpackhi_epi8(src, z); // a8 .. a15, a24 .. a31

    __m256i t00 = _mm256_unpacklo_epi16(t0, z); // a0 .. a3,  a16 .. a19
    __m256i t01 = _mm256_unpackhi_epi16(t0, z); // a4 .. a7,  a20 .. a23
    __m256i t02 = _mm256_unpacklo_epi16(t1, z); // a8 .. a11, a24 .. a27
    __m256i t03 = _mm256_unpackhi_epi16(t1, z); // a12.. a15, a28 .. a31

    dst0 = _mm256_permute2f128_si256(t00, t01, 0x20); // a0 .. a7
    dst1 = _mm256_permute2f128_si256(t02, t03, 0x20); // a8 .. a15
    dst2 = _mm256_permute2f128_si256(t00, t01, 0x31); // a16.. a23
    dst3 = _mm256_permute2f128_si256(t02, t03, 0x31); // a24.. a31
}

// src: 32 elements of 8bit
// dst0: 0th-7th 32bit expanded elements of src
// dst1: 8th-15th 32bit expanded elements of src
// dst2: 16th-23rd 32bit expanded elements of src
// dst3: 24th-31st 32bit expanded elements of src
static inline void expand_u8to32f(const __m256i& src, __m256& dst0, __m256& dst1, __m256& dst2, __m256& dst3)
{
    __m256i a0, a1, a2, a3;
    expand_u8tou32(src, a0, a1, a2, a3);

    dst0 = _mm256_cvtepi32_ps(a0); // a0 .. a7
    dst1 = _mm256_cvtepi32_ps(a1); // a8 .. a15
    dst2 = _mm256_cvtepi32_ps(a2); // a16.. a23
    dst3 = _mm256_cvtepi32_ps(a3); // a24.. a31
}

// src: 32 elements of 8bit
// dst0: 0th-3rd 64bit expanded elements of src
// dst1: 4th-7th 64bit expanded elements of src
// dst2: 8th-11th 64bit expanded elements of src
// dst3: 12th-15th 64bit expanded elements of src
// dst4: 16th-19th 64bit expanded elements of src
// dst5: 20th-23rd 64bit expanded elements of src
// dst6: 24th-27th 64bit expanded elements of src
// dst7: 28th-31st 64bit expanded elements of src
static inline void expand_u8to64f(const __m256i& src, __m256d& dst0, __m256d& dst1, __m256d& dst2, __m256d& dst3, __m256d& dst4, __m256d& dst5, __m256d& dst6, __m256d& dst7)
{
    const __m256i z = _mm256_setzero_si256();

    __m256i t0 = _mm256_unpacklo_epi8(src, z); // a0 .. a7,  a16 .. a23
    __m256i t1 = _mm256_unpackhi_epi8(src, z); // a8 .. a15, a24 .. a31

    __m256i t00 = _mm256_unpacklo_epi16(t0, z); // a0 .. a3,  a16 .. a19
    __m256i t01 = _mm256_unpackhi_epi16(t0, z); // a4 .. a7,  a20 .. a23
    __m256i t02 = _mm256_unpacklo_epi16(t1, z); // a8 .. a11, a24 .. a27
    __m256i t03 = _mm256_unpackhi_epi16(t1, z); // a12.. a15, a28 .. a31

    dst0 = _mm256_cvtepi32_pd(_mm256_extracti128_si256(t00, 0)); // a0  a1  a2  a3
    dst1 = _mm256_cvtepi32_pd(_mm256_extracti128_si256(t01, 0)); // a4  a5  a6  a7
    dst2 = _mm256_cvtepi32_pd(_mm256_extracti128_si256(t02, 0)); // a8  a9  a10 a11
    dst3 = _mm256_cvtepi32_pd(_mm256_extracti128_si256(t03, 0)); // a12 a13 a14 a15
    dst4 = _mm256_cvtepi32_pd(_mm256_extracti128_si256(t00, 1)); // a16 a17 a18 a19
    dst5 = _mm256_cvtepi32_pd(_mm256_extracti128_si256(t01, 1)); // a20 a21 a22 a23
    dst6 = _mm256_cvtepi32_pd(_mm256_extracti128_si256(t02, 1)); // a24 a25 a26 a27
    dst7 = _mm256_cvtepi32_pd(_mm256_extracti128_si256(t03, 1)); // a28 a29 a30 a31
}

// src: 16 elements of 16bit
// dst0: 0th-3rd 64bit expanded elements of src
// dst1: 4th-7th 64bit expanded elements of src
// dst2: 8th-11th 64bit expanded elements of src
// dst3: 12th-15th 64bit expanded elements of src
static inline void expand_u16to64f(const __m256i& src, __m256d& dst0, __m256d& dst1, __m256d& dst2, __m256d& dst3)
{
    const __m256i z = _mm256_setzero_si256();

    __m256i t0 = _mm256_unpacklo_epi16(src, z); // a0 .. a3, a8 .. a11
    __m256i t1 = _mm256_unpackhi_epi16(src, z); // a4 .. a7, a12.. a15

    dst0 = _mm256_cvtepi32_pd(_mm256_extracti128_si256(t0, 0)); // a0  a1  a2  a3
    dst1 = _mm256_cvtepi32_pd(_mm256_extracti128_si256(t1, 0)); // a4  a5  a6  a7
    dst2 = _mm256_cvtepi32_pd(_mm256_extracti128_si256(t0, 1)); // a8  a9  a10 a11
    dst3 = _mm256_cvtepi32_pd(_mm256_extracti128_si256(t1, 1)); // a12 a13 a14 a15
}

// src: 16 elements of 16bit
// dst0: 0th-7th 32bit expanded elements of src
// dst1: 8th-15th 32bit expanded elements of src
static inline void expand_u16tou32(const __m256i& src, __m256i& dst0, __m256i& dst1)
{
    const __m256i z = _mm256_setzero_si256();

    __m256i t0 = _mm256_unpacklo_epi16(src, z); // a0 .. a3, a8 .. a11
    __m256i t1 = _mm256_unpackhi_epi16(src, z); // a4 .. a7, a12.. a15

    dst0 = _mm256_permute2f128_si256(t00, t01, 0x20); // a0 .. a7
    dst1 = _mm256_permute2f128_si256(t00, t01, 0x31); // a8 .. a15
}

// src: 16 elements of 16bit
// dst0: 0th-7th 32bit expanded elements of src
// dst1: 8th-15th 32bit expanded elements of src
static inline void expand_u16to32f(const __m256i& src, __m256& dst0, __m256& dst1)
{
    __m256i a0, a1;
    expand_u16tou32(src, a0, a1);

    dst0 = _mm256_cvtepi32_ps(a0); // a0 .. a7
    dst1 = _mm256_cvtepi32_ps(a1); // a8 .. a15
}

// ptr: uchar pointer
// dst0--dst3: expanded float
static inline void load_expand(const uchar* ptr, __m256& dst0, __m256& dst1, __m256& dst2, __m256& dst3)
{
    expand_u8to32f(_mm256_loadu_si256((const __m256i*)ptr), dst0, dst1, dst2, dst3);
}

// ptr: uchar pointer
// dst0--dst7: expanded double float
static inline void load_expand(const uchar* ptr, __m256d& dst0, __m256d& dst1, __m256d& dst2, __m256d& dst3, __m256d& dst4, __m256d& dst5, __m256d& dst6, __m256d& dst7)
{
    expand_u8to64f(_mm256_loadu_si256((const __m256i*)ptr), dst0, dst1, dst2, dst3, dst4, dst5, dst6, dst7);
}

// ptr: uchar pointer
// dst0--dst3: expanded float
static inline void load_and_mask_expand(const uchar* ptr, const uchar* mask, __m256& dst0, __m256& dst1, __m256& dst2, __m256& dst3)
{
    __m256i z = _mm256_
    __m256i s = _mm256_loadu_si256((const __m256i*)ptr);
    __m256i m = _mm256_loadu_si256((const __m256i*)mask);
    s = _mm256_andnot_si256(
}

// ptr: ushort pointer
// dst0--dst1: expanded double float
static inline void load_expand(const ushort* ptr, __m256& dst0, __m256& dst1)
{
    expand_u16to32f(_mm256_loadu_si256((const __m256i*)ptr), dst0, dst1);
}

// ptr: ushort pointer
// dst0--dst1: expanded double float
static inline void load_expand(const ushort* ptr, __m256& dst0, __m256& dst1, __m256& dst2, __m256& dst3)
{
    expand_u16to64f(_mm256_loadu_si256((const __m256i*)ptr), dst0, dst1, dst2, dst3);
}

// ptr: uchar pointer
// m0--m3: expanded u32 mask
static inline void load_expand_mask(const uchar* ptr, __m256i& m0, __m256i& m1, __m256i& m2, __m256i& m3)
{
    expand_u8tou32(_mm256_loadu_si256((const __m256i*)ptr), m0, m1, m2, m3);
}

// ptr: uchar pointer
// m0--m1: expanded u16 mask
static inline void load_expand_mask(const uchar* ptr, __m256i& m0, __m256i& m1)
{
    expand_u8tou16(_mm256_loadu_si256((const __m256i*)ptr), m0, m1);
}

static inline void load_expand_mask(const uchar* ptr, __m256& m0, __m256& m1, __m256& m2, __m256& m3)
{
    __m256i t0, t1, t2, t3;
    load_expand_mask(ptr, t0, t1, t2, t3);

    m0 = _mm256_castsi256_ps(t0);
    m1 = _mm256_castsi256_ps(t1);
    m2 = _mm256_castsi256_ps(t2);
    m3 = _mm256_castsi256_ps(t3);
}

// deinterleave 3channel uchar
static inline void load_deinterleave(const uchar* ptr, __m256i& a, __m256i& b, __m256i& c)
{
    const int step = 32;
    __m256i t0 = _mm256_loadu_si256((const __m256i*)(ptr));            // a0  b0  c0 .. c9  a10 b10
    __m256i t1 = _mm256_loadu_si256((const __m256i*)(ptr + step));     // c10 a11 b11.. b20 c20 a21
    __m256i t2 = _mm256_loadu_si256((const __m256i*)(ptr + step * 2)); // b21 c21 a22.. a31 b31 c31
    __m256i t3 = _mm256_permute2f128_si256(t1, t2, 0x21);              // a16 b16 c16.. c25 a26 b26
    __m256i t4 = _mm256_permute2f128_si256(t2, t2, 0x33);              // c26 a27 b27.. a31 b31 c31 c26 a27 b27.. a31 b31 c31

    __m256i v00 = _mm256_unpacklo_epi8(t0, t3);              // a0  a16 b   b17.. a3  a19 b3  b19 b6  b22 c6  c22..b8  b24 c8  c24
    __m256i v01 = _mm256_unpackhi_epi8(t0, t3);              // c3  c19 a4  a20.. b5  b21 a6  a22 a9  a25 b9  b25..a11 a27 b11 b27
    __m256i v02 = _mm256_unpacklo_epi8(t1, t4);              // c11 c27 a12 a28.. c13 c29 a14 a30
    __m256i v03 = _mm256_unpackhi_epi8(t1, t4);              // b14 b30 c14 c30.. b16 b32 c16 c32
    __m256i v04 = _mm256_permute2f128_si256(v02, v03, 0x20); // c11 c27 a12 a28.. c13 c29 a14 a30 b14 b30 c14 c30..b16 b32 c16 c32
    __m256i v05 = _mm256_permute2f128_si256(v01, v03, 0x21); // a9  a25 b9  b25.. a11 a27 b11 b27 b14 b30 c14 c30..b16 b32 c16 c32

    __m256i v10 = _mm256_unpacklo_epi8(v00, v05);
    __m256i v11 = _mm256_unpackhi_epi8(v00, v05);
    __m256i v12 = _mm256_unpacklo_epi8(v01, v04);
    __m256i v13 = _mm256_unpackhi_epi8(v01, v04);
    __m256i v14 = _mm256_permute2f128_si256(v11, v12, 0x20);
    __m256i v15 = _mm256_permute2f128_si256(v10, v11, 0x31);

    __m256i v20 = _mm256_unpacklo_epi8(v14, v15);
    __m256i v21 = _mm256_unpackhi_epi8(v14, v15);
    __m256i v22 = _mm256_unpacklo_epi8(v10, v13);
    __m256i v23 = _mm256_unpackhi_epi8(v10, v13);
    __m256i v24 = _mm256_permute2f128_si256(v22, v20, 0x20);
    __m256i v25 = _mm256_permute2f128_si256(v20, v20, 0x11);

    __m256i v30 = _mm256_unpacklo_epi8(v24, v21);
    __m256i v31 = _mm256_unpackhi_epi8(v24, v21);
    __m256i v32 = _mm256_unpacklo_epi8(v23, v25);
    __m256i v33 = _mm256_unpackhi_epi8(v23, v25);
    __m256i v34 = _mm256_permute2f128_si256(v33, v30, 0x30);
    __m256i v35 = _mm256_permute2f128_si256(v31, v31, 0x11);
    __m256i v36 = _mm256_permute2f128_si256(v30, v31, 0x20);

    __m256i v40 = _mm256_unpacklo_epi8(v36, v34);
    __m256i v41 = _mm256_unpackhi_epi8(v36, v34);
    __m256i v42 = _mm256_unpacklo_epi8(v32, v35);
    __m256i v43 = _mm256_unpackhi_epi8(v32, v35);

    a = _mm256_permute2f128_si256(v40, v41, 0x20);
    b = _mm256_permute2f128_si256(v40, v41, 0x31);
    c = _mm256_permute2f128_si256(v42, v43, 0x20);
}

// deinterleave 3channel unsigned short
static inline void load_deinterleave(unsigned short* src, __m256i& a, __m256i& b, __m256i& c)
{
    __m256i s0 = _mm256_loadu_si256((const __m256i*)src);        // a1,  b1,  c1,  a2,  b2,  c2,  a3,  b3,  c3,  a4,  b4,  c4,  a5,  b5,  c5,  a6,
    __m256i s1 = _mm256_loadu_si256((const __m256i*)(src + 16)); // b6,  c6,  a7,  b7,  c7,  a8,  b8,  c8,  a9,  b9,  c9,  a10, b10, c10, a11, b11,
    __m256i s2 = _mm256_loadu_si256((const __m256i*)(src + 32)); // c11, a12, b12, c12, a13, b13, c13, a14, b14, c14, a15, b15, c15, a16, b16, c16,
    __m256i s3 = _mm256_permute2f128_si256(s1, s2, 0x21);        // a9,  b9,  c9,  a10, b10, c10, a11, b11, c11, a12, b12, c12, a13, b13, c13, a14,
    __m256i s4 = _mm256_permute2f128_si256(s2, s2, 0x33);        // b14, c14, a15, b15, c15, a16, b16, c16, b14, c14, a15, b15, c15, a16, b16, c16,

    __m256i v00 = _mm256_unpacklo_epi16(s0, s3);             // a1,  a9,  b1,  b9,  c1,  c9,  a2,  a10, c3,  c11, a4,  a12, b4,  b12, c4,  c12,
    __m256i v01 = _mm256_unpackhi_epi16(s0, s3);             // b2,  b10, c2,  c10, a3,  a11, b3,  b11, a5,  a13, b5,  b13, c5,  c13, a6,  a14,
    __m256i v02 = _mm256_unpacklo_epi16(s1, s4);             // b6,  b14, c6,  c14, a7,  a15, b7,  b15, x,   x,   x,   x,   x,   x,   x,   x,
    __m256i v03 = _mm256_unpackhi_epi16(s1, s4);             // c7,  c15, a8,  a16, b8,  b16, c8,  c16, x,   x,   x,   x,   x,   x,   x,   x,
    __m256i v04 = _mm256_permute2f128_si256(v02, v03, 0x20); // b6,  b14, c6,  c14, a7,  a15, b7,  b15, c7,  c15, a8,  a16, b8,  b16, c8,  c16,
    __m256i v05 = _mm256_permute2f128_si256(v01, v03, 0x21); // a5,  a13, b5,  b13, c5,  c13, a6,  a14, c7,  c15, a8,  a16, b8,  b16, c8,  c16,

    __m256i v10 = _mm256_unpacklo_epi16(v00, v05);           // a1,  a5,  a9,  a13, b1,  b5,  b9,  b13, c3,  c7,  c11, c15, a4,  a8,  a12, a16,
    __m256i v11 = _mm256_unpackhi_epi16(v00, v05);           // c1,  c5,  c9,  c13, a2,  a6,  a10, a14, b4,  b8,  b12, b16, c4,  c8,  c12, c16,
    __m256i v12 = _mm256_unpacklo_epi16(v01, v04);           // b2,  b6,  b10, b14, c2,  c6,  c10, c14, x,   x,   x,   x,   x,   x,   x,   x,
    __m256i v13 = _mm256_unpackhi_epi16(v01, v04);           // a3,  a7,  a11, a15, b3,  b7,  b11, b15, x,   x,   x,   x,   x,   x,   x,   x,
    __m256i v14 = _mm256_permute2f128_si256(v10, v11, 0x20); // a1,  a5,  a9,  a13, b1,  b5,  b9,  b13, c1,  c5,  c9,  c13, a2,  a6,  a10, a14,
    __m256i v15 = _mm256_permute2f128_si256(v13, v10, 0x30); // a3,  a7,  a11, a15, b3,  b7,  b11, b15, c3,  c7,  c11, c15, a4,  a8,  a12, a16,
    __m256i v16 = _mm256_permute2f128_si256(v11, v11, 0x33); // b4,  b8,  b12, b16, c4,  c8,  c12, c16, b4,  b8,  b12, b16, c4,  c8,  c12, c16,

    __m256i v20 = _mm256_unpacklo_epi16(v14, v15);           // a1,  a3,  a5,  a7,  a9,  a11, a13, a15, c1,  c3,  c5,  c7,  c9,  c11, c13, c15,
    __m256i v21 = _mm256_unpackhi_epi16(v14, v15);           // b1,  b3,  b5,  b7,  b9,  b11, b13, b15, a2,  a4,  a6,  a8,  a10, a12, a14, a16,
    __m256i v22 = _mm256_unpacklo_epi16(v12, v16);           // b2,  b4,  b6,  b8,  b10, b12, b14, b16, x,   x,   x,   x,   x,   x,   x,   x,
    __m256i v23 = _mm256_unpackhi_epi16(v12, v16);           // c2,  c4,  c6,  c8,  c10, c12, c14, c16, x,   x,   x,   x,   x,   x,   x,   x,
    __m256i v24 = _mm256_permute2f128_si256(v21, v23, 0x21); // a2,  a4,  a6,  a8,  a10, a12, a14, a16, c2,  c4,  c6,  c8,  c10, c12, c14, c16,

    __m256i v30 = _mm256_unpacklo_epi16(v20, v24);           // a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8,  c1,  c2,  c3,  c4,  c5,  c6,  c7,  c8,
    __m256i v31 = _mm256_unpackhi_epi16(v20, v24);           // a9,  a10, a11, a12, a13, a14, a15, a16, c9,  c10, c11, c12, c13, c14, c15, c16,
    __m256i v32 = _mm256_unpacklo_epi16(v21, v22);           // b1,  b2,  b3,  b4,  b5,  b6,  b7,  b8,  x,   x,   x,   x,   x,   x,   x,   x,
    __m256i v33 = _mm256_unpackhi_epi16(v21, v22);           // b9,  b10, b11, b12, b13, b14, b15, b16, x,   x,   x,   x,   x,   x,   x,   x,

    a = _mm256_permute2f128_si256(v30, v31, 0x20);           // a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8,  a9,  a10, a11, a12, a13, a14, a15, a16,
    b = _mm256_permute2f128_si256(v32, v33, 0x20);           // b1,  b2,  b3,  b4,  b5,  b6,  b7,  b8,  b9,  b10, b11, b12, b13, b14, b15, b16,
    c = _mm256_permute2f128_si256(v30, v31, 0x31);           // c1,  c2,  c3,  c4,  c5,  c6,  c7,  c8,  c9,  c10, c11, c12, c13, c14, c15, c16,
}
#endif

// write  AVX instructions only. do not use AVX2 instruction
#if CV_AVX
// load three 4-packed double vector and deinterleave
static inline void load_deinterleave(const double* ptr, __m256d& a, __m256d& b, __m256d& c)
{
    __m256d s0 = _mm256_loadu_pd(ptr);                 // a0, b0, c0, a1,
    __m256d s1 = _mm256_loadu_pd(ptr + 4);             // b1, c1, a2, b2,
    __m256d s2 = _mm256_loadu_pd(ptr + 8);             // c2, a3, b3, c3,
    __m256d s3 = _mm256_permute2f128_pd(s2, s2, 0x31); // b3, c3, b3, c3,
    __m256d s4 = _mm256_permute2f128_pd(s1, s2, 0x21); // a2, b2, c2, a3,

    __m256d v00 = _mm256_unpacklo_pd(s0, s4);             // a0, a2, c0, c2,
    __m256d v01 = _mm256_unpackhi_pd(s0, s4);             // b0, b2, a1, a3,
    __m256d v02 = _mm256_unpacklo_pd(s1, s3);             // b1, b3, x,  x,
    __m256d v03 = _mm256_unpackhi_pd(s1, s3);             // c1, c3, x,  x,
    __m256d v04 = _mm256_permute2f128_pd(v01, v03, 0x21); // a1, a3, c1, c3,

    __m256d v10 = _mm256_unpacklo_pd(v00, v04);           // a0, a1, c0, c1,
    __m256d v11 = _mm256_unpackhi_pd(v00, v04);           // a2, a3, c2, c3,
    __m256d v12 = _mm256_unpacklo_pd(v01, v02);           // b0, b1, x,  x,
    __m256d v13 = _mm256_unpackhi_pd(v01, v02);           // b2, b3, x,  x,

    a = _mm256_permute2f128_pd(v10, v11, 0x20);          // a0, a1, a2, a3,
    b = _mm256_permute2f128_pd(v12, v13, 0x20);          // b0, b1, b2, b3,
    c = _mm256_permute2f128_pd(v10, v11, 0x31);          // c0, c1, c2, c3,
}

// load three 8-packed float vector and deinterleave
// probably it's better to write down somewhere else
static inline void load_deinterleave(const float* ptr, __m256& a, __m256& b, __m256& c)
{
    __m256 s0 = _mm256_loadu_ps(ptr);                    // a0, b0, c0, a1, b1, c1, a2, b2,
    __m256 s1 = _mm256_loadu_ps(ptr + 8);                // c2, a3, b3, c3, a4, b4, c4, a5,
    __m256 s2 = _mm256_loadu_ps(ptr + 16);               // b5, c5, a6, b6, c6, a7, b7, c7,
    __m256 s3 = _mm256_permute2f128_ps(s1, s2, 0x21);    // a4, b4, c4, a5, b5, c5, a6, b6,
    __m256 s4 = _mm256_permute2f128_ps(s2, s2, 0x33);    // c6, a7, b7, c7, c6, a7, b7, c7,

    __m256 v00 = _mm256_unpacklo_ps(s0, s3);             // a0, a4, b0, b4, b1, b5, c1, c5,
    __m256 v01 = _mm256_unpackhi_ps(s0, s3);             // c0, c4, a1, a5, a2, a6, b2, b6,
    __m256 v02 = _mm256_unpacklo_ps(s1, s4);             // c2, c6, a3, a7, x,  x,  x,  x,
    __m256 v03 = _mm256_unpackhi_ps(s1, s4);             // b3, b7, c3, c7, x,  x,  x,  x,
    __m256 v04 = _mm256_permute2f128_ps(v02, v03, 0x20); // c2, c6, a3, a7, b3, b7, c3, c7,
    __m256 v05 = _mm256_permute2f128_ps(v01, v03, 0x21); // a2, a6, b2, b6, b3, b7, c3, c7,

    __m256 v10 = _mm256_unpacklo_ps(v00, v05);           // a0, a2, a4, a6, b1, b3, b5, b7,
    __m256 v11 = _mm256_unpackhi_ps(v00, v05);           // b0, b2, b4, b6, c1, c3, c5, c7,
    __m256 v12 = _mm256_unpacklo_ps(v01, v04);           // c0, c2, c4, c6, x,  x,  x,  x,
    __m256 v13 = _mm256_unpackhi_ps(v01, v04);           // a1, a3, a5, a7, x,  x,  x,  x,
    __m256 v14 = _mm256_permute2f128_ps(v11, v12, 0x20); // b0, b2, b4, b6, c0, c2, c4, c6,
    __m256 v15 = _mm256_permute2f128_ps(v10, v11, 0x31); // b1, b3, b5, b7, c1, c3, c5, c7,

    __m256 v20 = _mm256_unpacklo_ps(v14, v15);           // b0, b1, b2, b3, c0, c1, c2, c3,
    __m256 v21 = _mm256_unpackhi_ps(v14, v15);           // b4, b5, b6, b7, c4, c5, c6, c7,
    __m256 v22 = _mm256_unpacklo_ps(v10, v13);           // a0, a1, a2, a3, x,  x,  x,  x,
    __m256 v23 = _mm256_unpackhi_ps(v10, v13);           // a4, a5, a6, a7, x,  x,  x,  x,

    a = _mm256_permute2f128_ps(v22, v23, 0x20);          // a0, a1, a2, a3, a4, a5, a6, a7,
    b = _mm256_permute2f128_ps(v20, v21, 0x20);          // b0, b1, b2, b3, b4, b5, b6, b7,
    c = _mm256_permute2f128_ps(v20, v21, 0x31);          // c0, c1, c2, c3, c4, c5, c6, c7,
}

// realign four 3-packed vector to three 4-packed vector
static inline void v_pack4x3to3x4(const __m128i& s0, const __m128i& s1, const __m128i& s2, const __m128i& s3, __m128i& d0, __m128i& d1, __m128i& d2)
{
    d0 = _mm_or_si128(s0, _mm_slli_si128(s1, 12));
    d1 = _mm_or_si128(_mm_srli_si128(s1, 4), _mm_slli_si128(s2, 8));
    d2 = _mm_or_si128(_mm_srli_si128(s2, 8), _mm_slli_si128(s3, 4));
}

// separate high and low 128 bit and cast to __m128i
static inline void v_separate_lo_hi(const __m256& src, __m128i& lo, __m128i& hi)
{
    lo = _mm_castps_si128(_mm256_castps256_ps128(src));
    hi = _mm_castps_si128(_mm256_extractf128_ps(src, 1));
}

// interleave three 8-float vector and store
static inline void store_interleave(float* ptr, const __m256& a, const __m256& b, const __m256& c)
{
    __m128i a0, a1, b0, b1, c0, c1;
    v_separate_lo_hi(a, a0, a1);
    v_separate_lo_hi(b, b0, b1);
    v_separate_lo_hi(c, c0, c1);

    __m128i z = _mm_setzero_si128();
    __m128i u0, u1, u2, u3;
    v_transpose4x4(a0, b0, c0, z, u0, u1, u2, u3);
    v_pack4x3to3x4(u0, u1, u2, u3, a0, b0, c0);
    v_transpose4x4(a1, b1, c1, z, u0, u1, u2, u3);
    v_pack4x3to3x4(u0, u1, u2, u3, a1, b1, c1);

    _mm256_storeu_ps(ptr, _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_castsi128_ps(a0)), _mm_castsi128_ps(b0), 1));
    _mm256_storeu_ps(ptr + 8, _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_castsi128_ps(c0)), _mm_castsi128_ps(a1), 1));
    _mm256_storeu_ps(ptr + 16,  _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_castsi128_ps(b1)), _mm_castsi128_ps(c1), 1));
}

static inline void store_interleave(double* ptr, const __m256d& a, const __m256d& b, const __m256d& c)
{
    __m256d t0 = _mm256_unpacklo_pd(a, b);   // a0 b0 a2 b2 // a0 b0 c0 a1
    __m256d t1 = _mm256_unpackhi_pd(b, c);   // b1 c1 b3 c3 // b1 c1 a2 b2
    __m256d t2 = _mm256_blend_pd(c, a, 0xa); // c0 a1 c2 a3 // c2 a3 b3 c3

    _mm256_storeu_pd(ptr, _mm256_insertf128_pd(t0, _mm256_extractf128_pd(t2, 0), 1));     // a0 b0 c0 a1
    _mm256_storeu_pd(ptr + 4, _mm256_blend_pd(t0, t1, 3));                                // b1 c1 a2 b2
    _mm256_storeu_pd(ptr + 8, _mm256_insertf128_pd(t1, _mm256_extractf128_pd(t2, 1), 0)); // c2 a3 b3 c3
}
#endif

/* End of file */
