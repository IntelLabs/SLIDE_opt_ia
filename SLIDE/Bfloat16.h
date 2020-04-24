/*******************************************************************************
 * Copyright 2019 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *******************************************************************************/

#ifndef BFLOAT16_H
#define BFLOAT16_H

#include <immintrin.h>
#include <cmath>
#include <iostream>  // required for std::ostream o.w. can give compile error

struct bfloat16 {
  enum class Rounding : uint16_t {
    RNE = 0,
    TRUNC = 1
  };

  uint16_t bits_;

  union float_raw { // helper data type
    float fraw;
    uint32_t iraw;
    uint16_t wraw[2];
  };
  using float_raw = union float_raw;

  // Default constructor sets initial value to ZERO
  constexpr bfloat16(): bits_{0} { }

  // Constructor with explicit bits for bfloat16 value
  constexpr bfloat16(const uint16_t r): bits_{r} { }

  // Constructor from float that can ROUND (default) or TRUNCATE
  bfloat16(const float f, const Rounding flag = Rounding::RNE) {
    bits_ = cvt_float_to_bfloat16(f, flag);
  }

  template <size_t Nblock>
    static void bfloat16_block(float* in, bfloat16* out) {
      for (size_t i = 0; i < Nblock; ++i) {
        out[i] = bfloat16(in[i]);
      }
    };


  template <size_t Nblock>
    static void bf16ToFloat_block(bfloat16* in, float* out) {
      for (size_t i = 0; i < Nblock; ++i) {
        out[i] = static_cast<float>(in[i]);
      }
    };

  // Additional constructors from double, int, long and long long - depends on assignment operator
  template <typename T>
    bfloat16(const T v) {
      (*this) = v;
    }

  // Cast functions - convert bfloat16 values into others
  template <typename T>
    operator T() const {
      float_raw r;
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
      r.wraw[1] = 0;
      r.wraw[0] = bits_;
#else
      r.wraw[1] = bits_;
      r.wraw[0] = 0;
#endif
      return static_cast<T>(r.fraw);
    }

  explicit operator bool() const { return (bits_ & 0x7FFF) != 0; }

  // Construct bfloat16 from bits representation
  static constexpr bfloat16 from_bits(const uint16_t uibits_) {
    return bfloat16(uibits_); // depends on how we define bfloat(uint16_t) ctor
  }

  // Helper functions for helping with various roundings
  static uint16_t cvt_float_to_bfloat16 (const float& x, const Rounding flag = Rounding::RNE) {

    float_raw f = { .fraw = x };

#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    uint16_t r = f.wraw[0];
#else
    uint16_t r = f.wraw[1];
#endif

    if ( (r & 0x7F80) == 0x0000 ) { // zero or denormals (flush to zero)
      r &= 0x8000;
    } else if ( (f.iraw & 0x7FFFFFFF) > 0x7F800000 ) { // nan
      r |= (1 << 6); // quietize the NaN
    } else { // round to nearest even and truncate (takes care of infinite case as well)
      if (flag == Rounding::RNE) {
        uint32_t rounding_bias = 0x7FFF + (r & 0x1); // 0x00007FFF + LSB(of bf16)
        f.iraw += rounding_bias;
      }
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
      r = f.wraw[0];
#else
      r = f.wraw[1];
#endif
      // r = (f.iraw >> 16); // should be okay for both endians
    }

    return r;
  }

  // -- Below are some functions not was part of the original header file -
  // Note that this is not wanted in proposal (the one with flag) but without flag
  // is desired in the proposal. If we were to add flag and default value below
  // then compile error is given. To eliminate we need to get rid of the other implementation.
  // This needs to be discussed with the customer.
  bfloat16(const double v, const Rounding flag) {
    bits_ = cvt_float_to_bfloat16(v, flag);
  }

  // Assignment operator using bfloat16
  bfloat16& operator=(const bfloat16& bf){
    if (this == &bf){
      return *this;
    }

    bits_ = bf.bits_;
    return *this;
  }

  // Assignment operators using others
  template <typename T>
    bfloat16& operator=(const T v){

      float_raw f;
      f.fraw = static_cast<T>(v);
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
      bits_ = f.wraw[0];
#else
      bits_ = f.wraw[1];
#endif
      return *this;
    }

  constexpr bfloat16 (const bfloat16& other): bits_{other.bits_} {}
  ~bfloat16()                     = default;
};


// Arithmetic operators
inline bfloat16 operator+(const bfloat16& x, const bfloat16& y) { return bfloat16(static_cast<float>(x) + static_cast<float>(y));  }
inline bfloat16 operator-(const bfloat16& x, const bfloat16& y) { return bfloat16(static_cast<float>(x) - static_cast<float>(y)); }
inline bfloat16 operator*(const bfloat16& x, const bfloat16& y) { return bfloat16(static_cast<float>(x) * static_cast<float>(y)); }
inline bfloat16 operator/(const bfloat16& x, const bfloat16& y) { return bfloat16(static_cast<float>(x) / static_cast<float>(y)); }
inline bfloat16 operator-(const bfloat16& x) { return bfloat16(-static_cast<float>(x)); } // unary minus
inline bfloat16 operator+(const bfloat16& x) { return bfloat16(+static_cast<float>(x)); } // unary plus

inline bfloat16& operator+=(bfloat16& x, const bfloat16& y) {
  x = x + y;
  return x;
}
inline bfloat16& operator-=(bfloat16& x, const bfloat16& y) {
  x = x - y;
  return x;
}
inline bfloat16& operator*=(bfloat16& x, const bfloat16& y) {
  x = x * y;
  return x;
}
inline bfloat16& operator/=(bfloat16& x, const bfloat16& y) {
  // assert( y == 0 && "Division by zero detected");
  x = x / y;
  return x;
}

// Bitwise operators
inline bfloat16 operator&(const bfloat16& x, const bfloat16& y) {
  uint16_t xny = x.bits_ & y.bits_;
  bfloat16 tmp{xny};
  return tmp;
}
inline bfloat16 operator|(const bfloat16& x, const bfloat16& y) {
  uint16_t xny = x.bits_ | y.bits_;
  bfloat16 tmp{xny};
  return tmp;
}
inline bfloat16 operator^(const bfloat16& x, const bfloat16& y) {
  uint16_t xny = x.bits_ ^ y.bits_;
  bfloat16 tmp{xny};
  return tmp;
}
inline bfloat16 operator~(const bfloat16& x) {
  uint16_t xn = ~x.bits_;
  bfloat16 tmp{xn};
  return tmp;
}
inline bfloat16 operator<<(const bfloat16& x, const uint16_t& s) {
  uint16_t xs = x.bits_ << s;
  bfloat16 tmp{xs};
  return tmp;
}
inline bfloat16 operator>>(const bfloat16& x, const uint16_t& s) {
  uint16_t xs = x.bits_ >> s;
  bfloat16 tmp{xs};
  return tmp;
}
inline bfloat16& operator&=(bfloat16& x, const bfloat16& y) { // assign bitwise and
  x = (x & y);
  return x;
}
inline bfloat16& operator|=(bfloat16& x, const bfloat16& y) { // assign bitwise or
  x = (x | y);
  return x;
}
inline bfloat16& operator^=(bfloat16& x, const bfloat16& y) { // assign bitwise xor
  x = (x ^ y);
  return x;
}

// Mixed precision operators
template <typename T>
inline T operator+(const bfloat16& x, const T& y) { return static_cast<T>(x) + y; }
// int operator+(const bfloat16& x, const int& y) = delete;  // if any type is meant to be deleted, this is how to do it

template <typename T> inline T operator-(const bfloat16& x, const T& y) { return static_cast<T>(x) - y; }
template <typename T> inline T operator*(const bfloat16& x, const T& y) { return static_cast<T>(x) * y; }
template <typename T> inline T operator/(const bfloat16& x, const T& y) { return static_cast<T>(x) / y; }

template <typename T> inline T operator+(const T& x, const bfloat16& y) { return x + static_cast<T>(y); }
template <typename T> inline T operator-(const T& x, const bfloat16& y) { return x - static_cast<T>(y); }
template <typename T> inline T operator*(const T& x, const bfloat16& y) { return x * static_cast<T>(y); }
template <typename T> inline T operator/(const T& x, const bfloat16& y) { return x / static_cast<T>(y); }

template <typename T>
inline T& operator+=(T& a, const bfloat16& b){
  a += static_cast<T>(b);
  return a;
}

template <typename T>
inline T& operator-=(T& a, const bfloat16& b){
  a -= static_cast<T>(b);
  return a;
}

template <typename T>
inline T& operator*=(T& a, const bfloat16& b){
  a *= static_cast<T>(b);
  return a;
}

template <typename T>
inline T& operator/=(T& a, const bfloat16& b){
  a /= static_cast<T>(b);
  return a;
}

// Boolean operators with and without mixed types
template <typename T1, typename T2> inline bool operator< (const T1& x, const T2& y) { return static_cast<float>(x) <  static_cast<float>(y); }
template <typename T1, typename T2> inline bool operator<=(const T1& x, const T2& y) { return static_cast<float>(x) <= static_cast<float>(y); }
template <typename T1, typename T2> inline bool operator==(const T1& x, const T2& y) { return static_cast<float>(x) == static_cast<float>(y); }
template <typename T1, typename T2> inline bool operator> (const T1& x, const T2& y) { return static_cast<float>(x) >  static_cast<float>(y); }
template <typename T1, typename T2> inline bool operator>=(const T1& x, const T2& y) { return static_cast<float>(x) >= static_cast<float>(y); }
template <typename T1, typename T2> inline bool operator!=(const T1& x, const T2& y) { return static_cast<float>(x) != static_cast<float>(y); }
inline bool operator!(const bfloat16& x) { return !bool(x); } //logical complement

// Math/elementary functions
inline bfloat16 abs  (const bfloat16& x) { return bfloat16(fabsf(static_cast<float>(x))); }
inline bfloat16 fabs (const bfloat16& x) { return bfloat16(fabsf(static_cast<float>(x))); }
inline bfloat16 exp  (const bfloat16& x) { return bfloat16(expf(static_cast<float>(x))); }
inline bfloat16 exp2 (const bfloat16& x) { return bfloat16(exp2f(static_cast<float>(x))); }
inline bfloat16 exp10(const bfloat16& x) { return bfloat16(exp10f(static_cast<float>(x))); }
inline bfloat16 expm1(const bfloat16& x) { return bfloat16(expm1f(static_cast<float>(x))); }
inline bfloat16 log  (const bfloat16& x) { return bfloat16(logf(static_cast<float>(x))); }
inline bfloat16 log10(const bfloat16& x) { return bfloat16(log10f(static_cast<float>(x))); }
inline bfloat16 log1p(const bfloat16& x) { return bfloat16(log1pf(static_cast<float>(x))); }
inline bfloat16 log2 (const bfloat16& x) { return bfloat16(log2f(static_cast<float>(x))); }
inline bfloat16 sqrt (const bfloat16& x) { return bfloat16(sqrtf(static_cast<float>(x))); }
inline bfloat16 pow  (const bfloat16& x, const bfloat16& y) { return bfloat16(powf(static_cast<float>(x),static_cast<float>(y))); }
inline bfloat16 hypot(const bfloat16& x, const bfloat16& y) { return bfloat16(hypotf(static_cast<float>(x),static_cast<float>(y))); }
inline bfloat16 sin  (const bfloat16& x) { return bfloat16(sinf(static_cast<float>(x))); }
inline bfloat16 cos  (const bfloat16& x) { return bfloat16(cosf(static_cast<float>(x))); }
inline bfloat16 tan  (const bfloat16& x) { return bfloat16(tanf(static_cast<float>(x))); }
inline bfloat16 asin (const bfloat16& x) { return bfloat16(asinf(static_cast<float>(x))); }
inline bfloat16 acos (const bfloat16& x) { return bfloat16(acosf(static_cast<float>(x))); }
inline bfloat16 atan (const bfloat16& x) { return bfloat16(atanf(static_cast<float>(x))); }
inline bfloat16 atan2(const bfloat16& x, const bfloat16& y) { return bfloat16(atan2f(static_cast<float>(x), static_cast<float>(y))); }
inline bfloat16 tanh (const bfloat16& x) { return bfloat16(tanhf(static_cast<float>(x))); }
inline bfloat16 cosh (const bfloat16& x) { return bfloat16(coshf(static_cast<float>(x))); }
inline bfloat16 sinh (const bfloat16& x) { return bfloat16(sinhf(static_cast<float>(x))); }
inline bfloat16 atanh(const bfloat16& x) { return bfloat16(atanhf(static_cast<float>(x))); }
inline bfloat16 acosh(const bfloat16& x) { return bfloat16(acoshf(static_cast<float>(x))); }
inline bfloat16 asinh(const bfloat16& x) { return bfloat16(asinhf(static_cast<float>(x))); }
inline bfloat16 floor(const bfloat16& x) { return bfloat16(floorf(static_cast<float>(x))); }
inline bfloat16 ceil (const bfloat16& x) { return bfloat16(ceilf(static_cast<float>(x))); }
inline bfloat16 round(const bfloat16& x) { return bfloat16(roundf(static_cast<float>(x))); }
inline bfloat16 rint (const bfloat16& x) { return bfloat16(rintf(static_cast<float>(x))); }
inline bfloat16 erf  (const bfloat16& x) { return bfloat16(erff(static_cast<float>(x))); }
inline bfloat16 erfc (const bfloat16& x) { return bfloat16(erfcf(static_cast<float>(x))); }
inline bfloat16 cbrt (const bfloat16& x) { return bfloat16(cbrtf(static_cast<float>(x))); }
inline bfloat16 remainder(const bfloat16& x, const bfloat16& y) { return bfloat16(remainderf(static_cast<float>(x), static_cast<float>(y))); }
inline bfloat16 fdim (const bfloat16& x, const bfloat16& y) { return bfloat16(fdimf(static_cast<float>(x), static_cast<float>(y))); }
inline bfloat16 fma  (const bfloat16& x, const bfloat16& y, const bfloat16& z) { return bfloat16(fmaf(static_cast<float>(x), static_cast<float>(y), static_cast<float>(z))); }
// inline bfloat16 abs  (const float& x) { return bfloat16(fabsf(x)); }

// common math functions
inline bool isinf (const bfloat16& x) { return (x.bits_ & 0x7FFFu) == 0x7F80u; }
inline bool isqnan(const bfloat16& x) { return (x.bits_ & 0x7FC0u) == 0x7FC0u; }
inline bool issnan(const bfloat16& x) { return ((x.bits_ & 0x7F80u) == 0x7F80u) && ((x.bits_ & 0x007Fu) != 0x0000u); }
inline bool isnan (const bfloat16& x) {
  // return isqnan(x) || issnan(x)
  return ((x.bits_ & 0x7F80u) == 0x7F80u) && ((x.bits_ & 0x007Fu) != 0x0000u);
}
inline bool isfinite(const bfloat16& x) { return !(isnan(x) || isinf(x)); }

// extra operators
inline bfloat16& operator++(bfloat16& x) { // prefix increment

  bfloat16::float_raw r;
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  r.wraw[1] = 0;
  r.wraw[0] = x.bits_;
#else
  r.wraw[1] = x.bits_;
  r.wraw[0] = 0;
#endif

  r.fraw++;
  x = r.fraw;
  return x;
}
inline bfloat16 operator++(bfloat16& x, int unused) { // postfix increment
  bfloat16 temp = x;
  ++x;
  return temp;
}

inline bfloat16& operator--(bfloat16& x) { // prefix decrement

  bfloat16::float_raw r;
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  r.wraw[1] = 0;
  r.wraw[0] = x.bits_;
#else
  r.wraw[1] = x.bits_;
  r.wraw[0] = 0;
#endif

  r.fraw--;
  x = r.fraw;
  return x;
}
inline bfloat16 operator--(bfloat16& x, int unused) { // postfix decrement
  bfloat16 temp = x;
  --x;
  return temp;
}

// We have the std::ostream& operator<< version (see below - the namespace std section)
// inline dpcpp::ostream& operator<< (dpcpp::ostream &out, const bfloat16 &bf){
//     // BDEBUG("-friend in dpcpp - Using << overloading");
//     bfloat16::float_raw f;
//     f.wraw[1] = bf.bits_;
//     f.wraw[0] = 0;

//     out << f.fraw;
//     return out;
// }

static inline __m512 _mm512_mask_load_bf16_as_fp32(__m256i pack,
                                            __mmask16 k, const void *addr) {
  __m512i data = _mm512_cvtepu16_epi32(_mm256_mask_loadu_epi16(pack, k, addr));
  return _mm512_castsi512_ps(_mm512_bslli_epi128(data, 2));
}

static inline __m512 _mm512_maskz_load_bf16_as_fp32(__mmask16 k, const void *addr) {
  __m512i data = _mm512_cvtepu16_epi32(_mm256_maskz_loadu_epi16(k, addr));
  return _mm512_castsi512_ps(_mm512_bslli_epi128(data, 2));
}

static inline __m512 _mm512_load_bf16_as_fp32(const void *addr) {
  __m512i data = _mm512_cvtepu16_epi32(_mm256_loadu_epi16(addr));
  return _mm512_castsi512_ps(_mm512_bslli_epi128(data, 2));
}

static inline __m256i _mm512_cvt_fp32_to_bf16_emu_rne(const __m512& src) {
  __m512i isrc = _mm512_castps_si512(src);
  __m512i ones = _mm512_set1_epi32(0x1);
  __m512i bias = _mm512_set1_epi32(0x7fff);
  // t = (src >> 16) & 1;
  auto tmp = _mm512_and_si512(_mm512_srli_epi32(isrc, 16), ones);
  // t = 0x7fff + t;
  tmp = _mm512_add_epi32(tmp, bias);
  // t = src + t;
  tmp = _mm512_add_epi32(tmp, isrc);
  // t >>= 16;
  tmp = _mm512_srli_epi32(tmp, 16);
  return _mm512_cvtepi32_epi16(tmp);
}

static inline __m256i _mm512_cvt_fp32_to_bf16(__m512 src) {
#if OPT_CPX_BF16
  return _mm512_cvtneps_pbh(src);
#else
  return _mm512_cvt_fp32_to_bf16_emu_rne(src);
  // truncate
  //return _mm512_cvtepi32_epi16(_mm512_bsrli_epi128(_mm512_castps_si512(src), 2));
#endif
}

static inline void _mm512_mask_store_fp32_as_bf16(void *addr,
                                           __mmask16 k, __m512 data) {
  return _mm256_mask_storeu_epi16(addr, k, _mm512_cvt_fp32_to_bf16(data));

}

static inline void _mm512_store_fp32_as_bf16(void *addr, __m512 data) {
  return _mm256_storeu_epi16(addr, _mm512_cvt_fp32_to_bf16(data));

}

template <class T>
static inline __m512 _mm512_load(const void *addr) {
  if (std::is_same<float, T>::value) {
    return _mm512_load_ps(addr);
  } else {
    return _mm512_load_bf16_as_fp32(addr);
  }
}

template <class T>
static inline __m512 _mm512_maskz_load(__mmask16 k, void *addr) {
  if (std::is_same<float, T>::value) {
    return _mm512_maskz_load_ps(k, addr);
  } else {
    return _mm512_maskz_load_bf16_as_fp32(k, addr);
  }
}

template <class T>
static inline void _mm512_store(void *addr, __m512 data) {
  if (std::is_same<float, T>::value) {
    _mm512_store_ps(addr, data);
  } else {
    _mm512_store_fp32_as_bf16(addr, data);
  }
}

template <class T>
static inline void _mm512_mask_store(void *addr, __mmask16 k, __m512 data) {
  if (std::is_same<float, T>::value) {
    _mm512_mask_store_ps(addr, k, data);
  } else {
    _mm512_mask_store_fp32_as_bf16(addr, k, data);
  }
}

#endif
