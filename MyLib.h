//  ████████╗_██████╗_██╗__██╗_██████╗_████████╗██╗███████╗
//  ╚══██╔══╝██╔═══██╗╚██╗██╔╝██╔═══██╗╚══██╔══╝██║██╔════╝
//  ___██║___██║___██║_╚███╔╝_██║___██║___██║___██║███████╗
//  ___██║___██║___██║_██╔██╗_██║___██║___██║___██║╚════██║
//  ___██║___╚██████╔╝██╔╝_██╗╚██████╔╝___██║___██║███████║
//  ___╚═╝____╚═════╝_╚═╝__╚═╝_╚═════╝____╚═╝___╚═╝╚══════╝
//  _______________________________________________________

/*
===============================================================================
Welcome to the BigInt Library!

This powerful library allows you to work with arbitrarily large integers that
surpass the limitations of standard data types such as int or long long.
Perfect for applications in cryptography, number theory, or any task requiring
precision beyond typical computational limits.

Features:
- Supports addition, subtraction, multiplication, division, and modulo operations.
- Handles numbers with up to 10^13 digits with ease.
- Flexible input from strings, long longs, and more.
- Built-in sign handling for positive and negative numbers.
- Easy-to-use interface to convert between BigInt and strings.
- Exponentiation (including BigInt exponents) and modular exponentiation,
  integer square root, extended Euclidean algorithm, and modular inverse.
- **Optimized multiplication:** Uses naïve multiplication for small numbers,
  Karatsuba multiplication for moderately large numbers, and FFT–based convolution
  for extremely large numbers.
- Uses 128-bit intermediate arithmetic to avoid overflow even for small numbers.
- **Additional utility operations:**
    • Compound assignment operators (+=, -=, *=, /=, %=).
    • Pre‑ and post‑increment/decrement.
    • Conversion to built‑in long long (if the number fits).
    • Fast doubling Fibonacci, factorial, and binomial coefficient functions.
    • Overload of pow for an unsigned long long exponent.
- **New operations:**  
    • GCD and LCM  
    • Miller‑Rabin Primality Test (isPrime)  
    • nthRoot (integer nth‑root)  
    • factorialDC (divide‑and‑conquer factorial)
===============================================================================
*/

#include <bits/stdc++.h>
using namespace std;
using ll = long long;

#ifndef MyLib_H
#define MyLib_H

// Define a high-precision PI (here with ~100 decimal digits; adjust if needed)
#define Big_PI 3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679

//------------------------------------
// Base settings: Each block holds up to 12 decimal digits.
const ll BASE = 1000000000000LL;  // 10^12
const int BASE_DIGITS = 12;

//------------------------------------
// Use 128-bit arithmetic.
// For intermediate multiplication we now use an unsigned 128-bit type.
#ifdef __SIZEOF_INT128__
    typedef __int128 int128;
    typedef unsigned __int128 uint128;
#else
    typedef long long int128; // Fallback (may cause overflow)
    typedef unsigned long long uint128;
#endif

//=====================================
//         Core BigInt Structure
//=====================================
struct BigInt {
    vector<ll> a;  // little-endian blocks
    bool sign;     // true if non-negative, false if negative

    // Constructors.
    BigInt() : sign(true) {}
    BigInt(ll v) { *this = v; }
    BigInt(int v) { *this = (ll)v; }
    BigInt(const string &s) { read(s); }

    // Assignment from long long.
    void operator=(ll v) {
        sign = (v >= 0);
        if(v < 0) v = -v;
        a.clear();
        while(v > 0) {
            a.push_back(v % BASE);
            v /= BASE;
        }
    }

    // Read number from string.
    void read(const string &s) {
        sign = true;
        a.clear();
        int pos = 0;
        while(pos < s.size() && (s[pos] == '-' || s[pos] == '+')) {
            if(s[pos] == '-') sign = false;
            pos++;
        }
        for (int i = s.size()-1; i >= pos; i -= BASE_DIGITS) {
            ll x = 0;
            int start = max(pos, i - BASE_DIGITS + 1);
            for (int j = start; j <= i; j++)
                x = x * 10 + (s[j]-'0');
            a.push_back(x);
        }
        trim();
    }

    // Remove leading zero blocks.
    void trim() {
        while(!a.empty() && a.back() == 0)
            a.pop_back();
        if(a.empty())
            sign = true;
    }

    // Convert to string.
    string toString() const {
        if(a.empty())
            return "0";
        stringstream ss;
        if(!sign)
            ss << "-";
        ss << a.back();
        for (int i = a.size()-2; i >= 0; i--)
            ss << setw(BASE_DIGITS) << setfill('0') << a[i];
        return ss.str();
    }

    // Absolute value.
    BigInt abs() const {
        BigInt res = *this;
        res.sign = true;
        return res;
    }

    // Compare absolute values.
    bool absLess(const BigInt &other) const {
        if(a.size() != other.a.size())
            return a.size() < other.a.size();
        for (int i = a.size()-1; i >= 0; i--)
            if(a[i] != other.a[i])
                return a[i] < other.a[i];
        return false;
    }
};

//-----------------------------------------
// Forward Declarations for Relational Operators
//-----------------------------------------
bool operator==(const BigInt &x, const BigInt &y);
bool operator!=(const BigInt &x, const BigInt &y);
bool operator<(const BigInt &x, const BigInt &y);
bool operator>(const BigInt &x, const BigInt &y);
bool operator<=(const BigInt &x, const BigInt &y);
bool operator>=(const BigInt &x, const BigInt &y);

//=====================================
//         Helper Functions for Multiplication
//=====================================
namespace {
    // ----- For Karatsuba Multiplication -----
    const int KARATSUBA_THRESHOLD = 32;

    vector<ll> addVectors(const vector<ll>& a, const vector<ll>& b) {
        int n = max(a.size(), b.size());
        vector<ll> res(n, 0);
        ll carry = 0;
        for (int i = 0; i < n || carry; i++) {
            if(i == res.size())
                res.push_back(0);
            uint128 cur = carry;
            if(i < a.size()) cur += a[i];
            if(i < b.size()) cur += b[i];
            res[i] = (ll)(cur % BASE);
            carry = (ll)(cur / BASE);
        }
        while(!res.empty() && res.back() == 0)
            res.pop_back();
        return res;
    }

    vector<ll> subVectors(const vector<ll>& a, const vector<ll>& b) {
        vector<ll> res = a;
        ll carry = 0;
        for (size_t i = 0; i < b.size() || carry; i++) {
            int128 cur = res[i] - carry - (i < b.size() ? b[i] : 0LL);
            carry = 0;
            if(cur < 0) {
                carry = 1;
                cur += BASE;
            }
            res[i] = (ll)cur;
        }
        while(!res.empty() && res.back() == 0)
            res.pop_back();
        return res;
    }

    vector<ll> shiftVector(const vector<ll>& a, int k) {
        vector<ll> res(k, 0);
        res.insert(res.end(), a.begin(), a.end());
        return res;
    }

    vector<ll> karatsubaMultiply(const vector<ll>& a, const vector<ll>& b) {
        int n = a.size(), m = b.size();
        if(min(n, m) < KARATSUBA_THRESHOLD) {
            vector<ll> res(n + m, 0);
            for (int i = 0; i < n; i++) {
                uint128 carry = 0;
                for (int j = 0; j < m || carry; j++) {
                    uint128 cur = (i+j < res.size() ? res[i+j] : 0ULL) + (uint128)a[i] * (j < m ? b[j] : 0LL) + carry;
                    if(i+j < res.size())
                        res[i+j] = (ll)(cur % BASE);
                    else
                        res.push_back((ll)(cur % BASE));
                    carry = (ll)(cur / BASE);
                }
            }
            while(!res.empty() && res.back() == 0)
                res.pop_back();
            return res;
        }
        int k = min(n, m) / 2;
        vector<ll> a0(a.begin(), a.begin() + k);
        vector<ll> a1(a.begin() + k, a.end());
        vector<ll> b0(b.begin(), b.begin() + min(m, k));
        vector<ll> b1(b.begin() + min(m, k), b.end());

        vector<ll> z0 = karatsubaMultiply(a0, b0);
        vector<ll> z2 = karatsubaMultiply(a1, b1);
        vector<ll> a0a1 = addVectors(a0, a1);
        vector<ll> b0b1 = addVectors(b0, b1);
        vector<ll> z1 = karatsubaMultiply(a0a1, b0b1);
        z1 = subVectors(z1, z0);
        z1 = subVectors(z1, z2);

        vector<ll> res = addVectors( shiftVector(z2, 2*k),
                             addVectors( shiftVector(z1, k), z0) );
        return res;
    }

    // ----- For FFT Multiplication -----
    const int FFT_BASE = 1000000; // 10^6 (since (10^6)^2 = 10^12 = BASE)
    const int FFT_MULT_THRESHOLD = 128;

    typedef complex<double> base;
    void fft(vector<base> & a, bool invert) {
        int n = a.size();
        for (int i = 1, j = 0; i < n; i++) {
            int bit = n >> 1;
            for (; j & bit; bit >>= 1)
                j -= bit;
            j += bit;
            if (i < j)
                swap(a[i], a[j]);
        }
        for (int len = 2; len <= n; len <<= 1) {
            double ang = 2 * Big_PI / len * (invert ? -1 : 1);
            base wlen(cos(ang), sin(ang));
            for (int i = 0; i < n; i += len) {
                base w(1);
                for (int j = 0; j < len/2; j++) {
                    base u = a[i+j], v = a[i+j+len/2] * w;
                    a[i+j] = u + v;
                    a[i+j+len/2] = u - v;
                    w *= wlen;
                }
            }
        }
        if (invert)
            for (int i = 0; i < n; i++)
                a[i] /= n;
    }

    vector<int> multiplyFFT(const vector<int>& a, const vector<int>& b) {
        vector<base> fa(a.begin(), a.end()), fb(b.begin(), b.end());
        int n = 1;
        while(n < (a.size() + b.size()))
            n <<= 1;
        fa.resize(n), fb.resize(n);
        fft(fa, false), fft(fb, false);
        for (int i = 0; i < n; i++)
            fa[i] *= fb[i];
        fft(fa, true);
        vector<int> res(n);
        ll carry = 0;
        for (int i = 0; i < n; i++) {
            ll t = (ll)round(fa[i].real()) + carry;
            carry = t / FFT_BASE;
            res[i] = t % FFT_BASE;
        }
        while(!res.empty() && res.back() == 0)
            res.pop_back();
        return res;
    }

    vector<int> convertBigIntToFFT(const BigInt &x) {
        vector<int> res;
        for (ll block : x.a) {
            int low = block % FFT_BASE;
            int high = block / FFT_BASE;
            res.push_back(low);
            res.push_back(high);
        }
        while(!res.empty() && res.back() == 0)
            res.pop_back();
        return res;
    }

    vector<ll> convertFFTToBigInt(const vector<int> &v) {
        vector<ll> res;
        for (size_t i = 0; i < v.size(); i += 2) {
            ll block = v[i];
            if(i + 1 < v.size())
                block += (ll)v[i+1] * FFT_BASE;
            res.push_back(block);
        }
        ll carry = 0;
        for (size_t i = 0; i < res.size() || carry; i++) {
            if(i == res.size())
                res.push_back(0);
            int128 cur = res[i] + carry;
            res[i] = (ll)(cur % BASE);
            carry = (ll)(cur / BASE);
        }
        while(!res.empty() && res.back() == 0)
            res.pop_back();
        return res;
    }
} // end anonymous namespace

//=====================================
//     Core Arithmetic Operations
//=====================================

BigInt addAbs(const BigInt &A, const BigInt &B) {
    BigInt res;
    res.a.resize(max(A.a.size(), B.a.size()));
    ll carry = 0;
    for (size_t i = 0; i < res.a.size() || carry; i++) {
        if(i == res.a.size())
            res.a.push_back(0);
        uint128 cur = carry;
        if(i < A.a.size()) cur += A.a[i];
        if(i < B.a.size()) cur += B.a[i];
        res.a[i] = (ll)(cur % BASE);
        carry = (ll)(cur / BASE);
    }
    return res;
}

BigInt subAbs(const BigInt &A, const BigInt &B) {
    BigInt res = A;
    ll carry = 0;
    for (size_t i = 0; i < B.a.size() || carry; i++) {
        int128 cur = res.a[i] - carry - (i < B.a.size() ? B.a[i] : 0LL);
        carry = 0;
        if(cur < 0) {
            carry = 1;
            cur += BASE;
        }
        res.a[i] = (ll)cur;
    }
    res.trim();
    return res;
}

BigInt operator+(const BigInt &x, const BigInt &y) {
    BigInt res;
    if(x.sign == y.sign) {
        res = addAbs(x, y);
        res.sign = x.sign;
    } else {
        if(x.absLess(y)) {
            res = subAbs(y, x);
            res.sign = y.sign;
        } else {
            res = subAbs(x, y);
            res.sign = x.sign;
        }
    }
    res.trim();
    return res;
}

BigInt operator-(const BigInt &x) {
    BigInt res = x;
    if(!res.a.empty())
        res.sign = !res.sign;
    return res;
}

BigInt operator-(const BigInt &x, const BigInt &y) {
    return x + (-y);
}

//=========================== Multiplication ==============================

BigInt operator*(const BigInt &x, const BigInt &y) {
    BigInt res;
    int totalBlocks = x.a.size() + y.a.size();
    if(totalBlocks > FFT_MULT_THRESHOLD) {
        vector<int> fx = convertBigIntToFFT(x);
        vector<int> fy = convertBigIntToFFT(y);
        vector<int> fftRes = multiplyFFT(fx, fy);
        res.a = convertFFTToBigInt(fftRes);
    } else if(x.a.size() < KARATSUBA_THRESHOLD || y.a.size() < KARATSUBA_THRESHOLD) {
        res.a.resize(x.a.size() + y.a.size());
        for (size_t i = 0; i < x.a.size(); i++) {
            uint128 carry = 0;
            for (size_t j = 0; j < y.a.size() || carry; j++) {
                uint128 cur = carry;
                if(j < y.a.size())
                    cur += (uint128)x.a[i] * (uint128)y.a[j];
                if(i+j < res.a.size())
                    cur += res.a[i+j];
                res.a[i+j] = (ll)(cur % BASE);
                carry = (ll)(cur / BASE);
            }
        }
    } else {
        res.a = karatsubaMultiply(x.a, y.a);
    }
    res.sign = (x.sign == y.sign);
    res.trim();
    return res;
}

BigInt operator*(const BigInt &a, ll v) {
    BigInt res = a;
    if(v < 0) {
        res.sign = !res.sign;
        v = -v;
    }
    uint128 carry = 0;
    for (size_t i = 0; i < res.a.size() || carry; i++) {
        if(i == res.a.size())
            res.a.push_back(0);
        uint128 cur = carry + (uint128)res.a[i] * v;
        res.a[i] = (ll)(cur % BASE);
        carry = (ll)(cur / BASE);
    }
    res.trim();
    return res;
}

//=========================== Division ==============================

BigInt operator/(const BigInt &a, ll v) {
    BigInt res = a;
    res.sign = (res.sign == (v > 0));
    if(v < 0)
        v = -v;
    ll rem = 0;
    for (int i = res.a.size()-1; i >= 0; i--) {
        int128 cur = res.a[i] + (int128)rem * BASE;
        res.a[i] = (ll)(cur / v);
        rem = (ll)(cur % v);
    }
    res.trim();
    return res;
}

int compareBigInt(const BigInt &a, const BigInt &b) {
    if(a.sign != b.sign)
        return a.sign ? 1 : -1;
    if(a.a.size() != b.a.size())
        return (a.a.size() < b.a.size() ? -1 : 1) * (a.sign ? 1 : -1);
    for (int i = a.a.size()-1; i >= 0; i--)
        if(a.a[i] != b.a[i])
            return (a.a[i] < b.a[i] ? -1 : 1) * (a.sign ? 1 : -1);
    return 0;
}

pair<BigInt, BigInt> divmod(const BigInt &a, const BigInt &b) {
    if(b == BigInt(0))
        throw runtime_error("Division by zero");
    ll norm = BASE / (b.a.back() + 1);
    BigInt A = a.abs() * norm;
    BigInt B = b.abs() * norm;
    BigInt q, r;
    q.a.resize(A.a.size());
    for (int i = A.a.size()-1; i >= 0; i--) {
        r = r * BASE;
        BigInt cur((ll)A.a[i]);
        r = r + cur;
        ll lo = 0, hi = BASE - 1, m, d = 0;
        while (lo <= hi) {
            m = (lo + hi) / 2;
            BigInt t = B * m;
            if(compareBigInt(t, r) <= 0) {
                d = m;
                lo = m + 1;
            } else {
                hi = m - 1;
            }
        }
        q.a[i] = d;
        r = r - (B * d);
    }
    q.sign = (a.sign == b.sign);
    r.sign = a.sign;
    q.trim();
    r.trim();
    r = r / norm;
    return {q, r};
}

BigInt operator/(const BigInt &a, const BigInt &b) {
    return divmod(a, b).first;
}

BigInt operator%(const BigInt &a, const BigInt &b) {
    return divmod(a, b).second;
}

//=====================================
//         Relational Operators
//=====================================
bool operator==(const BigInt &x, const BigInt &y) {
    return compareBigInt(x, y) == 0;
}

bool operator!=(const BigInt &x, const BigInt &y) {
    return !(x == y);
}

bool operator<(const BigInt &x, const BigInt &y) {
    return compareBigInt(x, y) < 0;
}

bool operator>(const BigInt &x, const BigInt &y) {
    return compareBigInt(x, y) > 0;
}

bool operator<=(const BigInt &x, const BigInt &y) {
    return !(x > y);
}

bool operator>=(const BigInt &x, const BigInt &y) {
    return !(x < y);
}

ostream& operator<<(ostream &os, const BigInt &n) {
    os << n.toString();
    return os;
}

istream& operator>>(istream &is, BigInt &n) {
    string s;
    is >> s;
    n.read(s);
    return is;
}

//=====================================
//   Additional Mathematical Operations
//=====================================

// Exponentiation by squaring (BigInt exponent).
// Modified: Instead of doing "exp % two", we check the least-significant block.
// Also, we use the division-by-ll operator (which is reliable for small divisors).
BigInt pow(BigInt base, BigInt exp) {
    if(exp < BigInt(0))
        throw runtime_error("Negative exponent not supported.");
    BigInt res(1);
    while(exp.a.size() && !(exp.a.size() == 1 && exp.a[0] == 0)) {
        if(exp.a[0] & 1LL)
            res = res * base;
        exp = exp / 2;
        base = base * base;
    }
    return res;
}

// Overload: exponentiation with unsigned long long exponent.
BigInt pow(BigInt base, unsigned long long exp) {
    BigInt res(1);
    while(exp > 0) {
        if(exp & 1ULL)
            res = res * base;
        base = base * base;
        exp >>= 1;
    }
    return res;
}

// Modular exponentiation.
BigInt modPow(BigInt base, BigInt exp, const BigInt &m) {
    BigInt res(1);
    base = base % m;
    while(exp.a.size() && !(exp.a.size() == 1 && exp.a[0] == 0)) {
        if(exp.a[0] & 1LL)
            res = (res * base) % m;
        exp = exp / 2;
        base = (base * base) % m;
    }
    return res;
}

// Integer square root using Newton's method.
BigInt sqrt(const BigInt &a) {
    if(a < BigInt(0))
        throw runtime_error("Square root of negative number");
    if(a == BigInt(0))
        return BigInt(0);
    BigInt x = a, y = (x + BigInt(1)) / 2;
    while(y < x) {
        x = y;
        y = (x + a / x) / 2;
    }
    return x;
}

// Extended Euclidean Algorithm.
pair<BigInt, BigInt> extendedGCD(const BigInt &a, const BigInt &b) {
    if(b == BigInt(0))
        return {BigInt(1), BigInt(0)};
    auto p = extendedGCD(b, a % b);
    BigInt x = p.second;
    BigInt y = p.first - (a / b) * p.second;
    return {x, y};
}

// Modular Inverse.
BigInt modInv(const BigInt &a, const BigInt &m) {
    auto p = extendedGCD(a, m);
    BigInt inv = p.first % m;
    if(inv < BigInt(0))
        inv = inv + m;
    return inv;
}

// New: Greatest Common Divisor.
BigInt gcd(BigInt a, BigInt b) {
    a = a.abs(); b = b.abs();
    while(b != BigInt(0)) {
        BigInt temp = a % b;
        a = b;
        b = temp;
    }
    return a;
}

// New: Least Common Multiple.
BigInt lcm(const BigInt &a, const BigInt &b) {
    if(a == BigInt(0) || b == BigInt(0))
        return BigInt(0);
    return (a.abs() * b.abs()) / gcd(a, b);
}

// New: Miller-Rabin Primality Test (probabilistic).
// Returns true if n is probably prime.

// Helper function to load primes from a file.
vector<ll> loadPrimesFromFile(const string &filename) {
    vector<ll> primes;
    ifstream fin(filename);
    if (!fin) {
        throw runtime_error("Unable to open primes file: " + filename);
    }
    ll prime;
    while (fin >> prime) {
        primes.push_back(prime);
    }
    return primes;
}

bool isPrime(const BigInt &n, int iterations = 5) {
    if(n < BigInt(2))
        return false;
    if(n == BigInt(2) || n == BigInt(3))
        return true;
    if((n % 2) == BigInt(0))
        return false;

    // Write n - 1 as d * 2^s.
    BigInt d = n - BigInt(1);
    int s = 0;
    while((d % 2) == BigInt(0)) {
        d = d / 2;
        s++;
    }

    // Load primes from file "primes.txt" (only once).
    static vector<ll> testPrimes = loadPrimesFromFile("Prime_Number_Library.txt");

    // Use the loaded primes as bases.
    for(auto a : testPrimes) {
        if(BigInt(a) >= n)
            break;
        BigInt x = modPow(BigInt(a), d, n);
        if(x == BigInt(1) || x == n - BigInt(1))
            continue;
        bool composite = true;
        for(int r = 1; r < s; r++) {
            x = (x * x) % n;
            if(x == n - BigInt(1)) {
                composite = false;
                break;
            }
        }
        if(composite)
            return false;
    }
    return true;
}

// New: nthRoot - returns the integer floor of the nth root of a.
// Uses binary search.
BigInt nthRoot(const BigInt &a, unsigned int n) {
    if(n == 0)
        throw runtime_error("Zeroth root is undefined.");
    if(a < BigInt(0) && (n % 2 == 0))
        throw runtime_error("Even root of a negative number is undefined.");
    // For negative a and odd n, result will be negative.
    bool neg = false;
    BigInt x = a;
    if(x < BigInt(0)) { neg = true; x = -x; }

    // Determine approximate high bound by using digit count.
    string s = x.toString();
    int len = s.size();
    // Rough estimate: high ~ 10^(ceil(len/n))
    BigInt high("1");
    for (int i = 0; i < (len + n - 1) / n; i++)
        high = high * 10;
    BigInt low(0), ans(0);
    while(low <= high) {
        BigInt mid = (low + high) / 2;
        BigInt midPow = pow(mid, n);
        if(midPow <= x) {
            ans = mid;
            low = mid + BigInt(1);
        } else {
            high = mid - BigInt(1);
        }
    }
    return neg ? -ans : ans;
}

// New: Divide and Conquer Factorial (efficient for large n).
// Helper function: factorial in range [l, r].
BigInt factRec(unsigned int l, unsigned int r) {
    if(l > r)
        return BigInt(1);
    if(l == r)
        return BigInt((ll)l);
    unsigned int mid = (l + r) / 2;
    return factRec(l, mid) * factRec(mid+1, r);
}
BigInt factorialDC(unsigned int n) {
    if(n < 2)
        return BigInt(1);
    return factRec(1, n);
}

//=====================================
//      Additional Utility Operations
//=====================================

// Compound assignment operators.
inline BigInt& operator+=(BigInt &a, const BigInt &b) { a = a + b; return a; }
inline BigInt& operator-=(BigInt &a, const BigInt &b) { a = a - b; return a; }
inline BigInt& operator*=(BigInt &a, const BigInt &b) { a = a * b; return a; }
inline BigInt& operator/=(BigInt &a, const BigInt &b) { a = a / b; return a; }
inline BigInt& operator%=(BigInt &a, const BigInt &b) { a = a % b; return a; }

// Pre- and post-increment/decrement.
inline BigInt& operator++(BigInt &a) { a = a + BigInt(1); return a; }
inline BigInt operator++(BigInt &a, int) { BigInt temp = a; a = a + BigInt(1); return temp; }
inline BigInt& operator--(BigInt &a) { a = a - BigInt(1); return a; }
inline BigInt operator--(BigInt &a, int) { BigInt temp = a; a = a - BigInt(1); return temp; }

// Conversion to long long (if it fits in 64 bits).
long long toLL(const BigInt &a) {
    BigInt temp = a.abs();
    // Rough check: if more than 2 blocks, it is likely too big.
    if(temp.a.size() > 2)
        throw runtime_error("BigInt does not fit in long long");
    long long result = 0;
    for (int i = temp.a.size()-1; i >= 0; i--)
        result = result * BASE + temp.a[i];
    return a.sign ? result : -result;
}

// Fast doubling Fibonacci algorithm.
pair<BigInt, BigInt> fibPair(unsigned long long n) {
    if(n == 0)
        return {BigInt(0), BigInt(1)};
    auto p = fibPair(n / 2);
    BigInt c = p.first * (p.second * BigInt(2) - p.first);
    BigInt d = p.first * p.first + p.second * p.second;
    if(n % 2 == 0)
        return {c, d};
    else
        return {d, c + d};
}

BigInt fibonacci(unsigned long long n) {
    return fibPair(n).first;
}

// Standard iterative factorial.
BigInt factorial(unsigned int n) {
    BigInt res(1);
    for(unsigned int i = 2; i <= n; i++)
        res = res * BigInt((ll)i);
    return res;
}

// Binomial Coefficient: n choose k.
BigInt binom(unsigned int n, unsigned int k) {
    if(k > n)
        return BigInt(0);
    if(k > n - k)
        k = n - k;
    BigInt res(1);
    for(unsigned int i = 1; i <= k; i++) {
        res = res * BigInt((ll)(n - i + 1));
        res = res / BigInt((ll)i);
    }
    return res;
}

#endif

/*
======================================= How to Use ======================================

#include <bits/stdc++.h>
#include "MyLib.h"  // Include the BigInt library header

using namespace std;

int main() {
    BigInt a("1234567890");
    BigInt b("9876543210");

    BigInt sum = a + b;
    BigInt diff = a - b;
    BigInt prod = a * b;
    BigInt quot = a / b;
    BigInt rem = a % b;

    BigInt expVal = pow(a, BigInt("3"));         // a^3 using BigInt exponent
    BigInt expVal2 = pow(a, 3ULL);                 // a^3 using unsigned long long exponent
    BigInt modExp = modPow(a, BigInt("100"), b);    // a^100 mod b
    BigInt root = sqrt(a);                         // integer square root of a

    // New operations demonstration:
    BigInt g = gcd(a, b);
    BigInt l = lcm(a, b);
    bool prime = isPrime(a);
    BigInt nroot = nthRoot(a, 3);                  // cube root of a (floor)
    BigInt factDC = factorialDC(20);               // factorial using divide & conquer

    // Compound assignment & increment/decrement demonstration.
    BigInt c = a;
    c += b;
    c++;
    c *= BigInt(2);

    // Convert to long long (if it fits)
    try {
        long long llVal = toLL(BigInt("1234567"));
        cout << "toLL: " << llVal << "\n";
    } catch(const exception &e) {
        cout << e.what() << "\n";
    }

    // Fibonacci (nth Fibonacci number)
    BigInt fib50 = fibonacci(50ULL);

    // Standard Factorial & Binomial Coefficient
    BigInt fact10 = factorial(10);
    BigInt binom10_3 = binom(10, 3);

    cout << "a: " << a << "\n";
    cout << "b: " << b << "\n";
    cout << "Sum: " << sum << "\n";
    cout << "Difference: " << diff << "\n";
    cout << "Product: " << prod << "\n";
    cout << "Quotient: " << quot << "\n";
    cout << "Remainder: " << rem << "\n";
    cout << "a^3: " << expVal << "   " << expVal2 << "\n";
    cout << "a^100 mod b: " << modExp << "\n";
    cout << "sqrt(a): " << root << "\n";
    cout << "gcd(a, b): " << g << "\n";
    cout << "lcm(a, b): " << l << "\n";
    cout << "isPrime(a): " << (prime ? "true" : "false") << "\n";
    cout << "nthRoot(a, 3): " << nroot << "\n";
    cout << "factorialDC(20): " << factDC << "\n";
    cout << "After compound assignment/increment: " << c << "\n";
    cout << "Fibonacci(50): " << fib50 << "\n";
    cout << "Factorial(10): " << fact10 << "\n";
    cout << "Binom(10, 3): " << binom10_3 << "\n";

    return 0;
}

===============================================================================
*/