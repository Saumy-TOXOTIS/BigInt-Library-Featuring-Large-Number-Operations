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
- Handles numbers with up to 10^10 digits with ease.
- Flexible input from strings, long longs, and more.
- Built-in sign handling for positive and negative numbers.
- Easy-to-use interface to convert between BigInt and strings.

With this library, you can perform operations on huge integers without worrying 
about overflow, and all with efficient block-based methods to handle even 
the largest values. 

Get started with the BigInt library and work with numbers on a whole new scale!
===============================================================================
*/

#include <bits/stdc++.h>
using namespace std;
using ll = long long;

#ifndef MyLib_H
#define MyLib_H

// Each block holds up to 9 decimal digits.
const ll BASE = 1000000000000LL;
const int BASE_DIGITS = 12;

//===================================== Core Structure ========================================

struct BigInt {
    // The BigInt is stored in a "little-endian" format:
    // This means that the least-significant block of the number is stored at index 0.
    vector<ll> a;  
    // The 'sign' boolean indicates the sign of the number:
    // 'true' means the number is non-negative (positive or zero),
    // 'false' means the number is negative.
    bool sign;

    // Default Constructor:
    // Initializes a BigInt with a value of 0 (non-negative).
    BigInt() : sign(true) {}

    // Constructor from long long (ll):
    // Initializes the BigInt with a long long value.
    BigInt(ll v) { *this = v; }

    // Constructor from int:
    // Initializes the BigInt with an integer value, converting it to long long.
    BigInt(int v) { *this = (ll)v; }

    // Constructor from string:
    // Reads the BigInt value from a string, handling optional signs ('+' or '-') as well.
    BigInt(const string &s) { read(s); }

    // Assignment from long long:
    // Converts a long long integer to a BigInt and stores it in the 'a' vector in little-endian format.
    void operator=(ll v) {
        sign = (v >= 0);  // Set the sign based on the value of 'v'.
        if(v < 0) v = -v;  // If the number is negative, make it positive temporarily.
        a.clear();  // Clear any previous data.
        while(v > 0) {
            a.push_back(v % BASE);  // Store the current block (up to BASE value).
            v /= BASE;  // Move to the next block by dividing by BASE.
        }
    }

    // Read from string:
    // Reads a string representation of the number, correctly handling the sign and dividing it into blocks.
    // The number is split into blocks of up to BASE_DIGITS digits each.
    void read(const string &s) {
        sign = true;  // Assume the number is positive initially.
        a.clear();  // Clear any existing data.
        int pos = 0;
        
        // Handle optional sign ('+' or '-'). If present, set the sign accordingly.
        while(pos < (int)s.size() && (s[pos] == '-' || s[pos] == '+')) {
            if(s[pos] == '-') sign = false;  // Negative sign found.
            pos++;
        }

        // Process the string from right to left, extracting blocks of up to BASE_DIGITS digits.
        for (int i = s.size()-1; i >= pos; i -= BASE_DIGITS) {
            ll x = 0;  // Initialize the current block.
            int start = max(pos, i - BASE_DIGITS + 1);  // Start of the current block.
            for (int j = start; j <= i; j++)  // Convert the digits into a number.
                x = x * 10 + (s[j] - '0');  // Build the block.
            a.push_back(x);  // Add the block to the vector.
        }
        trim();  // Remove leading zeros if any.
    }

    // Remove leading zero blocks:
    // This function trims any unnecessary zero blocks from the vector.
    // If the BigInt becomes empty, set it to 0 (non-negative).
    void trim() {
        while(!a.empty() && a.back() == 0)  // Remove zero blocks from the back.
            a.pop_back();
        if(a.empty())  // If the vector is empty, the number is 0.
            sign = true;  // Set the sign to positive (0 is non-negative).
    }

    // Convert BigInt to string:
    // Converts the BigInt back to a string for easy printing.
    // It adds the sign (if negative) and ensures the formatting of each block.
    string toString() const {
        if(a.empty())  // If the number is 0, return "0".
            return "0";
        stringstream ss;
        if(!sign)  // If the number is negative, add the '-' sign.
            ss << "-";
        ss << a.back();  // Add the most significant block.
        // Add the remaining blocks, ensuring each one is padded with leading zeros as needed.
        for (int i = (int)a.size()-2; i >= 0; i--)
            ss << setw(BASE_DIGITS) << setfill('0') << a[i];
        return ss.str();
    }

    // Absolute value:
    // Returns a new BigInt that represents the absolute value of the current BigInt.
    BigInt abs() const {
        BigInt res = *this;  // Copy the current BigInt.
        res.sign = true;  // Set the sign to positive.
        return res;
    }

    // Compare absolute values:
    // Compares the absolute values of the current BigInt ('this') and another BigInt ('other').
    // Returns true if the absolute value of 'this' is less than the absolute value of 'other'.
    bool absLess(const BigInt &other) const {
        if(a.size() != other.a.size())  // If the sizes differ, the smaller one is less.
            return a.size() < other.a.size();
        for (int i = (int)a.size()-1; i >= 0; i--) {
            if(a[i] != other.a[i])  // Compare block by block.
                return a[i] < other.a[i];  // Return true if 'this' is less.
        }
        return false;  // The absolute values are equal.
    }
};

//===================================== Arithematic Part Logic ========================================

//=========================== Addition ==============================

// Addition of absolute values.
BigInt addAbs(const BigInt &A, const BigInt &B) {
    BigInt res;
    res.a.resize(max(A.a.size(), B.a.size()));
    ll carry = 0;
    for (size_t i = 0; i < res.a.size() || carry; i++) {
        if(i == res.a.size())
            res.a.push_back(0);
        ll cur = carry;
        if(i < A.a.size()) cur += A.a[i];
        if(i < B.a.size()) cur += B.a[i];
        carry = cur >= BASE;
        if(carry) cur -= BASE;
        res.a[i] = cur;
    }
    return res;
}

//=========================== Subtraction ==============================

// Subtraction of absolute values (assumes |A| >= |B|).
BigInt subAbs(const BigInt &A, const BigInt &B) {
    BigInt res = A;
    ll carry = 0;
    for (size_t i = 0; i < B.a.size() || carry; i++) {
        res.a[i] -= carry + (i < B.a.size() ? B.a[i] : 0);
        carry = res.a[i] < 0;
        if(carry)
            res.a[i] += BASE;
    }
    res.trim();
    return res;
}

// Operator +.
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

// Unary minus.
BigInt operator-(const BigInt &x) {
    BigInt res = x;
    if(!res.a.empty())
        res.sign = !res.sign;
    return res;
}

// Operator -.
BigInt operator-(const BigInt &x, const BigInt &y) {
    return x + (-y);
}

//=========================== Multiplications ==============================

// Naïve multiplication (O(n*m)). For huge numbers, consider Karatsuba or FFT-based methods.
BigInt operator*(const BigInt &x, const BigInt &y) {
    BigInt res;
    res.a.resize(x.a.size() + y.a.size());
    for (size_t i = 0; i < x.a.size(); i++) {
        ll carry = 0;
        for (size_t j = 0; j < y.a.size() || carry; j++) {
            ll cur = res.a[i+j] +
                     x.a[i] * (j < y.a.size() ? y.a[j] : 0LL) + carry;
            res.a[i+j] = cur % BASE;
            carry = cur / BASE;
        }
    }
    res.sign = (x.sign == y.sign);
    res.trim();
    return res;
}

// Multiplication of BigInt by an ll.
BigInt operator*(const BigInt &a, ll v) {
    BigInt res = a;
    if(v < 0) {
        res.sign = !res.sign;
        v = -v;
    }
    ll carry = 0;
    for (size_t i = 0; i < res.a.size() || carry; i++) {
        if(i == res.a.size())
            res.a.push_back(0);
        ll cur = carry + res.a[i] * v;
        res.a[i] = cur % BASE;
        carry = cur / BASE;
    }
    res.trim();
    return res;
}

// Division of BigInt by an ll.
BigInt operator/(const BigInt &a, ll v) {
    BigInt res = a;
    res.sign = (res.sign == (v > 0));
    if(v < 0)
        v = -v;
    ll rem = 0;
    for (int i = res.a.size()-1; i >= 0; i--) {
        ll cur = res.a[i] + rem * BASE;
        res.a[i] = cur / v;
        rem = cur % v;
    }
    res.trim();
    return res;
}

// Helper: compare two BigInts. Returns -1 if a < b, 0 if a == b, 1 if a > b.
int compareBigInt(const BigInt &a, const BigInt &b) {
    if(a.sign != b.sign)
        return a.sign ? 1 : -1;
    if(a.a.size() != b.a.size())
        return (a.a.size() < b.a.size() ? -1 : 1) * (a.sign ? 1 : -1);
    for (int i = a.a.size()-1; i >= 0; i--) {
        if(a.a[i] != b.a[i])
            return (a.a[i] < b.a[i] ? -1 : 1) * (a.sign ? 1 : -1);
    }
    return 0;
}

//=========================== Division ==============================

// Division: returns a pair (quotient, remainder).
// This implements block‑based long division using normalization.
// (For huge numbers, consider Newton–Raphson based methods.)
pair<BigInt, BigInt> divmod(const BigInt &a, const BigInt &b) {
    // It is assumed that b is nonzero.
    ll norm = BASE / (b.a.back() + 1);
    BigInt A = a.abs() * norm;
    BigInt B = b.abs() * norm;
    BigInt q, r;
    q.a.resize(A.a.size());
    // Process blocks from most-significant to least-significant.
    for (int i = A.a.size()-1; i >= 0; i--) {
        // r = r * BASE + current block.
        r = r * BASE;
        BigInt cur((ll)A.a[i]);
        r = r + cur;
        // Binary search for the maximum digit d such that B * d <= r.
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
    // Unnormalize the remainder.
    r = r / norm;
    return {q, r};
}

BigInt operator/(const BigInt &a, const BigInt &b) {
    return divmod(a, b).first;
}

BigInt operator%(const BigInt &a, const BigInt &b) {
    return divmod(a, b).second;
}

#endif

/*
======================================= How to Use ======================================

#include <bits/stdc++.h>
#include "MyLib.h"  // Include the BigInt library header

using namespace std;

int main() {
    -- Create BigInt objects from strings.
    BigInt a("12345678901234567890");
    BigInt b("9876543210");

    // Perform arithmetic operations.
    BigInt sum = a + b;
    BigInt diff = a - b;
    BigInt prod = a * b;
    BigInt quot = a / b;
    BigInt rem = a % b;

    // Output the results.
    cout << "a: " << a.toString() << endl;
    cout << "b: " << b.toString() << endl;
    cout << "Sum: " << sum.toString() << endl;
    cout << "Difference: " << diff.toString() << endl;
    cout << "Product: " << prod.toString() << endl;
    cout << "Quotient: " << quot.toString() << endl;
    cout << "Remainder: " << rem.toString() << endl;

    return 0;
}

===============================================================================
*/