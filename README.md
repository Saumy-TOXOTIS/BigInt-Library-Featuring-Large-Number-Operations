# 🚀 BigInt Library Featuring Large Number Operations

A C++ BigInt Library designed to handle extremely large numbers efficiently, allowing seamless operations on numbers up to **10^13 digits**. This library provides the precision and performance!

![C++](https://img.shields.io/badge/C%2B%2B-11%2B-blue.svg?style=flat-square&logo=c%2B%2B)
![GitHub](https://img.shields.io/badge/Repo-GitHub-black.svg?style=flat-square&logo=github)

---

## 🏗 Library Features

| Feature                      | Description                                                         |
|------------------------------|---------------------------------------------------------------------|
| **Arithmetic Operations**     | Addition, Subtraction, Multiplication, Division, Modulo            |
| **Base Representation**      | Optimized for **BASE = 10^9** to efficiently handle large numbers  |
| **Block-Based Approach**     | High-performance BigInt handling with block-based techniques       |
| **Sign Handling**            | Proper management of both **positive** and **negative** numbers    |
| **String Conversion**        | Easy conversion between BigInts and strings for flexible I/O       |
| **Optimized Multiplication** | Uses **Karatsuba** and **FFT-based convolution** for efficiency    |
| **Exponentiation**           | Supports **BigInt exponents** and **modular exponentiation**       |
| **GCD & LCM**                | Compute **Greatest Common Divisor** and **Least Common Multiple**  |
| **Prime Testing**            | Miller-Rabin primality test for **fast prime verification**        |
| **Advanced Math Functions**  | Square root, nth-root, Euler’s totient, factorial, Fibonacci       |
| **Number Properties**        | Check for **perfect squares, palindromes, and digital roots**      |
| **Factorization**            | **Pollard-Rho** algorithm for prime factorization                  |
| **Logarithms**               | **Floor logarithm** for large integer bases                        |
| **Collatz Sequence**         | **Generate Collatz Sequence** for desired number ( for positive integers only)                   |

---

## 🛠 Additional Features

- **Compound Operators**: Supports `+=, -=, *=, /=, %=` for seamless arithmetic.
- **Pre/Post Increment & Decrement**: Just like built-in integers.
- **Conversion to long long**: If the BigInt fits within a standard integer type.
- **Modular Square Root**: Computes square roots under a modulus.
- **Binomial Coefficients**: Efficient calculation for combinatorial operations.
- **Efficient Factorial Computation**: Uses **Divide & Conquer factorial** algorithm.
- **Flexible Input Handling**: Accepts numbers from **strings, integers, and other formats**.

---

## 📈 Example Usage:

### Example Code to Use BigInt Library:

```cpp
#include <bits/stdc++.h>
#include "MyLib.h"  // Include the BigInt library header

using namespace std;

int main() {
    // Create BigInt objects from strings.
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
    BigInt factor = pollardRho(b);                 // a non-trivial factor of b
    unsigned int flog = floorLog(a, 10);           // floor(log_10(a))
    BigInt msqrt = modSqrt(BigInt("56"), BigInt("101")); 
    BigInt tot = totient(a);                       // Euler's totient of a
    vector<BigInt> pf = primeFactors(a);           // prime factors of a
    BigInt sdig = sumDigits(a);                    // sum of digits of a
    BigInt droot = digitalRoot(a);                 // digital root of a
    bool psq = isPerfectSquare(a);                 // is a perfect square?
    bool pal = isPalindrome(a);                    // is a palindrome?

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
    cout << "pollardRho(b): " << factor << "\n";
    cout << "floorLog(a, 10): " << flog << "\n";
    cout << "modSqrt(56, 101): " << msqrt << "\n";
    cout << "totient(a): " << tot << "\n";
    cout << "primeFactors(a): ";
    for(auto &pfac : pf)
        cout << pfac << " ";
    cout << "\n";
    cout << "sumDigits(a): " << sdig << "\n";
    cout << "digitalRoot(a): " << droot << "\n";
    cout << "isPerfectSquare(a): " << (psq ? "true" : "false") << "\n";
    cout << "isPalindrome(a): " << (pal ? "true" : "false") << "\n";
    cout << "After compound assignment/increment: " << c << "\n";
    cout << "Fibonacci(50): " << fib50 << "\n";
    cout << "Factorial(10): " << fact10 << "\n";
    cout << "Binom(10, 3): " << binom10_3 << "\n";

    // Collatz Sequence Generator
    for(auto e : collatzSequence(a))
    {
        cout<<e<<endl;
    }

    return 0;
}
```

---
## 🚀 Contributions

I welcome contributions from the community! 🌱 Whether it's a feature request, a bug fix, or an enhancement, we’re excited to see your ideas. Feel free to fork the repository, submit a pull request, and help make this library even better!
