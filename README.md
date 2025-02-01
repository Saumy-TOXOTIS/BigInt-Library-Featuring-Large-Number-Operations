# 🚀 BigInt Library Featuring Large Number Operations

A C++ BigInt Library designed to handle extremely large numbers efficiently, allowing seamless operations on numbers up to 10^10 digits. Whether you're working with cryptographic applications, scientific calculations, or large-scale data analysis, this library provides the precision and performance you need!

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

---

## 📈 Example Usage:

### Example Code to Use BigInt Library:

```cpp
#include <iostream>
#include "MyLib.h"  -- Include the BigInt library header

using namespace std;

int main() {
    -- Create BigInt objects from strings.
    BigInt a("12345678901234567890");
    BigInt b("9876543210");

    -- Perform arithmetic operations.
    BigInt sum = a + b;
    BigInt diff = a - b;
    BigInt prod = a * b;
    BigInt quot = a / b;
    BigInt rem = a % b;

    -- Output the results.
    cout << "a: " << a.toString() << "\n";
    cout << "b: " << b.toString() << "\n";
    cout << "Sum: " << sum.toString() << "\n";
    cout << "Difference: " << diff.toString() << "\n";
    cout << "Product: " << prod.toString() << "\n";
    cout << "Quotient: " << quot.toString() << "\n";
    cout << "Remainder: " << rem.toString() << "\n";

    return 0;
}
