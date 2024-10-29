# Pushing Cython to its Limits in Scikit-learn

[Link to slides](https://thomasjpfan.github.io/pydata-nyc-2024-cython-in-scikit-learn/)

scikit-learn is a machine-learning library for Python that uses NumPy and SciPy for numerical operations. Scikit-learn has its own compiled code for performance-critical computation written in C, C++, and Cython. The library primarily focuses on Cython for compiled code because it is easy to use and approachable. In this talk, we dive into many techniques scikit-learn employs to utilize Cython fully. We will cover features like using the C++ standard library within Cython, fused types, code generation with the Tempita engine, and OpenMP for parallelization.

## License

This repo is under the [MIT License](LICENSE).
