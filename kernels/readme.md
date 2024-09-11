# Kernels Summary
These kernels do a simple convolution based on the implementation of Yflows.
Pointers are declared as restrict in the kernels.

## Kernel 01

 - Filter of 3x3
 - Base has no knowledge of loop trip counts
 - Opt knows their constant sizes
 - In Yflows, it seems that the baseline they compared also had no knowledge of constants and their optimized version did

## Kernel 02
 - Filter of 5x5
 - Otherwise same as Kernel 01

## Kernel 03

 - Filter of 3x3
 - The base is the opt version of Kernel 01
 - The opt adds to the base pragmas to fully unroll the filter loops

## Kernel 04

 - Filter of 5x5
 - Otherwise same as Kernel 03

