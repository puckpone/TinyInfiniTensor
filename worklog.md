# 1.27
I met a error when make:
```
make: *** No rule to make target `format', needed by`.PHONY'Stop.
```
The reason is the command 'make' will use the 'ninja' as default, the solution is as follows:
```
GNU "make" has been installed as "gmake""

If you need to use it as "make", you can add a "gnubin" directory to your PATH from your bashrc like:

PATH="/opt/homebrew/opt/make/libexec/gnubin:$PATH"
```

# 1.29

completed allocator


# 1.30

complete cast, clip, concat, matmul, transpose.

TODO: learn the graph optimizing rule, through videos and ppts.

# 2.1
I try so hard😭😭

73% tests passed, 3 tests failed out of 11

Total Test time (real) =   3.54 sec

The following tests FAILED:

          9 - test_nativecpu_concat (Subprocess aborted)

         10 - test_nativecpu_elementwise (SEGFAULT)

         11 - test_nativecpu_transpose (SEGFAULT)

---
all tests passed!

the main reasons are: alloc, dataMalloc
