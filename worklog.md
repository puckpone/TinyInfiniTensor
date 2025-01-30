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