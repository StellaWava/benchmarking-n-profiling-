This project goal is reconstructing knowledge continuity for high performance scientific computing through:
- map an algorithm's dependence structure
- identify its dwarf class and data type 
- outline its bottleneck
- measure its roofline position
- carry that reasoning upward into LLM execution, training, and finetuning

The central line is simple:

> The dependence graph is the dwarf. Fix the dependence structure of an algorithm and you have already determined its available parallelism, arithmetic intensity, roofline position, and the system level it belongs on.

# Identifying the dwarf 
The Berkeley View created over 20 years ago, summarizes data movement and the computational bottlenecks through 13 dwarfs. All these dwarf reintroduce the mathematical foundations unto which all the algorithms and all the current computer architecture and software sit, i.e. sparse Matrices, Dense Matrices, the ..... The dwarfing goal was to highlight the bottlenecks faced with the current computing world with some structure. 

Our project is exploiting only 3 of the 12 in the effort to reproduce in an indexed manner, not only their genesis, but how they've influenced high performance computing and innovation for over a century. These include; 
a) STREAM TRIAD (C = A + B*scalar; one of the simplest sum-product vector updates)
b) Sparse Matrix Vector (y = Ax; where A is a sparse vector)
c) General Matrix Multiply (C = xAB + nC; a grid product-sum update)
d) Fast Fourier Transform -FFT (.........; BONUS: as is one of the complex application for the engine)
e) Combinatorial Logic (TBD)


# Snapshot of computing infrastructure for the last 20 years -including bottlenecks


# Measuring roofline 



