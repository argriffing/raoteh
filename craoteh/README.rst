Primary process rate matrix data format
=======================================

The rate matrix should be easy to read but not too binary.
Let the first line have the number of states N,
where the states are numbered 0 through N-1.
Let the rest of the rate matrix file consist of whitespace (tab) separated
triples on each line, where the triple is (source_state, sink_state, rate).
Missing entries will be assumed to be zero,
and the diagonal entries of the rate matrix will be determined automatically.

