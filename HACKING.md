
For development, use the `runtests.py` script.
By default, the tests marked `@slow` are not run.
To enable a specific slow test for profiling,
use a command like the following:

```
python -m cProfile -o tmjp.profile runtests.py -t raoteh/sampler/tests/test_sample_tmjp.py:test_sample_tmjp_v1
```

This command will create a file called `tmjp.profile` that contains
some profiling information.
This file is verbose because it contains information about many
irrelevant functions in the `scipy` and `networkx` packages.
The following blog post shows how to use the `pstats` interactive mode
to sift through this file for the relevant timing information.

http://stefaanlippens.net/python_profiling_with_pstats_interactive_mode

