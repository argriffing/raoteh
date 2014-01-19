/*
 * Rao-Teh sampler in C.
 * This is for a specific statistical model
 * related to the human genetics of cancer.
 */


  /* Read the options, or rather use a control file.
   * - tree, using python csr format;
   *   http://docs.scipy.org/doc/scipy/reference/generated/
   *   scipy.sparse.csr_matrix.html; several arrays in the same file
   *   - indptr (1d array of indices -- pointers into an array)
   *   - indices (1d array of indices -- child node indices)
   *   - data (1d array of indices -- branch lengths)
   *   - preorder (1d array of indices -- first index corresponds to the root)
   * - tolerance process parameters
   *   - rate off to on (a floating point number)
   *   - rate on to off (a floating point number)
   * - primary process parameters
   *   - primary process rate matrix (2d array of floating point numbers)
   *   - prior distribution at the root (1d array of floating point numbers)
   * - connection between the primary process and the tolerance processes
   *   - map from primary state to tolerance class
   * - data; for every site in the alignment, for every node in the tree
   *   - set of allowed primary states
   *   - for each tolerance class, the set of allowed tolerance states
   */


#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <gsl/gsl_rng.h>

int read_primary_rate_matrix(FILE *fin)
{
}

int main(int argc, char **argv)
{
  return 0;
}

