/*
 * Rao-Teh sampler in C.
 * This is for a specific statistical model
 * related to the human genetics of cancer.
 */

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <gsl/gsl_rng.h>

int main(int argc, char **argv)
{
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
  const gsl_rng_type * T;
  gsl_rng * r;

  int i, n = 10;

  gsl_rng_env_setup();

  T = gsl_rng_default;
  r = gsl_rng_alloc (T);

  for (i = 0; i < n; i++) 
    {
      double u = gsl_rng_uniform (r);
      printf ("%.5f\n", u);
    }

  gsl_rng_free (r);

  return 0;
}


int foo(int argc, char **argv)
{
  int aflag = 0;
  int bflag = 0;
  char *cvalue = NULL;
  int index;
  int c;

   opterr = 0;
 
   while ((c = getopt (argc, argv, "abc:")) != -1)
     switch (c)
       {
       case 'a':
         aflag = 1;
         break;
       case 'b':
         bflag = 1;
         break;
       case 'c':
         cvalue = optarg;
         break;
       case '?':
         if (optopt == 'c')
           fprintf (stderr, "Option -%c requires an argument.\n", optopt);
         else if (isprint (optopt))
           fprintf (stderr, "Unknown option `-%c'.\n", optopt);
         else
           fprintf (stderr,
                    "Unknown option character `\\x%x'.\n",
                    optopt);
         return 1;
       default:
         abort ();
       }
 
   printf ("aflag = %d, bflag = %d, cvalue = %s\n",
           aflag, bflag, cvalue);
 
   for (index = optind; index < argc; index++)
     printf ("Non-option argument %s\n", argv[index]);
   return 0;
 }

