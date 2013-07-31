#include <assert>

#include "tree.h"

struct tag_node *next_sib; /* next sibling in the ring of siblings */
struct tag_node *parent; /* parent of the current sib */
struct tag_node *arb_child; /* arbitrary child fo the current sib */

int assert_int_equal(long a, long b, char *s)
{
  if (a != b)
  {
    printf("FAIL\n\n");
    printf("%s\n", s);
    printf("actual:%ld\n", a);
    printf("desired:%ld\n", b);
    abort();
  }
}

int test_recursively_count_nodes_in_tree()
{
  int nnodes;
  struct tag_node *root;

  /* test an empty test tree */
  root = (struct tag_node *) 0;
  nnodes_actual = recursively_count_nodes_in_tree(root);
  nnodes_desired = 0;
  assert_int_equal(nnodes_actual, nnodes_desired, "empty tree");

  /* test a tree with only a root */
  root = create_node();
  root->next_sib = root;
  nnodes_actual = recursively_count_nodes_in_tree(root);
  nnodes_desired = 1;
  assert_int_equal(nnodes_actual, nnodes_desired, "tree with only a root");
  free(root);

  /* test a path tree */
}


int main(int argc, char **argv)
{
  /* Allocate a small tree. */
  return 0;
}

