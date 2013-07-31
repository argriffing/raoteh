/*
 * The branch length of a node refers to the length of the branch
 * connecting to the parent node.
 * The root node does not necessarily have a meaningful branch length.
 */

struct tag_node
{
  int primary_state;
  int tolerance_mask;
  struct tag_node *next_sib; /* next sibling in the ring of siblings */
  struct tag_node *parent; /* parent of the current sib */
  struct tag_node *arb_child; /* arbitrary child fo the current sib */
  int is_skeleton_node;
  double branch_length;
};

struct tag_node *create_node();

