#include "tree.h"

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

struct tag_node *create_node()
{
  struct tag_node *node = (struct tag_node *) malloc(sizeof(struct tag_node));
  node->next_sib = NULL;
  node->parent = NULL;
  node->arb_child = NULL;
}

/*
 * Destroy the subtree.
 * Do not destroy the siblings of the root node.
 * Rewire the parent node appropriately.
 */
void recursively_destroy_subtree(struct tag_node *root)
{
  /* if the root is null then we are already finished */
  if (!root)
  {
    return;
  }

  /*
   * If the parent node connects to this root node,
   * then reconnect it to a different arbitrary child node if the parent
   * has more than one child, otherwise mark the arbitrary child as null.
   */
  if (root->parent)
  {
    if (root->next_sib != root)
    {
      root->parent->arb_child = root->next_sib;
    } else {
      root->parent->arb_child = NULL;
    }
  }

  /*
   * If the root of the subtree to be deleted
   * is connected to a sibling node,
   * then remove the root from this cycle.
   */

  /* If the root of the subtree to be deleted has multiple
  if (root->arb_child)
  {
    struct tag_node *tmp = root;
  }
}


int recursively_count_nodes_in_tree(struct tag_node *root)
{
  if (!root)
  {
    return 0;
  }
  /* accumulate a count of the nodes */
  int nnodes = 0;
  /* initialize the node pointer */
  struct tag_node *tmp = root;
  /* loop through all circularly linked siblings of the node pointer */
  while (1)
  {
    if (tmp->arbitrary_child)
    {
      /* add all child nodes of the current sib node */
      nnodes += recursively_count_nodes_in_tree(tmp->arbitrary_child);
    }
    /* add the node that corresponds to the current sib */
    nnodes += 1;
    /* update the pointer to the next sibling in the ring of sib nodes */
    tmp = tmp->next_sib;
    if (tmp == root)
    {
      return nnodes;
    }
  }
}


/*
 * the input array should is expected to have been preallocated
 * to hold the right number of pointers.
 * The initial index passed should be zero.
 * This is recursive.
 * Return the current index.
 * The value returned from the top level should be
 * the preallocated number nnodes.
 */
int build_dfs_node_array(
    struct tag_node *root, struct tag_node *arr, int current_index)
{
  /* initialize the node pointer */
  struct tag_node *tmp = root;
  /* loop through all circularly linked siblings of the node pointer */
  while (1)
  {
    /* add the node that corresponds to the current sib */
    nnodes += 1;
    if (tmp->arbitrary_child)
    {
      /* add all child nodes of the current sib node */
      nnodes += recursively_count_nodes_in_tree(tmp->arbitrary_child);
    }
    /* update the pointer to the next sibling in the ring of sib nodes */
    tmp = tmp->next_sib;
    if (tmp == root)
    {
      return nnodes;
    }
  }
  current_index += build_dfs_node_array(root, arr, current_index + 1);
  return current_index
}


/* 
 * For building a circularly linked list in dfs order.
 * The last (or equivalently prev from first) node will be the root.
 * tnl means tag_node_list.
 */
struct tnl
{
  struct tag_node *node;
  struct tag_node *next;
  struct tag_node *prev;
};


/*
 * Build or extend the dfs list of nodes.
 */
struct tnl *extend_dfs_list(
    struct tag_node *root,
    struct tnl **phead, struct tnl **ptail)
{
  if (!head)
  {
    head = foo;
  }
  if (!tail)
  {
    current = 
  }

  struct tnl
  struct tag_node_list *list = (struct tag_node_list *) malloc(
      sizeof(struct tag_node_list *));
}

int bisect_edges(tag_node *root)
{
  struct tag_node *tmp = root;
}

int recursively_count_nodes_in_tree(struct tag_node *root)
{}

