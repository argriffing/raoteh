"""
Tree layout.

There must be a way to do this without implementing it myself...

"""
from __future__ import division, print_function


def layout_tree(root, leaf_to_y=None):
    distance = 0
    nleaves = 0
    vert_out = []
    horz_out = []
    nodes_out = []
    rlayout(root, distance, nleaves, leaf_to_y, vert_out, horz_out, nodes_out)
    return vert_out, horz_out, nodes_out


def rlayout(root, x, nleaves, leaf_to_y, vert_out, horz_out, nodes_out):
    """
    The input has a very specific format.
    Each node of the rooted tree is a (handle, blen, child_nodes) sequence.
    This is defined recursively.
    The handle can be anything, or maybe required to be hashable.
    The blen is the branch length which is None for the root
    and which is a non-negative float for other nodes.
    The child_nodes is a sequence of child nodes.
    The x argument is the distance from the root.
    The nleaves arg is the number of leaves observed already.
    The leaf_to_y arg maps each leaf to a y coordinate.

    The output is similarly picky.
    The vert_out is a sequence of (x, y1, y2, handle) lines.
    The horz_out is a sequence of (y, x1, x2, handle1, handle2) lines.
    The nodes_out is a sequence of (x, y, handle) lines.

    The returned value is (x, y, n) giving the x and y location of the root
    and the number of leaves n under the root.

    In this notation, x is like depth and y is like breadth.

    """
    handle, blen, child_nodes = root
    if child_nodes:
        n = 0
        nnodes = len(child_nodes)
        node_ys = []
        for i, node in enumerate(child_nodes):
            node_handle, node_blen, node_child_nodes = node
            node_x, node_y, node_n = rlayout(node,
                    x + node_blen, nleaves, leaf_to_y,
                    vert_out, horz_out, nodes_out)
            node_ys.append(node_y)
            nleaves += node_n
            n += node_n
            nodes_out.append((x, node_y, handle))
            nodes_out.append((node_x, node_y, node_handle))
            horz_out.append((node_y, x, node_x, handle, node_handle))
        min_y = min(node_ys)
        max_y = max(node_ys)
        vert_out.append((x, min_y, max_y, handle))
        y = (min_y + max_y) / 2
    else:
        if leaf_to_y:
            y = leaf_to_y[handle]
        else:
            y = nleaves
        n = 1
    return x, y, n



def main():
    root = ('root', None, (
        ('l1', 3, []),
        ('l2', 4, [])))
    vert_out, horz_out, nodes_out = layout_tree(root)
    print()
    print('The vert_out is a sequence of (x, y1, y2) lines.')
    print(vert_out)
    print()
    print('The horz_out is a sequence of (y, x1, x2, handle1, handle2) lines.')
    print(horz_out)
    print()
    print('The nodes_out is a sequence of (x, y, handle) lines.')
    print(nodes_out)
    print()



if __name__ == '__main__':
    main()


