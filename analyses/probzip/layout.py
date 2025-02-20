def tree_layout(G, root, width=1.0, vert_gap=0.2, vert_loc=0, xcenter=0.5, pos=None, parent=None):
    """
    Compute the positions for a hierarchical layout of a tree or DAG.
    
    Parameters:
    - G: networkx graph (should be a tree or DAG).
    - root: the root node of the current branch.
    - width: horizontal space allocated for this branch.
    - vert_gap: gap between levels of hierarchy.
    - vert_loc: vertical location of the root.
    - xcenter: horizontal location of the root.
    - pos: dictionary of positions (used in recursion).
    - parent: parent of the current root (to avoid revisiting in undirected graphs).

    Returns:
    - pos: A dictionary mapping each node to its (x, y) position.
    """
    if pos is None:
        pos = {root: (xcenter, vert_loc)}
    else:
        pos[root] = (xcenter, vert_loc)
    
    # Get neighbors; for undirected graphs, remove the parent to avoid going backwards.
    neighbors = list(G.neighbors(root))
    if parent is not None and parent in neighbors:
        neighbors.remove(parent)
    
    if len(neighbors) != 0:
        # Divide the horizontal space among children.
        dx = width / len(neighbors)
        next_x = xcenter - width / 2 - dx / 2
        for neighbor in neighbors:
            next_x += dx
            pos = tree_layout(G, neighbor, width=dx, vert_gap=vert_gap,
                                vert_loc=vert_loc - vert_gap, xcenter=next_x, pos=pos, parent=root)
    return pos