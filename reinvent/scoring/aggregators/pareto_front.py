import math

'''
get_pareto_front computes the Pareto frontier of a set of points in multi-dimensional space. 
The minimum number of points in the Pareto frontier can be specified with the `min_points` parameter. 
If the number of points in the Pareto frontier is less than `min_points`, additional points are added to the
frontier based on their Euclidean distance to the frontier.

Adapted from: https://github.com/justinormont/pareto-frontier/tree/master
'''

def get_pareto_frontier(points, direction="maximize", min_points=5):
    if not isinstance(points, list) or not all(isinstance(p, (list, tuple)) and len(p) > 0 for p in points):
        raise TypeError("Require list of points with at least one dimension each")

    pareto_frontier = []

    for p in points:
        # Remove points from the Pareto front that are dominated by `p`
        pareto_frontier = [q for q in pareto_frontier if not _dominates(p, q, direction)]
        # Add `p` to the Pareto front if it is not dominated by any existing point
        if not any(_dominates(q, p, direction) for q in pareto_frontier):
            pareto_frontier.append(p)

    if len(pareto_frontier) < min_points:
        remaining_points = [p for p in points if p not in pareto_frontier]
        additional_points = sorted(
            remaining_points,
            key=lambda x: _euclidean_distance(x, pareto_frontier)
        )
        pareto_frontier.extend(additional_points[:min_points - len(pareto_frontier)])

    return pareto_frontier

def _dominates(a, b, direction="maximize"):
    if direction in {"maximize"}:
        return all(ai >= bi for ai, bi in zip(a, b)) and any(ai > bi for ai, bi in zip(a, b))
    elif direction in {"minimize"}:
        return all(ai <= bi for ai, bi in zip(a, b)) and any(ai < bi for ai, bi in zip(a, b))
    else:
        raise ValueError(f"Invalid direction: {direction}")

def _euclidean_distance(point, frontier):
    """
    Calculate the minimum Euclidean distance from a point to any point in the Pareto frontier.
    """
    return min(math.sqrt(sum((xi - yi) ** 2 for xi, yi in zip(point, pareto_point))) for pareto_point in frontier)