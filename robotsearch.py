import heapq
import time
from collections import deque, namedtuple

class RobotCourierProblem:
    def __init__(self, grid, start, packages, drop):
        self.grid = grid
        self.h = len(grid)
        self.w = len(grid[0])
        self.start = start
        self.packages = tuple(packages)
        self.drop = drop
        self.n = len(self.packages)

    def in_bounds(self, pos):
        r, c = pos
        return 0 <= r < self.h and 0 <= c < self.w

    def passable(self, pos):
        r, c = pos
        return self.grid[r][c] != '#'

    def is_goal(self, state):
        r, c, next_idx, carrying = state
        return next_idx == self.n and not carrying

    def start_state(self):
        return (self.start[0], self.start[1], 0, False)

    def neighbors(self, state):
        r, c, next_idx, carrying = state
        results = []
        for dr, dc, name in [(-1,0,'N'), (1,0,'S'), (0,-1,'W'), (0,1,'E')]:
            nr, nc = r+dr, c+dc
            if self.in_bounds((nr,nc)) and self.passable((nr,nc)):
                results.append((name, (nr, nc, next_idx, carrying), 1))
        if not carrying and next_idx < self.n and (r, c) == self.packages[next_idx]:
            results.append(('Pickup', (r, c, next_idx, True), 0))
        if carrying and (r, c) == self.drop:
            results.append(('Deliver', (r, c, next_idx + 1, False), 0))
        return results

Node = namedtuple('Node', ['state', 'cost', 'parent', 'action'])

def uniform_cost_search(problem):
    start = problem.start_state()
    frontier = []
    heapq.heappush(frontier, (0, Node(start, 0, None, None)))
    explored = dict()
    nodes_expanded = 0
    max_frontier = 1
    while frontier:
        max_frontier = max(max_frontier, len(frontier))
        cost, node = heapq.heappop(frontier)
        state = node.state
        if state in explored and explored[state] < cost:
            continue
        if problem.is_goal(state):
            path = []
            cur = node
            while cur is not None:
                path.append(cur.state)
                cur = cur.parent
            path.reverse()
            return path, cost, {'expanded': nodes_expanded, 'max_frontier': max_frontier}
        nodes_expanded += 1
        explored[state] = cost
        for action, next_state, step_cost in problem.neighbors(state):
            new_cost = cost + step_cost
            if next_state not in explored or new_cost < explored.get(next_state, float('inf')):
                child = Node(next_state, new_cost, node, action)
                heapq.heappush(frontier, (new_cost, child))
    return None, float('inf'), {'expanded': nodes_expanded, 'max_frontier': max_frontier}

def manhattan(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def courier_heuristic(problem, state):
    r, c, next_idx, carrying = state
    pos = (r, c)
    if next_idx >= problem.n:
        if carrying:
            return manhattan(pos, problem.drop)
        return 0
    if carrying:
        return manhattan(pos, problem.drop)
    pkg = problem.packages[next_idx]
    return manhattan(pos, pkg) + manhattan(pkg, problem.drop)

def a_star_search(problem):
    start = problem.start_state()
    frontier = []
    start_h = courier_heuristic(problem, start)
    heapq.heappush(frontier, (start_h, 0, Node(start, 0, None, None)))
    explored = dict()
    nodes_expanded = 0
    max_frontier = 1
    while frontier:
        max_frontier = max(max_frontier, len(frontier))
        priority, cost, node = heapq.heappop(frontier)
        state = node.state
        if state in explored and explored[state] < cost:
            continue
        if problem.is_goal(state):
            path = []
            cur = node
            while cur is not None:
                path.append(cur.state)
                cur = cur.parent
            path.reverse()
            return path, cost, {'expanded': nodes_expanded, 'max_frontier': max_frontier}
        nodes_expanded += 1
        explored[state] = cost
        for action, next_state, step_cost in problem.neighbors(state):
            new_cost = cost + step_cost
            h = courier_heuristic(problem, next_state)
            f = new_cost + h
            if next_state not in explored or new_cost < explored.get(next_state, float('inf')):
                child = Node(next_state, new_cost, node, action)
                heapq.heappush(frontier, (f, new_cost, child))
    return None, float('inf'), {'expanded': nodes_expanded, 'max_frontier': max_frontier}

def reconstruct_actions(path_states, problem):
    actions = []
    for i in range(1, len(path_states)):
        r0, c0, n0, carry0 = path_states[i-1]
        r1, c1, n1, carry1 = path_states[i]
        if (r0, c0) != (r1, c1):
            if r1 == r0 - 1: actions.append('N')
            elif r1 == r0 + 1: actions.append('S')
            elif c1 == c0 - 1: actions.append('W')
            elif c1 == c0 + 1: actions.append('E')
            else: actions.append('Move')
        else:
            if not carry0 and carry1:
                actions.append('Pickup')
            elif carry0 and not carry1:
                actions.append('Deliver')
            else:
                actions.append('NoOp')
    return actions

def sample_map_example():
    grid = [
        ".........",
        ".###..#..",
        ".#..#.#..",
        ".#..#....",
        ".#..###..",
        ".....#...",
        ".........",
    ]
    grid = [list(row) for row in grid]
    start = (6, 0)
    packages = [(0, 8), (2, 2), (4, 0)]
    drop = (0, 0)
    problem = RobotCourierProblem(grid, start, packages, drop)
    return problem

def run_experiment(problem):
    path_u, cost_u, stats_u = uniform_cost_search(problem)
    path_a, cost_a, stats_a = a_star_search(problem)
    print("UCS:", cost_u, stats_u)
    print("A*:", cost_a, stats_a)

if __name__ == "__main__":
    prob = sample_map_example()
    run_experiment(prob)
