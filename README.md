# AI---LAB
# Maze :
import collections
import heapq

maze = [
    ['S','0','1','0'],
    ['1','0','1','0'],
    ['0','0','0','G']
]

ROWS, COLS = len(maze), len(maze[0])
MOVES = [(0, 1), (0, -1), (1, 0), (-1, 0)] # Right, Left, Down, Up

def find_pos(symbol):
    for i in range(ROWS):
        for j in range(COLS):
            if maze[i][j] == symbol:
                return i, j

def valid(x, y):
    return 0 <= x < ROWS and 0 <= y < COLS and maze[x][y] != '1'

#BFS
def bfs_maze():
    start = find_pos('S')
    goal = find_pos('G')
    queue = collections.deque([(start, [start])])
    visited = set([start])

    while queue:
        (x, y), path = queue.popleft()
        if (x, y) == goal:
            return path

        for dx, dy in MOVES:
            nx, ny = x+dx, y+dy
            if valid(nx, ny) and (nx, ny) not in visited:
                visited.add((nx, ny))
                queue.append(((nx, ny), path + [(nx, ny)]))

#DFS
def dfs_maze():
    start = find_pos('S')
    goal = find_pos('G')
    stack = [(start, [start])]
    visited = set()

    while stack:
        (x, y), path = stack.pop()
        if (x, y) == goal:
            return path

        if (x, y) not in visited:
            visited.add((x, y))
            for dx, dy in MOVES:
                nx, ny = x+dx, y+dy
                if valid(nx, ny):
                    stack.append(((nx, ny), path + [(nx, ny)]))

#A*
def manhattan(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def astar_maze():
    start = find_pos('S')
    goal = find_pos('G')
    pq = []
    heapq.heappush(pq, (0, start, [start]))
    visited = set()

    while pq:
        f, current, path = heapq.heappop(pq)
        if current == goal:
            return path

        if current not in visited:
            visited.add(current)
            for dx, dy in MOVES:
                nx, ny = current[0]+dx, current[1]+dy
                if valid(nx, ny):
                    g = len(path)
                    h = manhattan((nx, ny), goal)
                    heapq.heappush(pq, (g+h, (nx, ny), path + [(nx, ny)]))

# Sample Input
print("BFS Path:", bfs_maze())
print("DFS Path:", dfs_maze())
print("A* Path:", astar_maze())


