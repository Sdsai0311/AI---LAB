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

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

def plot_maze_path(maze_data, path, title):
    ROWS, COLS = len(maze_data), len(maze_data[0])

    # Map maze characters to numerical values for plotting
    # S: 0, G: 1, 0: 2, 1: 3
    numerical_maze = np.zeros((ROWS, COLS), dtype=int)
    for r in range(ROWS):
        for c in range(COLS):
            if maze_data[r][c] == 'S':
                numerical_maze[r][c] = 0  # Start
            elif maze_data[r][c] == 'G':
                numerical_maze[r][c] = 1  # Goal
            elif maze_data[r][c] == '0':
                numerical_maze[r][c] = 2  # Empty
            elif maze_data[r][c] == '1':
                numerical_maze[r][c] = 3  # Wall

    # Define colors for each maze element
    colors = ['green', 'red', 'lightgray', 'black'] # S, G, 0, 1
    cmap = plt.cm.colors.ListedColormap(colors)
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(COLS, ROWS))

    # Plot the maze background
    ax.imshow(numerical_maze, cmap=cmap, norm=norm, origin='upper', extent=[-0.5, COLS-0.5, ROWS-0.5, -0.5])

    # Draw grid lines
    ax.set_xticks(np.arange(COLS + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(ROWS + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=1)
    ax.tick_params(which="minor", size=0)

    # Plot the path
    if path:
        path_rows = [p[0] for p in path]
        path_cols = [p[1] for p in path]
        ax.plot(path_cols, path_rows, color='blue', linewidth=3, marker='o', markersize=8, label='BFS Path')

    # Add labels to 'S' and 'G'
    start_pos = find_pos('S')
    goal_pos = find_pos('G')
    ax.text(start_pos[1], start_pos[0], 'S', ha='center', va='center', color='white', fontsize=12, fontweight='bold')
    ax.text(goal_pos[1], goal_pos[0], 'G', ha='center', va='center', color='white', fontsize=12, fontweight='bold')

    # Set plot title and labels
    ax.set_title(title)
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    ax.set_xticks(np.arange(COLS))
    ax.set_yticks(np.arange(ROWS))

    # Create a legend
    legend_handles = [
        mpatches.Patch(color='green', label='Start (S)'),
        mpatches.Patch(color='red', label='Goal (G)'),
        mpatches.Patch(color='lightgray', label='Empty (0)'),
        mpatches.Patch(color='black', label='Wall (1)')
    ]
    if path:
        legend_handles.append(plt.Line2D([0], [0], color='blue', linewidth=3, marker='o', markersize=8, label='BFS Path'))

    ax.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    plt.show()

# Get the BFS path
bfs_path = bfs_maze()

# Call the plotting function
plot_maze_path(maze, bfs_path, "BFS Path Visualization")
