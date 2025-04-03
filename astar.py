# --- START OF FILE astar_improved.py ---

import numpy as np
from heapq import heappush, heappop
# Using Manhattan distance for grid movement is often slightly faster
# import math # No longer needed for heuristic
import pygame
import time

class Node:
    def __init__(self, position, g_cost, h_cost, parent=None):
        self.position = position
        self.g_cost = g_cost
        self.h_cost = h_cost
        self.f_cost = g_cost + h_cost
        self.parent = parent

    # __lt__ is essential for heapq to work correctly
    def __lt__(self, other):
        # Tie-breaking using h_cost can sometimes speed up exploration
        if self.f_cost == other.f_cost:
            return self.h_cost < other.h_cost
        return self.f_cost < other.f_cost

    # __eq__ and __hash__ allow Nodes to be stored in sets/dictionaries if needed based on position
    def __eq__(self, other):
        return self.position == other.position

    def __hash__(self):
        return hash(self.position)


def calculate_heuristic(current, goal):
    """Calculates Manhattan distance."""
    return abs(current[0] - goal[0]) + abs(current[1] - goal[1])
    # Original Euclidean distance (also valid but potentially slower):
    # return math.sqrt((current[0] - goal[0]) ** 2 + (current[1] - goal[1]) ** 2)


def get_neighbors(position, grid):
    neighbors = []
    # Only orthogonal movements allowed
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)] # Up, Down, Left, Right

    for dx, dy in directions:
        new_x, new_y = position[0] + dx, position[1] + dy
        # Check bounds
        if 0 <= new_x < grid.shape[0] and 0 <= new_y < grid.shape[1]:
            # Check if not an obstacle (value 5)
            if grid[new_x, new_y] != 5:
                neighbors.append((new_x, new_y))

    return neighbors


def astar(grid, start, goal, visualize_callback=None):
    """
    Performs the A* search algorithm.

    Args:
        grid (np.array): The grid representing the map (0=walkable, 5=obstacle).
        start (tuple): The starting position (row, col).
        goal (tuple): The goal position (row, col).
        visualize_callback (function, optional): Callback function for visualization.

    Returns:
        list: A list of tuples representing the path from start to goal, or None if no path exists.
    """
    open_list_heap = [] # Priority queue (min-heap) storing Node objects
    open_list_positions = set() # For fast checking if a position is in the open list
    closed_set = set() # Stores visited positions (tuples)
    # Store g_costs to efficiently check if a better path is found
    g_costs = {start: 0}

    # Calculate initial heuristic and create start node
    h_start = calculate_heuristic(start, goal)
    start_node = Node(start, 0, h_start)

    # Add start node to the open list
    heappush(open_list_heap, start_node)
    open_list_positions.add(start)

    while open_list_heap:
        # Get the node with the lowest f_cost from the heap
        current_node = heappop(open_list_heap)
        current_pos = current_node.position
        open_list_positions.remove(current_pos) # Remove from position tracker

        # Goal reached? Reconstruct path.
        if current_pos == goal:
            path = []
            temp = current_node
            while temp:
                path.append(temp.position)
                temp = temp.parent
            # Provide final path for visualization before returning
            if visualize_callback:
                visualize_callback(grid, closed_set, open_list_positions, current_pos, path[::-1])
                time.sleep(1) # Pause to show final path clearly
            return path[::-1] # Return reversed path (start -> goal)

        # Add current position to the closed set (already processed)
        closed_set.add(current_pos)

        # Visualization callback (optional)
        if visualize_callback:
            # Pass the set of positions for accurate visualization
            visualize_callback(grid, closed_set, open_list_positions, current_pos, None)

        # Explore neighbors
        for neighbor_pos in get_neighbors(current_pos, grid):
            # Skip if neighbor is already processed
            if neighbor_pos in closed_set:
                continue

            # Calculate tentative g_cost (cost from start to neighbor through current)
            # Assume cost of moving to neighbor is 1
            tentative_g_cost = current_node.g_cost + 1

            # If this neighbor is already in the open list with a lower or equal g_cost, skip
            # This check uses the g_costs dictionary for efficiency
            if neighbor_pos in g_costs and tentative_g_cost >= g_costs[neighbor_pos]:
                continue

            # This path to neighbor is the best found so far. Record/Update it.
            g_costs[neighbor_pos] = tentative_g_cost
            h_cost = calculate_heuristic(neighbor_pos, goal)
            parent = current_node
            neighbor_node = Node(neighbor_pos, tentative_g_cost, h_cost, parent)

            # Add neighbor to the open list heap and position tracker
            # No need to check if it's already there; heapq handles priorities.
            # If a node for this position was already added with a higher g_cost,
            # this new one with lower g_cost (and f_cost) will be popped first.
            heappush(open_list_heap, neighbor_node)
            open_list_positions.add(neighbor_pos)


    # If the open list becomes empty and goal was not reached, no path exists
    print("No path found.")
    return None

# --- Visualizer Class remains the same ---
class Visualizer:
    def __init__(self, grid, start_pos, goal_pos): # Add start/goal for drawing
        pygame.init()
        self.cell_size = 30
        self.grid = grid
        self.width = grid.shape[1] * self.cell_size
        self.height = grid.shape[0] * self.cell_size
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("A* Pathfinding Visualization (Improved)")
        self.start_pos = start_pos
        self.goal_pos = goal_pos

        # Kolory
        self.colors = {
            'background': (255, 255, 255),
            'wall': (50, 50, 50),          # Darker grey for walls
            'grid_line': (200, 200, 200),
            'start': (0, 255, 0),         # Green
            'goal': (255, 0, 0),           # Red
            'open_list': (173, 216, 230),  # Light Blue
            'closed_set': (211, 211, 211), # Light Grey
            'current': (0, 0, 255),        # Blue for current node being processed
            'final_path': (255, 165, 0),   # Orange
        }

    def visualize(self, grid, closed_set, open_list_positions, current_pos, final_path=None):
        self.screen.fill(self.colors['background'])

        # Rysowanie siatki i przeszkód
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                rect = pygame.Rect(c * self.cell_size, r * self.cell_size, self.cell_size, self.cell_size)
                color = self.colors['background']
                if grid[r, c] == 5:
                    color = self.colors['wall']
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, self.colors['grid_line'], rect, 1) # Linie siatki

        # Rysowanie zbioru zamkniętego (light grey)
        for pos in closed_set:
            if pos != self.start_pos and pos != self.goal_pos: # Don't overwrite start/goal
                pygame.draw.rect(self.screen, self.colors['closed_set'],
                                 (pos[1] * self.cell_size + 1, pos[0] * self.cell_size + 1, self.cell_size - 2, self.cell_size - 2))

        # Rysowanie listy otwartej (light blue)
        for pos in open_list_positions:
            if pos != self.start_pos and pos != self.goal_pos:
                pygame.draw.rect(self.screen, self.colors['open_list'],
                                 (pos[1] * self.cell_size + 1, pos[0] * self.cell_size + 1, self.cell_size - 2, self.cell_size - 2))

        # Rysowanie aktualnie przetwarzanego węzła (blue)
        if current_pos and current_pos != self.start_pos and current_pos != self.goal_pos:
            pygame.draw.rect(self.screen, self.colors['current'], (
                current_pos[1] * self.cell_size + 1, current_pos[0] * self.cell_size + 1, self.cell_size - 2, self.cell_size - 2))

        # Rysowanie końcowej ścieżki (orange)
        if final_path:
            for pos in final_path:
                if pos != self.start_pos and pos != self.goal_pos:
                    pygame.draw.rect(self.screen, self.colors['final_path'],
                                     (pos[1] * self.cell_size + 1, pos[0] * self.cell_size + 1, self.cell_size - 2, self.cell_size - 2))

        # Rysowanie Startu (green) i Celu (red) - always on top
        start_rect = pygame.Rect(self.start_pos[1] * self.cell_size, self.start_pos[0] * self.cell_size, self.cell_size, self.cell_size)
        goal_rect = pygame.Rect(self.goal_pos[1] * self.cell_size, self.goal_pos[0] * self.cell_size, self.cell_size, self.cell_size)
        pygame.draw.rect(self.screen, self.colors['start'], start_rect)
        pygame.draw.rect(self.screen, self.colors['goal'], goal_rect)

        pygame.display.flip()
        # Adjust sleep time for desired visualization speed
        time.sleep(0.01) # Faster visualization


def load_map_from_file(filename):
    try:
        with open(filename, 'r') as file:
            lines = file.readlines()
            grid = []
            for line in lines:
                # Ensure line is not empty before processing
                stripped_line = line.strip()
                if stripped_line:
                    row = [int(x) for x in stripped_line.split()]
                    grid.append(row)
            # Check if grid is empty or rows have inconsistent lengths
            if not grid:
                raise ValueError("Grid file is empty.")
            first_row_len = len(grid[0])
            if not all(len(row) == first_row_len for row in grid):
                raise ValueError("Grid rows have inconsistent lengths.")
        return np.array(grid)
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return None
    except ValueError as e:
        print(f"Error reading grid file: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while loading the grid: {e}")
        return None

if __name__ == "__main__":
    grid = load_map_from_file('grid.txt')

    if grid is not None:
        start = (0, 0)
        # Corrected goal coordinates (within bounds 0-19)
        goal = (18, 19)

        # --- Input Validation ---
        valid_input = True
        rows, cols = grid.shape
        if not (0 <= start[0] < rows and 0 <= start[1] < cols):
            print(f"Error: Start position {start} is out of grid bounds (0-{rows-1}, 0-{cols-1}).")
            valid_input = False
        elif grid[start[0], start[1]] == 5:
            print(f"Error: Start position {start} is on an obstacle.")
            valid_input = False

        if not (0 <= goal[0] < rows and 0 <= goal[1] < cols):
            print(f"Error: Goal position {goal} is out of grid bounds (0-{rows-1}, 0-{cols-1}).")
            valid_input = False
        elif grid[goal[0], goal[1]] == 5:
            print(f"Error: Goal position {goal} is on an obstacle.")
            valid_input = False
        # --- End Input Validation ---

        if valid_input:
            visualizer = Visualizer(grid, start, goal) # Pass start/goal to visualizer
            path = astar(grid, start, goal, visualize_callback=visualizer.visualize)

            if path:
                print("Path found:", path)
                print("Path length:", len(path) -1) # Number of steps

                # Keep the visualization window open until closed by the user
                running = True
                while running:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            running = False
                    # Optionally redraw the final state in the loop if needed
                    # visualizer.visualize(grid, set(), set(), goal, path) # Redraw final path
                    pygame.time.wait(50) # Prevent high CPU usage in idle loop

            # No "else" needed here, astar function prints "No path found." if necessary

        pygame.quit() # Ensure pygame quits properly regardless of outcome