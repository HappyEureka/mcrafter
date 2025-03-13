import numpy as np
import heapq

class PathFinding:
    def __init__(self, walkable, grid_size=(64, 64)):
        self.rows, self.cols = grid_size
        self.walkable = walkable
        self.directions = {(-1, 0): 3, (1, 0): 4, (0, -1): 1, (0, 1): 2}
    
    def is_valid(self, node):
        r, c = node
        return 0 <= r < self.rows and \
                0 <= c < self.cols and \
                self.grid[node] in self.curr_walkable
    
    def generate_path(self, parent, node):
        path = []
        while node is not None:
            path.append(node)
            node = parent[node]
        path.reverse()
        return path
    
    def find(self, grid, start, target_item):
        self.grid = grid
        self.curr_walkable = self.walkable.copy()
        start = tuple(start)  # Convert to tuple for hashing

        open_list = []
        heapq.heappush(open_list, (0, start))
        closed_set = set()
        parent = {start: None}
        while open_list:
            current_cost, current_node = heapq.heappop(open_list)
            if current_node in closed_set:
                continue
            closed_set.add(current_node)
            if self.grid[current_node] == target_item:
                path = self.generate_path(parent, current_node)
                # find direction
                np_path = np.array(path)
                movements = np_path[1:] - np_path[:-1]
                return [self.directions[tuple(move)] for move in movements]

            # Explore neighbors
            for r, c in self.directions.keys():
                neighbor = (current_node[0] + r, current_node[1] + c)
                if neighbor in closed_set or not (0 <= neighbor[0] < self.rows and 0 <= neighbor[1] < self.cols):
                    continue
                if self.grid[neighbor] == target_item:
                    parent[neighbor] = current_node
                    path = self.generate_path(parent, neighbor)
                    # find direction
                    np_path = np.array(path)
                    movements = np_path[1:] - np_path[:-1]
                    return [self.directions[tuple(move)] for move in movements]
                if not self.is_valid(neighbor):
                    continue
                new_cost = current_cost + 1  # Uniform cost
                heapq.heappush(open_list, (new_cost, neighbor))
                parent[neighbor] = current_node

        # If the target is not reachable, find the closest node
        target_positions = np.argwhere(self.grid == target_item)
        if target_positions.size == 0:
            # Target item not in grid at all
            return None

        min_distance = float('inf')
        closest_node = None
        for node in closed_set:
            for target_pos in target_positions:
                distance = abs(node[0] - target_pos[0]) + abs(node[1] - target_pos[1])
                if distance < min_distance:
                    min_distance = distance
                    closest_node = node

        if closest_node is not None:
            path = self.generate_path(parent, closest_node)
            np_path = np.array(path)
            movements = np_path[1:] - np_path[:-1]
            return [self.directions[tuple(move)] for move in movements]
        else:
            return None
if __name__ == '__main__':
    # Example usage with 2D numpy array:
    grid = np.array([
        [2, 3, 3, 2, 2, 2],
        [16, 4, 4, 2, 2, 2],
        [8, 3, 3, 2, 2, 2],
        [16, 1, 2, 4, 4, 3],
        [1, 1, 13, 16, 4, 3],
        [1, 1, 1, 4, 4, 3],
    ])
    
    pf = PathFinding(walkable={2,4,5}, grid_size=(6, 6))
    target_item = 16
    path = pf.find(grid, start=(0, 0), target_item=16)
    print(path)