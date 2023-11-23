import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from stable_baselines3 import PPO
import gym
from queue import PriorityQueue
import json

# Define your custom Warehouse environment

class CustomWarehouseEnv(gym.Env):
    def __init__(self, size, num_racks, rack_width):
        self.size = size
        self.num_racks = num_racks
        self.rack_width = rack_width

        self.grid = np.zeros((self.size, self.size), dtype=int)
        self.picker_position = (0, 0)
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(
            low=0, high=2, shape=(self.size, self.size, 1), dtype=np.uint8
        )
        self.target_items = []  # Initialize the list of items to be picked

    def step(self, action):
        dx, dy = 0, 0
        if action == 0:  # left
            dx -= 1
        elif action == 1:  # right
            dx += 1
        elif action == 2:  # up
            dy -= 1
        elif action == 3:  # down
            dy += 1

        new_x = min(max(0, self.picker_position[0] + dx), self.size - 1)
        new_y = min(max(0, self.picker_position[1] + dy), self.size - 1)

        # Check if the new position is in a rack area, and if so, don't move
        if self.grid[new_x, new_y] != 1:
            self.picker_position = (new_x, new_y)

        self.update_observation()

        reward = 1 if self.picker_position in self.target_items else 0
        done = len(self.target_items) == 0
        
        # Return info dict 
        info = {'episode': {}}
        return self._observe(), reward, done, inf
    
    def reset(self):
        self.grid = np.zeros((self.size, self.size), dtype=int)
        self.picker_position = (0, 0)
        return self._observe()
    
    def _observe(self):
        obs = np.copy(self.grid)
        obs[self.picker_position[0], self.picker_position[1]] = 2
        return obs.reshape((self.size, self.size, 1))

    def update_observation(self):
        obs = np.copy(self.grid)
        obs[self.picker_position[0], self.picker_position[1]] = 2
        for item in self.target_items:
            obs[item[0], item[1]] = 3
        return obs

    def render(self, mode='human'):
        if mode == 'rgb_array':
            return self.grid

    def calculate_optimal_route_from_position(self, start_position): #Use a suitable pathfinding algorithm
        def astar(grid, start, targets):
            def heuristic(node):
                return min(abs(node[0] - target[0]) + abs(node[1] - target[1]) for target in targets)

            def reconstruct_path(came_from, current):
                path = []
                while current in came_from:
                    path.insert(0, current)
                    current = came_from[current]
                return path

            open_set = PriorityQueue()
            open_set.put((0, start))
            came_from = {}
            g_score = {(x, y): float("inf") for x in range(len(grid))
                       for y in range(len(grid[0]))}

            g_score[start] = 0

            while not open_set.empty():
                _, current = open_set.get()

                if current in targets:
                    return reconstruct_path(came_from, current)

                for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    new_x, new_y = current[0] + dx, current[1] + dy

                    if (
                        0 <= new_x < len(grid)
                        and 0 <= new_y < len(grid[0])
                        and grid[new_x][new_y] != 1
                    ):
                        tentative_g_score = g_score[current] + 1

                        if tentative_g_score < g_score[(new_x, new_y)]:
                            came_from[(new_x, new_y)] = current
                            g_score[(new_x, new_y)] = tentative_g_score
                            f_score = tentative_g_score + \
                                heuristic((new_x, new_y))
                            open_set.put((f_score, (new_x, new_y)))

            # If no valid path is found, return an empty list
            return []

        # Make a copy of the grid
        grid_copy = np.copy(self.grid)

        # Set the positions of items to be picked as aisles (2)
        for item in self.target_items:
            grid_copy[item[0], item[1]] = 2

        # Calculate the optimal route from the specified picker's position to items
        optimal_route = []
        current_position = start_position

        while self.target_items:
            path = astar(grid_copy, current_position, self.target_items)
            if not path:
                # Handle the case where no path was found
                print("No path found to remaining items.")
                break

            # Ensure there is a valid next position
            next_position = path[1] if len(path) > 1 else path[0]
            optimal_route.extend(path[:-1])

            if next_position in self.target_items:
                self.target_items.remove(next_position)

            current_position = next_position

        # Calculate the optimal distance
        optimal_distance = len(optimal_route)

        return optimal_route, optimal_distance


# Streamlit UI
st.title("Warehouse Picker Simulation")

# Define the size of your environment
size = st.number_input(f"Warehouse Dimensions", value=19, min_value=1)
num_racks = st.number_input(f"No. of Racks", value=6, min_value=1)
rack_width = st.number_input(f"Rack Width", value=2, min_value=1)
env = CustomWarehouseEnv(size, num_racks, rack_width)

# Initialize the model with MlpPolicy
model = PPO("MlpPolicy", env, verbose=0)
model.learn(total_timesteps=100)  # You can adjust the number of timesteps

# Create an empty grid
grid = np.zeros((size, size), dtype=int)

# Place the racks and aisles in the grid
rack_spacing = (size - (num_racks * rack_width) - 2) // (num_racks - 1)
bottom_padding = 1  # Number of grids to pad at the bottom
top_padding = 1  # Number of grids to pad at the top

rack_labels = iter("ABCDEFGHIJKLMNOPQRSTUVWXYZ")  # Use alphabet labels
rack_label_dict = {}  # Create a dictionary to store rack labels

for i in range(num_racks):
    # Start the rack after the first aisle space
    rack_x = i * (rack_width + rack_spacing) + 1
    grid[bottom_padding:size - top_padding, rack_x: rack_x + rack_width] = 1

    # Assign the next alphabet label to the rack
    rack_label = next(rack_labels)

    # Update the grid with the rack label
    rack_label_dict[rack_label] = (
        bottom_padding, size - top_padding, rack_x, rack_x + rack_width)
    grid[bottom_padding:size - top_padding, rack_x: rack_x +
         rack_width] = 1  # Indicate the rack with 1


# Initialize the picker's position (bottom left by default)
picker_x, picker_y = size - 1, 0

# Create legends for aisles, racks, picker, items, and optimal route
legends = {
    0: "Aisles",
    1: "Racks",
    2: "Picker",
    3: "Items",  # Items to be picked
    4: "Optimal Route from Current Position",
}

# Display the environment in Streamlit
st.title("Custom Warehouse Environment")

# Allow the user to dynamically specify the picker's location (bottom left by default)
st.sidebar.header("Picker's Position")
user_picker_x = st.sidebar.selectbox(
    "X-coordinate (0-18)", list(range(size)))
user_picker_y = st.sidebar.selectbox(
    "Y-coordinate (0-18)", list(range(size)))

# Check if the user-selected picker's position is valid (not in a rack)
if grid[user_picker_x, user_picker_y] != 1:
    # Update the picker's position
    picker_x, picker_y = user_picker_x, user_picker_y
    grid[picker_x, picker_y] = 2

# Allow the user to input the orders in this format
st.sidebar.header("Input Orders")
order_json = st.sidebar.text_area(
    "Enter orders in this format", '[ [3, 4], [6, 7] ]')
# Parse the input to get the list of orders
try:
    orders = json.loads(order_json)
except json.JSONDecodeError:
    orders = []

# Clear previously picked items
env.target_items = []

# Validate and display the items to be picked on the grid
for order in orders:
    if isinstance(order, list) and len(order) == 2:
        item_x, item_y = order
        if 0 <= item_x < size and 0 <= item_y < size and grid[item_x, item_y] == 1:
            env.target_items.append((item_x, item_y))
            grid[item_x, item_y] = 3  # Display items on the grid

# Calculate the optimal picking route
if env.target_items:
    optimal_route, optimal_distance = env.calculate_optimal_route_from_position(
        (picker_x, picker_y))

    # Mark the optimal route coordinates in yellow on the grid
    for coord in optimal_route:
        grid[coord[0], coord[1]] = 4  # Yellow color for optimal route

# Display the optimal route and distance
st.write(f"Optimal Picking Route: {optimal_route}")
st.write(f"Optimal Distance from Picker: {optimal_distance}")


# Create a colormap to display aisles, racks, picker, items, and optimal route in different colors
cmap = plt.cm.colors.ListedColormap(
    ['white', 'gray', 'green', 'blue', 'yellow'])

fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(grid, cmap=cmap, extent=[
    0, size, size, 0], origin="upper")
ax.set_xticks(np.arange(0, size, 1))
ax.set_yticks(np.arange(0, size, 1))
ax.grid(color='r', linewidth=2)
ax.set_aspect('equal')

# Add rack labels to the grid
for label, (y1, y2, x1, x2) in rack_label_dict.items():
    ax.text((x1 + x2) / 2, (y1 + y2) / 2, label,
            ha='center', va='center', fontsize=12, fontweight='bold', color='black')


# Create legends as small squares
legend_elements = [
    Patch(facecolor='white', edgecolor='black', label=legends[0]),
    Patch(facecolor='gray', edgecolor='black', label=legends[1]),
    Patch(facecolor='green', edgecolor='black', label=legends[2]),
    Patch(facecolor='blue', edgecolor='black', label=legends[3]),
    Patch(facecolor='yellow', edgecolor='black',label=legends[4]),
]

# Display legends
ax.legend(handles=legend_elements, loc='upper right', fontsize='medium')
st.pyplot(fig)
