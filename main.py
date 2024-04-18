
import pygame
import random
import time
import psutil
import os
import pygame_menu
import math
import heapq 
import cProfile
import tracemalloc

from memory_profiler import profile
from collections import deque




WIDTH, HEIGHT = 600, 600
ROWS = None
COLS = None
SQUARE_SIZE = None
DELAY = 0.0  # Delay between computer moves in seconds
level=None
search_method=None

WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
PURPLE = (128, 0, 128)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)


execution_times = []
memory_usages = []
num_moves = []

obstaculos_nivel_9 = None
obstaculos_nivel_10 = None



# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Sliding Game")

font = pygame.font.Font(None, 36)
def start_game(selected_level, selected_search, menu):
    global level, search_method  # Declare level and search_method as global variables

    menu.disable()
    level = selected_level[0][1]
    search_method = selected_search[0][1]
    print("The level selected is:", level)
    print("The search method selected is:", search_method)
    main(level, search_method, menu)


# Function to draw the level selection menu
def draw_level_selection_menu():
    screen.fill(BLUE)

    menu = pygame_menu.Menu(height=600, width=600, title='SlideQuest',
                            theme=pygame_menu.themes.THEME_BLUE)

    levels = [('Level 1', 1), ('Level 2', 2),
              ('Level 3', 3), ('Level 4', 4),
              ('Level 5', 5), ('Level 6', 6),
              ('Level 7', 7), ('Level 8', 8), 
              ('Level 9', 9), ('Level 10', 10)]

    level_selector = menu.add.selector('Select Level:', levels, onchange=None)

    search_selector = menu.add.selector('Select Mode:', [('Human', 'human'), ('Random', 'random'),
                                                                  ('Breadth-first', 'breadth_first'),
                                                                  ('Depth-first', 'depth_first'),
                                                                  ('Depth-limited5', 'depth_limited5'),
                                                                  ('Depth-limited8', 'depth_limited8'),
                                                                  ('Iterative Deepening', 'iterative_deepening'),
                                                                  ('Uniform Cost', 'uniform_cost'),
                                                                  ('Greedy Search', 'greedy'),
                                                                  ('A* Search H1', 'astar'),
                                                                  ('A* Search H2', 'astar2'),
                                                                  ('Weighted A* Search', 'weighted_astar'),
                                                                  ('Simulated Annealing', 'simulated_annealing'),
                                                                  ('Hill Climbing','hill climbing')],

                                         onchange=None)
    menu.add.button('Rules', show_game_rules)

    menu.add.button('Start', lambda menu=menu: start_game(level_selector.get_value(),
                                                           search_selector.get_value(), menu), menu)

    menu.add.label('Lara Neves and Helena Costa', max_char=-1, font_size=14)
    menu.add.label('Artificial Intelligence, MECD 2024', max_char=-1, font_size=14)

    menu.enable()  
    menu.mainloop(screen)  # Start the menu loop


def show_game_rules():
    rules_menu = pygame_menu.Menu(height=600, width=600, title='Game Rules', theme=pygame_menu.themes.THEME_BLUE)
    rules = "Rules:\n" \
            "1. Move the purple piece to the white space to win the game.\n" \
            "2. You can move the piece in the desired direction (up, down, left, or right) but be watch out, there could be obstacles in the way!\n" \
            "3. The game has 8 levels of increasing difficulty, where the grid size changes as well as the number of obstacles.\n" \
            "4. If you're playing with a search method, watch the computer find the solution!\n" \
            "5. Have fun!\n"
    rules_menu.add.label(rules, max_char=-1, font_size=20, font_color=BLACK)
    rules_menu.add.button('Back', draw_level_selection_menu)
    rules_menu.mainloop(screen)

    return rules_menu



# Function to draw the congratulations screen using PygameMenu

def draw_congratulations_screen(menu=None):
    if menu is not None:
        menu.clear()
        menu.add.label("Congratulations!", max_char=-1, font_size=30)
        menu.add.button("Play Again", lambda: print("Play Again clicked") or draw_level_selection_menu())
        menu.add.button("Exit", pygame_menu.events.EXIT, menu)
        menu.enable()  
    else:
        print("Menu object is None. Skipping drawing congratulations screen.")

    return menu


# Function to draw the grid of the game
def draw_grid():
    for x in range(0, WIDTH, SQUARE_SIZE):
        for y in range(0, HEIGHT, SQUARE_SIZE):
            rect = pygame.Rect(x, y, SQUARE_SIZE, SQUARE_SIZE)
            pygame.draw.rect(screen, WHITE, rect, 3)

# Function to draw the pieces
def draw_pieces(player_pos, goal_pos, obstacles):

    for pos, color in [(player_pos, PURPLE), (goal_pos, BLACK)]:
        x, y = pos[0] * SQUARE_SIZE + SQUARE_SIZE // 2, pos[1] * SQUARE_SIZE + SQUARE_SIZE // 2
        pygame.draw.circle(screen, color, (x, y), SQUARE_SIZE // 3)

    for pos in obstacles:
        x, y = pos[0] * SQUARE_SIZE, pos[1] * SQUARE_SIZE
        pygame.draw.rect(screen, BLACK, (x, y, SQUARE_SIZE, SQUARE_SIZE))


def generate_positions(level):

    global obstaculos_nivel_9,obstaculos_nivel_10
    obstacle_positions = None

    GRID_SIZE = 100

    def is_adjacent(pos1, pos2):
        x1, y1 = pos1
        x2, y2 = pos2
        return abs(x1 - x2) <= 1 and abs(y1 - y2) <= 1

    def generate_obstacles(NUM_OBSTACLES):
        
        random.seed(13)
        obstacles = []

        while len(obstacles) < NUM_OBSTACLES:

            obstacle_pos = (random.randint(0, COLS - 1), random.randint(0, ROWS - 1))
            if obstacle_pos != player_pos and obstacle_pos != goal_pos and not is_adjacent(obstacle_pos, player_pos):
                if obstacle_pos not in obstacles:
                    obstacles.append(obstacle_pos)

        for obstacle in obstacles:
            x, y = obstacle
            if (x > 0 and (x - 1, y) not in obstacles) and \
            (x < COLS - 1 and (x + 1, y) not in obstacles) and \
            (y > 0 and (x, y - 1) not in obstacles) and \
            (y < ROWS - 1 and (x, y + 1) not in obstacles):
                obstacles.remove(obstacle)
                new_obstacle_pos = obstacle
                while new_obstacle_pos in obstacles or new_obstacle_pos == player_pos or is_adjacent(new_obstacle_pos, player_pos):
                    new_obstacle_pos = (random.randint(0, COLS - 1), random.randint(0, ROWS - 1))
                obstacles.append(new_obstacle_pos)

        assert len(obstacles) == NUM_OBSTACLES

        return obstacles

    if level in [1, 2, 3]:
        ROWS, COLS = 4, 4
        if level==1:
            obstacle_positions = [(1, 2), (2,0), (3,3)]
            goal_pos=(2,3)
            player_pos = (0,0)
        elif level==2:
            obstacle_positions=[(1,1),(0,3),(2,0), (2,2),(3,0)]
            goal_pos=(2,1)
            player_pos = (0,0)
        elif level==3:
            obstacle_positions=[(1,1),(0,3),(2,2), (3,0)]
            goal_pos=(3,2)
            player_pos = (0,0)


    elif level in [4, 5, 6]:
        ROWS, COLS = 6, 6
        if level == 4:
            obstacle_positions =   [(1,1), (3,0), (4,1),(5,1),(2,2),(3,3), (3,2), (4,3), (0,3), (1,4), (3,4),(5,5)]
            goal_pos=(4,2)
            player_pos = (0,0)
        elif level==5:
            obstacle_positions=[(0,0), (3,0),(4,0),(4,1),(4,2),(1,2),(2,2),(2,3), (3,4),(4,4),(5,4),(1,5)]
            goal_pos=(5,0)
            player_pos = (1,3)
        elif level == 6:
            obstacle_positions =  [(1,0),(3,0),(5,0),(1,2),(3,2),(1,3),(3,3),(4,3),(1,4),(2,4),(2,5)]
            goal_pos=(0,5)
            player_pos = (4,5)
    elif level in [7,8]:
        ROWS, COLS = 8, 8
        if level==7:
            obstacle_positions =  [(0,4),(1,1),(1,2),(2,3),(2,6),(3,0),(3,2),(3,3),(3,5),(4,3),(4,7),(5,2),(5,3),(5,5),(6,2),(6,6),(7,1),(7,4)]
            goal_pos=(7,0)
            player_pos = (7,7)


        elif level ==8: #impossible level
            obstacle_positions =  [(0,1),(0,2),(0,3),(1,1),(1,5),(1,6),(2,1),(2,4),(2,7),(3,1),(3,2),(3,5),(4,5),(5,1),(5,2),(5,3),(5,6),(5,7),(6,2),(6,3),(6,4)]
            goal_pos=(2,6)
            player_pos = (1,7)

    elif level == 9:
        ROWS, COLS = 100, 100
        player_pos = (0,0)
        goal_pos= (99,99)
        WIDTH, HEIGHT = 1200, 1200

 
        obstacle_positions= generate_obstacles(3000)
        elemento1=(29,25)
        indice1 = obstacle_positions.index(elemento1)
        obstacle_positions[indice1]=(29,26)

        elemento2=(68,97)
        indice2 = obstacle_positions.index(elemento2)
        obstacle_positions[indice2]=(69,97)

        elemento3=(99,98)
        indice3 = obstacle_positions.index(elemento3)
        obstacle_positions[indice3]=(99,96)


        obstaculos_nivel_9=obstacle_positions

    elif level == 10:
        ROWS, COLS = 150, 150
        player_pos=(0,0)
        goal_pos=(149,149)
        WIDTH, HEIGHT = 1500, 1500
        obstacle_positions= generate_obstacles(6750)

        elemento1=(97,147)
        indice1 = obstacle_positions.index(elemento1)
        obstacle_positions[indice1]=(99,148)


    if obstacle_positions:
        obstacles = obstacle_positions
    else:
        obstacles = None

    return player_pos, goal_pos, obstacles, ROWS, COLS


# Function to compute the movements of pieces
def move_piece(direction, player_pos, goal_pos, obstacles):
    x, y = player_pos

    if direction == 'up' and x > 0 and (x - 1, y) not in obstacles:
        x -= 1
    elif direction == 'down' and x < ROWS - 1 and (x + 1, y) not in obstacles:
        x += 1
    elif direction == 'left' and y > 0 and (x, y - 1) not in obstacles:
        y -= 1
    elif direction == 'right' and y < COLS - 1 and (x, y + 1) not in obstacles:
        y += 1

    player_pos = (x, y)
    return player_pos

# Function to let computer make a random move
def computer_random_move(player_pos, obstacles):
    possible_moves = []
    x, y = player_pos
    if (x, y - 1) not in obstacles and y > 0:
        possible_moves.append((x, y - 1))
    if (x, y + 1) not in obstacles and y < ROWS - 1:
        possible_moves.append((x, y + 1))
    if (x - 1, y) not in obstacles and x > 0:
        possible_moves.append((x - 1, y))
    if (x + 1, y) not in obstacles and x < COLS - 1:
        possible_moves.append((x + 1, y))
    return random.choice(possible_moves)

#Search Methods
# Function to let computer make a move using breadth-first search
def computer_bfs_move(player_pos, goal_pos, obstacles):
    queue = deque([(player_pos, [])])
    visited = set()

    while queue:
        pos, path = queue.popleft()
        if pos == goal_pos:
            return path
        if pos in visited:
            continue
        visited.add(pos)
        x, y = pos
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            new_pos = (x + dx, y + dy)
            if new_pos not in obstacles and 0 <= new_pos[0] < COLS and 0 <= new_pos[1] < ROWS:
                queue.append((new_pos, path + [new_pos]))

# Function to let computer make a move using depth-first search
def computer_dfs_move(player_pos, goal_pos, obstacles): 
    stack = []
    visited = set()
    nodes_explored = 0

    starting_state = (player_pos, [])
    stack.append(starting_state)
    visited.add(player_pos)

    while stack:
        if not stack:
            break
        nodes_explored += 1
        cur_pos, path = stack.pop()

        if cur_pos == goal_pos:
            return path

        x, y = cur_pos
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            new_pos = (x + dx, y + dy)
            if new_pos not in obstacles and 0 <= new_pos[0] < COLS and 0 <= new_pos[1] < ROWS and new_pos not in visited:
                stack.append((new_pos, path + [new_pos]))
                visited.add(new_pos)

    return []  # If no valid move found, return an empty list

# Function to let computer make a move using depth-limited search
def computer_depth_limited_move(player_pos, goal_pos, obstacles, depth_limit):
    def recursive_dls(node, depth, path, visited):
        if node == goal_pos:
            return path
        if depth == 0:
            return []

        visited.add(node)
        x, y = node
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            new_pos = (x + dx, y + dy)
            if new_pos not in obstacles and 0 <= new_pos[0] < COLS and 0 <= new_pos[1] < ROWS and new_pos not in visited:
                result = recursive_dls(new_pos, depth - 1, path + [new_pos], visited)
                if result:
                    return result
        return []

    return recursive_dls(player_pos, depth_limit, [player_pos], set())

# Function to let computer make a move using iterative-deepening search
def computer_iterative_deepening_move(player_pos, goal_pos, obstacles, max_depth):
    for depth_limit in range(1, max_depth + 1):
        result = computer_depth_limited_move(player_pos, goal_pos, obstacles, depth_limit)
        if result:
            return result
    return [] 


# Function to let computer make a move using uniform cost search
def computer_uniform_cost_move(player_pos, goal_pos, obstacles):
    heap = [(0, player_pos, [])]  # Priority queue (heap) with initial cost of 0
    visited = set()

    while heap:
        cost, pos, path = heapq.heappop(heap)  # Pop node with minimum cost
        if pos == goal_pos:
            return path
        if pos in visited:
            continue
        visited.add(pos)
        x, y = pos
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            new_pos = (x + dx, y + dy)
            if new_pos not in obstacles and 0 <= new_pos[0] < COLS and 0 <= new_pos[1] < ROWS:
                new_cost = cost + 1  # Uniform cost - increment by 1 for each step
                heapq.heappush(heap, (new_cost, new_pos, path + [new_pos]))  # Push node with updated cost

    return []  


# Heuristic function for Greedy Search 
def heuristic(current_pos, goal_pos):
    return abs(current_pos[0] - goal_pos[0]) + abs(current_pos[1] - goal_pos[1]) # (Manhattan distance)


# Function to let computer make a move using Greedy Search
def computer_greedy_move(player_pos, goal_pos, obstacles):
    frontier = deque([(player_pos, [])])
    visited = set()

    while frontier:
        pos, path = frontier.popleft()
        if pos == goal_pos:
            return path  
        if pos in visited:
            continue
        visited.add(pos)
        x, y = pos
        neighbors = [
            (x + dx, y + dy) for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]
            if (x + dx, y + dy) not in obstacles
            and 0 <= x + dx < COLS
            and 0 <= y + dy < ROWS
        ]
        neighbors.sort(key=lambda x: heuristic(x, goal_pos))
        for neighbor in neighbors:
            if neighbor not in visited:
                frontier.append((neighbor, path + [neighbor]))  

    return []

# Heuristic function for A* Search (Max Distance)
def heuristic_max_distance(current_pos, goal_pos):
    return max(abs(current_pos[0] - goal_pos[0]), abs(current_pos[1] - goal_pos[1]))

# Function to let computer make a move using A* Search with Max Distance heuristic
def computer_astar_max_distance_move(player_pos, goal_pos, obstacles):
    frontier = deque([(player_pos, [], 0)])
    visited = set()

    while frontier:
        pos, path, cost = frontier.popleft()
        if pos == goal_pos:
            return path  
        if pos in visited:
            continue
        visited.add(pos)
        x, y = pos
        neighbors = [
            (x + dx, y + dy) for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]
            if (x + dx, y + dy) not in obstacles
            and 0 <= x + dx < COLS
            and 0 <= y + dy < ROWS
        ]
        neighbors.sort(key=lambda x: heuristic_max_distance(x, goal_pos) + cost + 1)
        for neighbor in neighbors:
            if neighbor not in visited:
                frontier.append((neighbor, path + [neighbor], cost + 1))

    return []


# Heuristic function for A* Search 
def heuristic_astar(current_pos, goal_pos):
    return abs(current_pos[0] - goal_pos[0]) + abs(current_pos[1] - goal_pos[1]) #(Manhattan distance)

# Function to let computer make a move using A* Search
def computer_astar_move(player_pos, goal_pos, obstacles):
    frontier = deque([(player_pos, [], 0)])
    visited = set()

    while frontier:
        pos, path, cost = frontier.popleft()
        if pos == goal_pos:
            return path  
        if pos in visited:
            continue
        visited.add(pos)
        x, y = pos
        neighbors = [
            (x + dx, y + dy) for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]
            if (x + dx, y + dy) not in obstacles
            and 0 <= x + dx < COLS
            and 0 <= y + dy < ROWS
        ]
        neighbors.sort(key=lambda x: heuristic_astar(x, goal_pos) + cost + 1)
        for neighbor in neighbors:
            if neighbor not in visited:
                frontier.append((neighbor, path + [neighbor], cost + 1)) 

    return []


# Heuristic function for Weighted A* Search 
def heuristic_weighted_astar(current_pos, goal_pos, weight):
    return weight * (abs(current_pos[0] - goal_pos[0]) + abs(current_pos[1] - goal_pos[1])) #(Manhattan distance)

# Function to let computer make a move using Weighted A* Search
def computer_weighted_astar_move(player_pos, goal_pos, obstacles, weight):
    frontier = deque([(player_pos, [], 0)])
    visited = set()

    while frontier:
        pos, path, cost = frontier.popleft()
        if pos == goal_pos:
            return path  
        if pos in visited:
            continue
        visited.add(pos)
        x, y = pos
        neighbors = [
            (x + dx, y + dy) for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]
            if (x + dx, y + dy) not in obstacles
            and 0 <= x + dx < COLS
            and 0 <= y + dy < ROWS
        ]
        neighbors.sort(key=lambda x: heuristic_weighted_astar(x, goal_pos, weight) + cost + 1)
        for neighbor in neighbors:
            if neighbor not in visited:
                frontier.append((neighbor, path + [neighbor], cost + 1))  

    return []






#Metaheuristics


def hill_climbing(player_pos, goal_pos, obstacles):
    current_position = player_pos
    visited = set()  
    path = [current_position]
    while current_position != goal_pos:
        visited.add(current_position) 
        next_moves = []
        x, y = current_position
 
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            new_pos = (x + dx, y + dy)
            if new_pos not in obstacles and 0 <= new_pos[0] < COLS and 0 <= new_pos[1] < ROWS and new_pos not in path:
                next_moves.append((new_pos, heuristic(new_pos, goal_pos)))
        if not next_moves:
            break  
        next_moves.sort(key=lambda x: x[1])
        best_move = next_moves[0][0]
        if heuristic(best_move, goal_pos) >= heuristic(current_position, goal_pos):
            running=False  # If the best move doesn't improve the heuristic, break the loop
        current_position = best_move
        path.append(best_move)
    return path




#Simulated Annealing
def cost_function(current_pos, goal_pos):
    return math.sqrt((current_pos[0] - goal_pos[0])**2 + (current_pos[1] - goal_pos[1])**2)

def random_neighbor(current_pos, obstacles):

    x, y = current_pos
    moves = [(0, 1), (0, -1), (1, 0), (-1, 0)] 
    valid_moves = []
    for dx, dy in moves:
        new_x, new_y = x + dx, y + dy
        if (new_x, new_y) not in obstacles:
            valid_moves.append((new_x, new_y))
    return random.choice(valid_moves) if valid_moves else current_pos

def simulated_annealing(player_pos, goal_pos, obstacles, initial_temperature=1.0, cooling_rate=0.99, iterations=100):
    current_pos = player_pos
    best_pos = player_pos
    best_cost = cost_function(player_pos, goal_pos)
    path = [current_pos]
    current_cost = best_cost
    temperature = initial_temperature

    while temperature > 0.00001:
        for _ in range(iterations):
            next_pos = random_neighbor(current_pos, obstacles)
            next_cost = cost_function(next_pos, goal_pos)
            delta_E = next_cost - current_cost
            if delta_E < 0 or random.uniform(0, 1) < math.exp(-delta_E / temperature):
                current_pos = next_pos
                current_cost = next_cost
                if current_cost < best_cost:
                    best_pos = current_pos
                    best_cost = current_cost
            path.append(best_pos)
        temperature *= cooling_rate

    return path


def draw_hint(screen, player_pos, goal_pos, obstacles):
    next_move = computer_bfs_move(player_pos, goal_pos, obstacles)
    if next_move:
        x, y = player_pos
        next_x, next_y = next_move[0]
        if next_x == x and next_y == y - 1:
            return "UP"
        elif next_x == x and next_y == y + 1:
            return "DOWN"
        elif next_x == x - 1 and next_y == y:
            return "LEFT"
        elif next_x == x + 1 and next_y == y:
            return "RIGHT"
    return None


# Function to draw the "Hint" icon
def draw_hint_message(screen, next_move):
    if next_move:
        font_size = 20
        font = pygame.font.Font(None, font_size)
        hint_text = font.render(f"Next move: {next_move}", True, WHITE) 
        screen.blit(hint_text, (30, 10))  


help_button_image = pygame.image.load('button.png') #load icon image
help_button_rect = help_button_image.get_rect()
help_button_rect.topleft = (10, 10) 
help_button_image = pygame.transform.scale(help_button_image, (15, 15))
asking_for_hint = True
HINT_DISPLAY_TIME = 3 



# Main game loop
def main(level, search_method, menu):
    global ROWS, COLS, SQUARE_SIZE, asking_for_hint

    print("Level selected:", level)
    print("Search method selected:", search_method)

    player_pos, goal_pos, obstacles, ROWS, COLS = generate_positions(level)
    SQUARE_SIZE = WIDTH // COLS
    asking_for_hint = True

    moves = 0
    start_time = time.time()
    tracemalloc.start()


    running = True
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Seu Jogo")

    display_hint_message = False
    hint_timer = 0

    # Main game loop
    while running:
        screen.fill(BLUE)
        draw_grid()
        draw_pieces(player_pos, goal_pos, obstacles)
        screen.blit(help_button_image, help_button_rect)

        if display_hint_message and time.time() - hint_timer < HINT_DISPLAY_TIME:
            draw_hint_message(screen, next_move)

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    if help_button_rect.collidepoint(event.pos):
                        next_move = draw_hint(screen, player_pos, goal_pos, obstacles)
                        display_hint_message = True
                        hint_timer = time.time()
            elif event.type == pygame.KEYDOWN and search_method == 'human':
                if event.key == pygame.K_UP:
                    if player_pos[1] > 0 and (player_pos[0], player_pos[1] - 1) not in obstacles:
                        player_pos = (player_pos[0], player_pos[1] - 1)
                elif event.key == pygame.K_DOWN:
                    if player_pos[1] < ROWS - 1 and (player_pos[0], player_pos[1] + 1) not in obstacles:
                        player_pos = (player_pos[0], player_pos[1] + 1)
                elif event.key == pygame.K_LEFT:
                    if player_pos[0] > 0 and (player_pos[0] - 1, player_pos[1]) not in obstacles:
                        player_pos = (player_pos[0] - 1, player_pos[1])
                elif event.key == pygame.K_RIGHT:
                    if player_pos[0] < COLS - 1 and (player_pos[0] + 1, player_pos[1]) not in obstacles:
                        player_pos = (player_pos[0] + 1, player_pos[1])

                if player_pos == goal_pos:
                    print("Congratulations! You reached the goal.")
                    running = False
                

        if search_method == 'random':
            player_pos = computer_random_move(player_pos, obstacles)
            moves += 1

            if player_pos == goal_pos:
                print("The computer reached the goal.")
                running = False 


        if search_method == 'breadth_first':
            path = computer_bfs_move(player_pos, goal_pos, obstacles)
            if path:
                player_pos = path[0]
            moves += 1

            if player_pos == goal_pos:
                print("The computer reached the goal.")
                running = False
            elif not path:
                print("The game is unsolvable.")
                running = False

        elif search_method == 'depth_first':
            path = computer_dfs_move(player_pos, goal_pos, obstacles)
            if path:
                for step in path:
                    moves+=1
                    player_pos = step
                    # pygame.time.wait(int(DELAY * 1000))  
                    draw_grid()
                    draw_pieces(player_pos, goal_pos, obstacles)
                    pygame.display.flip()  
                if player_pos == goal_pos:
                    print("The computer reached the goal.")
                    running = False
            else:
                print("The game is unsolvable.")
                running = False

        elif search_method == 'greedy':
            steps = computer_greedy_move(player_pos, goal_pos, obstacles)
            if steps:
                for step in steps:
                    moves+=1
                    player_pos = step
                    #pygame.time.wait(int(DELAY * 1000)) 
                    draw_grid()
                    draw_pieces(player_pos, goal_pos, obstacles)
                    pygame.display.flip()  
                if player_pos == goal_pos:
                    print("The computer reached the goal.")
                    running = False
            else:
                print("The game is unsolvable.")
                running = False


        elif search_method == 'astar':
            steps = computer_astar_move(player_pos, goal_pos, obstacles)
            if steps:
                for step in steps:
                    moves+=1
                    player_pos = step
                    #pygame.time.wait(int(DELAY * 1000)) 
                    draw_grid()
                    draw_pieces(player_pos, goal_pos, obstacles)
                    pygame.display.flip()  
                if player_pos == goal_pos:
                    print("The computer reached the goal.")
                    running = False
            else:
                print("The game is unsolvable.")
                running = False

                
        elif search_method == 'astar2':
            steps = computer_astar_max_distance_move(player_pos, goal_pos, obstacles)
            if steps:
                for step in steps:
                    moves += 1
                    player_pos = step
                    #pygame.time.wait(int(DELAY * 1000)) 
                    draw_grid()
                    draw_pieces(player_pos, goal_pos, obstacles)
                    pygame.display.flip()  
                if player_pos == goal_pos:
                    print("The computer reached the goal.")
                    running = False
            else:
                print("The game is unsolvable.")
                running = False
                   
        elif search_method == 'weighted_astar':
            steps = computer_weighted_astar_move(player_pos, goal_pos, obstacles,4)
            if steps:
                for step in steps:
                    moves += 1
                    player_pos = step
                    #pygame.time.wait(int(DELAY * 1000)) 
                    draw_grid()
                    draw_pieces(player_pos, goal_pos, obstacles)
                    pygame.display.flip()  
                if player_pos == goal_pos:
                    print("The computer reached the goal.")
                    running = False
            else:
                print("The game is unsolvable.")
                running = False               
        elif search_method == 'depth_limited5':
            steps = computer_depth_limited_move(player_pos, goal_pos, obstacles, depth_limit=5)
            if steps:
                for step in steps:
                    moves+=1
                    player_pos = step
                    #pygame.time.wait(int(DELAY * 1000)) 
                    draw_grid()
                    draw_pieces(player_pos, goal_pos, obstacles)
                    pygame.display.flip()  
                if player_pos == goal_pos:
                    print("The computer reached the goal.")
                    running = False
            else:
                print("The game is unsolvable within the depth limit.")
                running = False


        elif search_method == 'depth_limited8':
            steps = computer_depth_limited_move(player_pos, goal_pos, obstacles, depth_limit=8)
            if steps:
                for step in steps:
                    moves+=1
                    player_pos = step
                    #pygame.time.wait(int(DELAY * 1000))  
                    draw_grid()
                    draw_pieces(player_pos, goal_pos, obstacles)
                    pygame.display.flip()  
                if player_pos == goal_pos:
                    print("The computer reached the goal.")
                    running = False
            else:
                print("The game is unsolvable within the depth limit.")
                running = False


        elif search_method == 'iterative_deepening':
            steps = computer_iterative_deepening_move(player_pos, goal_pos, obstacles, max_depth=10)
            if steps:
                for step in steps:
                    moves+=1
                    player_pos = step
                    #pygame.time.wait(int(DELAY * 1000))  
                    draw_grid()
                    draw_pieces(player_pos, goal_pos, obstacles)
                    pygame.display.flip()  

                if player_pos == goal_pos:
                    print("The computer reached the goal.")
                    running = False
            else:
                print("The game is unsolvable.")
                running = False


        elif search_method == 'uniform_cost':
            steps = computer_uniform_cost_move(player_pos, goal_pos, obstacles)
            if steps:
                for step in steps:
                    moves+=1
                    player_pos = step
                    #pygame.time.wait(int(DELAY * 1000))  
                    draw_grid()
                    draw_pieces(player_pos, goal_pos, obstacles)
                    pygame.display.flip()  

                if player_pos == goal_pos:
                    print("The computer reached the goal.")
                    running = False
            else:
                print("The game is unsolvable.")
                running = False

        elif search_method == 'simulated_annealing':
            steps = simulated_annealing(player_pos, goal_pos, obstacles)
            
            for step in steps:
                moves += 1
                player_pos = step
                draw_grid()
                draw_pieces(player_pos, goal_pos, obstacles)
                pygame.display.flip()

                if player_pos == goal_pos:
                    print("The computer reached the goal.")
                    running = False
                    break
                
        elif search_method == 'hill climbing':
            steps = hill_climbing(player_pos, goal_pos, obstacles)
            if steps:
                for step in steps:
                    moves+=1
                    player_pos = step
                    #pygame.time.wait(int(DELAY * 1000))  
                    draw_grid()
                    draw_pieces(player_pos, goal_pos, obstacles)
                    pygame.display.flip() 

                if player_pos == goal_pos:
                    print("The computer reached the goal.")
                    running = False
            else:
                print("The game is unsolvable.")
                running = False

                
        if display_hint_message and time.time() - hint_timer > HINT_DISPLAY_TIME:
            display_hint_message = False
        

    
            
    
    end_time = time.time()
    execution_time = end_time - start_time
    snapshot = tracemalloc.take_snapshot()

    peak = tracemalloc.get_tracemalloc_memory()

    print("Peak memory usage:", peak, "bytes")
    memory_usages.append(peak)
    tracemalloc.stop()

    
    print(f"Number of moves: {moves}")
    print(f"Execution time: {execution_time} seconds")
    execution_times.append(execution_time)
    num_moves.append(moves)
    
    
    mean_execution_time = sum(execution_times) / len(execution_times)
    mean_memory_usage = sum(memory_usages) / len(memory_usages)
    mean_num_moves = sum(num_moves) / len(num_moves)
    print()
    print("total list times is: ", execution_times)
    print("total list memory is: ", memory_usages)
    print("total list movesis: ", num_moves)

    print(f"Mean Execution Time: {mean_execution_time} seconds")
    print(f"Mean Memory Usage: {mean_memory_usage} KB")
    print(f"Mean Number of Moves: {mean_num_moves}")



    draw_congratulations_screen(menu)
    menu.mainloop(screen)


    pygame.quit()

# Main Pygame loop
def run_game():
    while True:
        menu = draw_level_selection_menu()
        menu.mainloop(screen)  # Start the menu loop
        if not menu.is_enabled():  # Check if menu is inactive (game has ended)
            break  # Exit the game loop when the menu is inactive

if __name__ == '__main__':
    run_game()
