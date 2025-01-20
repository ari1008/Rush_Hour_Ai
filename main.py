import arcade
import random
import json
import pickle
from collections import defaultdict
import matplotlib.pyplot as plt


SPRITE_SIZE = 64

# Actions
MOVES = {
    'UP': (0, -1),  # Haut : diminue y
    'DOWN': (0, 1),  # Bas : augmente y
    'LEFT': (-1, 0),  # Gauche : diminue x
    'RIGHT': (1, 0)  # Droite : augmente x
}

MOVES_NUMBER = {
    65364: 'UP',  # Flèche haut
    65362: 'DOWN',  # Flèche bas
    65361: 'LEFT',  # Flèche gauche
    65363: 'RIGHT'  # Flèche droite
}

MOVES_STR = {
    'UP': 65364,
    'DOWN': 65362,
    'LEFT': 65361,
    'RIGHT': 65363
}

MOVEMENT_AXES = {
    'UP': 'V',
    'DOWN': 'V',
    'LEFT': 'H',
    'RIGHT': 'H'
}

SCREEN_WIDTH = 700
SCREEN_HEIGHT = 700
SCREEN_TITLE = "Rush Hour"
TILE_SIZE = 80
PARKING_OFFSET = 70


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


def get_color_enum(color_name):
    try:
        return arcade.color.__dict__[color_name.upper()]
    except KeyError:
        raise ValueError(f"Color '{color_name}' not found in arcade.color")


class Car:
    def __init__(self, color, is_main=False, direction="V", points=None):
        if points is None:
            points = []
        self.color: arcade.color = get_color_enum(color)
        self.direction = direction
        self.is_main = is_main
        self.points = points


class Parking:
    def __init__(self, cols, rows, exit_point, cars=None):
        if cars is None:
            cars = []
        self.cols = cols
        self.rows = rows
        self.exit_point = exit_point
        self.cars = cars

    def get_car_positions(self):
        car_positions = []
        for car in self.cars:
            for point in car.points:
                car_positions.append(PositionCar(car, point.x, point.y))
        return car_positions

    def find_car(self, x, y):
        for car in self.cars:
            for point in car.points:
                if point.x == x and point.y == y:
                    return car
        return None

    def find_car_color(self, color):
        for car in self.cars:
            if car.color == color:
                return car
        return None

    def you_win(self):
        for car in self.cars:
            if car.is_main:
                for point in car.points:
                    # Vérifiez si la voiture principale atteint la sortie
                    if (point.x == self.exit_point.x and point.y == self.exit_point.y) or \
                            (point.x == self.exit_point.x and point.y + 1 == self.exit_point.y) or \
                            (point.x == self.exit_point.x and point.y - 1 == self.exit_point.y) or \
                            (point.x + 1 == self.exit_point.x and point.y == self.exit_point.y) or \
                            (point.x - 1 == self.exit_point.x and point.y == self.exit_point.y):
                        return True
        return False

    def move_car(self, car, move_number):
        direction = MOVES_NUMBER[move_number]
        move = MOVES[direction]
        if car.direction != MOVEMENT_AXES[direction]:
            print("Mouvement non autorisé !")
            return False

        for point in car.points:
            new_x = point.x + move[0]
            new_y = point.y + move[1]
            if not (0 <= new_x < self.cols and 0 <= new_y < self.rows):
                print("Mouvement hors des limites !")
                return False

            if self.find_car(new_x, new_y) and self.find_car(new_x, new_y) != car:
                print("Collision détectée !")
                return False

        for point in car.points:
            point.x += move[0]
            point.y += move[1]
        #print(f"Car {car.color} moved {direction}")
        return True


class PositionCar:
    def __init__(self, car, x, y):
        self.car = car
        self.x = x
        self.y = y


class Environment:
    def __init__(self, config_file):
        self.config = None
        self.parking = None
        self.load_config(config_file)

    def load_config(self, config_file):
        with open(config_file, 'r') as f:
            self.config = json.load(f)

    def create_parking(self):
        parking = Parking(
            self.config["map"]["height"],
            self.config["map"]["width"],
            Point(self.config["exit"]["x"], self.config["exit"]["y"]),
            self.create_cars()
        )
        return parking

    def create_cars(self):
        cars = [Car("red", True, direction=self.config["main_car"]["direction"], points=[Point(p["x"], p["y"]) for p in self.config["main_car"]["case"]])]
        for car in self.config["cars"]:
            new_car = Car(
                color=car["color"],
                is_main=False,
                direction=car["direction"],
                points=[Point(p["x"], p["y"]) for p in car["case"]]
            )
            cars.append(new_car)

        return cars

    def setup(self):
        self.parking = self.create_parking()

    def to_string(self):
        for row in range(self.parking.rows):
            for col in range(self.parking.cols):
                if any([car.x == col and car.y == row for car in self.parking.get_car_positions()]):
                    print("X", end="")
                else:
                    print(".", end="")
            print()


class RushHourGame(arcade.Window):
    def __init__(self, env):
        super().__init__(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE)
        self.env = env
        arcade.set_background_color(arcade.color.DARK_SPRING_GREEN)
        self.input = None
        self.score = 0
        self.agent_state = ""
        self.win_message = ""
        self.game_over = False
        self.paused = False
        self.all_scores = []  # Store all scores

    def on_draw(self):
        """Affiche les éléments du jeu."""
        self.clear()
        arcade.start_render()

        self.draw_grass()
        self.draw_parking()
        self.draw_cars()
        self.draw_exit()

        arcade.draw_text(
            f"Score: {self.score}",
            10,
            SCREEN_HEIGHT - 30,
            arcade.color.WHITE,
            16,
            bold=True,
        )

        arcade.draw_text(
            f"Agent State: {self.agent_state}",
            10,
            SCREEN_HEIGHT - 60,
            arcade.color.WHITE,
            16,
            bold=True,
        )

        arcade.draw_text(
            f"All Scores: {self.all_scores}",
            10,
            SCREEN_HEIGHT - 90,
            arcade.color.WHITE,
            16,
            bold=True,
        )

        if self.win_message:
            arcade.draw_text(
                self.win_message,
                SCREEN_WIDTH // 2 - 100,
                SCREEN_HEIGHT // 2,
                arcade.color.YELLOW,
                24,
                bold=True,
            )

        if self.game_over:
            arcade.draw_text(
                self.win_message,
                SCREEN_WIDTH // 2 - 100,
                SCREEN_HEIGHT // 2,
                arcade.color.YELLOW,
                24,
                bold=True,
            )
            arcade.draw_text(
                f"Final Score: {self.score}",
                SCREEN_WIDTH // 2 - 80,
                SCREEN_HEIGHT // 2 - 40,
                arcade.color.WHITE,
                20,
                bold=True,
            )

    def draw_parking(self):
        arcade.draw_rectangle_filled(
            TILE_SIZE * self.env.parking.cols / 2 + PARKING_OFFSET,
            TILE_SIZE * self.env.parking.rows / 2 + PARKING_OFFSET,
            TILE_SIZE * self.env.parking.cols,
            TILE_SIZE * self.env.parking.rows,
            arcade.color.GRAY,
        )
        for row in range(self.env.parking.rows + 1):
            y = row * TILE_SIZE + PARKING_OFFSET
            arcade.draw_line(
                PARKING_OFFSET,
                y,
                PARKING_OFFSET + self.env.parking.rows * TILE_SIZE,
                y,
                arcade.color.BLACK,
                2,
            )
        for col in range(self.env.parking.cols + 1):
            x = col * TILE_SIZE + PARKING_OFFSET
            arcade.draw_line(
                x,
                PARKING_OFFSET,
                x,
                PARKING_OFFSET + self.env.parking.cols * TILE_SIZE,
                arcade.color.BLACK,
                2,
            )

    def draw_cars(self):
        for car in self.env.parking.cars:
            for point in car.points:
                x = point.x * TILE_SIZE + TILE_SIZE / 2 + PARKING_OFFSET
                y = point.y * TILE_SIZE + TILE_SIZE / 2 + PARKING_OFFSET
                arcade.draw_rectangle_filled(x, y, TILE_SIZE - 10, TILE_SIZE - 10, car.color)

    def draw_grass(self):
        arcade.draw_rectangle_filled(
            SCREEN_WIDTH / 2,
            SCREEN_HEIGHT / 2,
            SCREEN_WIDTH,
            SCREEN_HEIGHT,
            arcade.color.DARK_SPRING_GREEN,
        )

    def draw_exit(self):
        x = self.env.parking.exit_point.x * TILE_SIZE + PARKING_OFFSET
        y = self.env.parking.exit_point.y * TILE_SIZE + TILE_SIZE / 2 + PARKING_OFFSET
        arcade.draw_rectangle_filled(x + TILE_SIZE / 2, y, TILE_SIZE, TILE_SIZE, arcade.color.LIGHT_RED_OCHRE)
        arcade.draw_text("EXIT", x + TILE_SIZE / 2 - 20, y - 10, arcade.color.BLACK, 16, bold=True)

    def on_mouse_release(self, x, y, button, modifiers):
        self.input = (x, y)

    def on_key_press(self, key, modifiers):
        if key in MOVES_NUMBER and self.input is not None:
            x, y = self.input
            col = (x - PARKING_OFFSET) // TILE_SIZE
            row = (y - PARKING_OFFSET) // TILE_SIZE
            car: Car = self.env.parking.find_car(col, row)
            if car is not None:
                self.env.parking.move_car(car, key)
                #print(f"Car {car.color} at {col}, {row} moves {MOVES_NUMBER[key]}")
                self.clear()
                self.on_draw()
                if car.is_main and self.env.parking.you_win():
                    print("Vous avez gagné !")
                    self.show_win_message()

    def show_win_message(self):
        """Affiche un message de victoire."""

        self.clear()
        arcade.start_render()
        arcade.draw_text(
            "Bravo, vous avez gagné !",
            SCREEN_WIDTH / 2 - 150,
            SCREEN_HEIGHT / 2,
            arcade.color.WHITE,
            24,
            align="center",
            anchor_x="center",
            anchor_y="center",
            width=300
        )
        arcade.pause(3)
        arcade.close_window()


class RushHourAgent:
    def __init__(self):
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.discount = 0.99  # Augmenter pour favoriser les récompenses futures
        self.learning_rate = 0.1  # Réduire pour plus de stabilité
        self.epsilon = 1.0
        self.epsilon_decay = 0.9995  # Décroissance plus lente
        self.epsilon_min = 0.1
        self.available_moves = [65364, 65362, 65361, 65363]  # UP, DOWN, LEFT, RIGHT

    def get_state_key(self, parking):
        cars_info = []

        main_car = next(car for car in parking.cars if car.is_main)
        cars_info.append(self.get_string_car(main_car))

        other_cars = []
        for car in parking.cars:
            if not car.is_main:
                car_info = self.get_string_car(car)
                other_cars.append(car_info)

        other_cars.sort(key=lambda x: x[0])
        cars_info.extend(other_cars)

        return str(cars_info)


    def get_string_car(self, car):
        if car.direction == 'H':
            return f"{car.color}({car.points[0].x},{car.points[-1].x}),H,{len(car.points)}"
        else:
            return f"{car.color}({car.points[0].y},{car.points[-1].y}),V,{len(car.points)}"


    def is_blocking_exit(self, car, main_car, parking):
        """Vérifie si une voiture bloque le chemin vers la sortie."""
        if car.is_main:
            return False

        exit_x = parking.exit_point.x
        exit_y = parking.exit_point.y
        main_positions = [(p.x, p.y) for p in main_car.points]
        car_positions = [(p.x, p.y) for p in car.points]

        # Vérifie si la voiture est entre la voiture principale et la sortie
        for pos in car_positions:
            if main_car.direction == 'H':
                if (pos[1] == main_positions[0][1] and
                        min(main_positions[0][0], exit_x) <= pos[0] <= max(main_positions[0][0], exit_x)):
                    return True
            else:
                if (pos[0] == main_positions[0][0] and
                        min(main_positions[0][1], exit_y) <= pos[1] <= max(main_positions[0][1], exit_y)):
                    return True
        return False

    def distance_to_main_car(self, car, main_car):
        """Calcule la distance minimale entre une voiture et la voiture principale."""
        min_distance = float('inf')
        for p1 in car.points:
            for p2 in main_car.points:
                dist = abs(p1.x - p2.x) + abs(p1.y - p2.y)
                min_distance = min(min_distance, dist)
        return min_distance

    def count_empty_spaces(self, parking):
        """Compte le nombre de cases vides dans le parking."""
        occupied = set()
        for car in parking.cars:
            for point in car.points:
                occupied.add((point.x, point.y))
        total_spaces = parking.rows * parking.cols
        return total_spaces - len(occupied)

    def is_exit_blocked(self, parking):
        """Vérifie si la sortie est bloquée par d'autres voitures."""
        exit_x = parking.exit_point.x
        exit_y = parking.exit_point.y

        # Vérifie les cases adjacentes à la sortie
        adjacent_positions = [
            (exit_x + 1, exit_y),
            (exit_x - 1, exit_y),
            (exit_x, exit_y + 1),
            (exit_x, exit_y - 1)
        ]

        blocked_count = 0
        for x, y in adjacent_positions:
            if 0 <= x < parking.cols and 0 <= y < parking.rows:
                car = parking.find_car(x, y)
                if car and not car.is_main:
                    blocked_count += 1

        return blocked_count

    def choose_action(self, state, parking):
        available_moves = self.available_moves_cars(parking)

        if not available_moves:
            return None

        if random.random() < self.epsilon:
            return random.choice(available_moves)

        # Obtenir les Q-valeurs pour l'état actuel
        q_values = self.q_table[state]

        # Si aucune Q-valeur n'existe pour cet état, choisir aléatoirement
        if not q_values:
            return random.choice(available_moves)


        available_moves_tuples = [(color, direction, value) for color, direction, value in available_moves]

        best_action = None
        best_value = float('-inf')

        for action_tuple in available_moves_tuples:
            if action_tuple in q_values:
                value = q_values[action_tuple]
                if value > best_value:
                    best_value = value
                    best_action = action_tuple

        # Si aucune action connue n'est trouvée parmi les mouvements disponibles
        if best_action is None:
            return random.choice(available_moves)

        return best_action

    def available_moves_cars(self, parking):
        possible_moves = []

        for direction, move_vector in MOVES.items():
            for car in parking.cars:
                if car.direction == MOVEMENT_AXES[direction]:
                    can_move = True
                    car_points = self.get_car_points(car)
                    for x, y in car_points:
                        new_x = x + move_vector[0]
                        new_y = y + move_vector[1]

                        if not (0 <= new_x < parking.rows and 0 <= new_y < parking.rows):
                            can_move = False
                            break

                        if self.is_position_occupied(parking, new_x, new_y, car):
                            can_move = False
                            break

                    if can_move:
                        possible_moves.append((car.color, direction, 1))

        return possible_moves

    def get_car_points(self, car):
        points = []
        if car.direction == 'H':
            for point in car.points:
                points.append((point.x, point.y))
        else:  # 'V'
            for point in car.points:
                points.append((point.x, point.y))
        return points

    def is_position_occupied(self, parking, x, y, excluded_car):
        for other_car in parking.cars:
            if other_car != excluded_car:
                if other_car.direction == 'H':
                    for point in other_car.points:
                        if point.y == y and point.x == x:
                            return True
                else:  # 'V'
                    for point in other_car.points:
                        if point.x == x and point.y == y:
                            return True
        return False


    def learn(self, state, action, reward, next_state):
        best_next_value = max(self.q_table[next_state].values()) if self.q_table[next_state] else 0
        old_value = self.q_table[state][action]
        self.q_table[state][action] = old_value + self.learning_rate * (
                reward + self.discount * best_next_value - old_value)
        self.epsilon = max(0.0, self.epsilon * 0.999)  # Reduce epsilon over time

    def save_qtable(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(dict(self.q_table), f)

    def load_qtable(self, filename):
        try:
            with open(filename, 'rb') as f:
                loaded_dict = pickle.load(f)
                self.q_table = defaultdict(lambda: defaultdict(float), loaded_dict)
            return True
        except FileNotFoundError:
            return False


class RushHourGameAI(RushHourGame):
    def __init__(self, env, agent, mode='manual'):
        super().__init__(env)
        self.agent = agent
        self.mode = mode  # 'manual' or 'auto'
        self.current_car = None
        self.episode_steps = 0
        self.max_steps = 1000
        self.training = True
        self.last_state = None
        self.last_action = None
        self.score = 0
        self.agent_state = ""
        self.win_message = ""
        self.game_over = False
        self.paused = False
        self.best_score = int(1e9)
        self.all_scores = []
        self.reward = 0

    def update(self, delta_time):
        if not self.mode == 'auto' or self.paused:
            return

        if self.env.parking.you_win():
            self.agent.save_qtable('qtable.pickle')
            if self.last_state and self.last_action:
                self.agent.learn(self.last_state, self.last_action, 700, None)
            if self.score < self.best_score:
                self.best_score = self.score
            self.all_scores.append(self.score)
            print(f"Victoire ! Score: {self.score}, Steps: {self.episode_steps}")
            self.reset_episode()
            return

        if self.episode_steps >= self.max_steps:
            print(f"Nombre maximum d'étapes atteint. Score: {self.score}")
            self.reset_episode()
            return

        # Sauvegarde des positions avant le mouvement
        old_positions = self.env.parking.get_car_positions()

        # Obtention de l'état actuel
        current_state = self.agent.get_state_key(self.env.parking)

        # Choix de l'action
        action = self.agent.choose_action(current_state, self.env.parking)
        if not action:  # Si aucune action n'est possible
            self.reset_episode()
            return

        car_color, direction, _ = action
        car = self.env.parking.find_car_color(car_color)

        # Exécution du mouvement
        if car:
            move_key = MOVES_STR[direction]
            moved = self.env.parking.move_car(car, move_key)

            if moved:
                # Calcul de la récompense et mise à jour du score
                reward = self.calculate_reward(car, old_positions)
                self.score += reward

                # Obtention du nouvel état
                next_state = self.agent.get_state_key(self.env.parking)

                # Apprentissage
                self.agent.learn(current_state, action, reward, next_state)

                # Mise à jour des variables de suivi
                self.last_state = current_state
                self.last_action = action
                self.episode_steps += 1
                self.agent_state = f"État: {current_state[:30]}..."

                # Rafraîchissement de l'affichage
                self.clear()
                self.on_draw()

    def find_movable_cars(self, move):
        """Find all cars that can be moved in the given direction."""
        movable_cars = []
        for car in self.env.parking.cars:
            if car.direction == MOVEMENT_AXES[MOVES_NUMBER[move]]:
                can_move = True
                move_vector = MOVES[MOVES_NUMBER[move]]
                for point in car.points:
                    new_x = point.x + move_vector[0]
                    new_y = point.y + move_vector[1]
                    if not (0 <= new_x < self.env.parking.cols and 0 <= new_y < self.env.parking.rows):
                        can_move = False
                        break
                    other_car = self.env.parking.find_car(new_x, new_y)
                    if other_car and other_car != car:
                        can_move = False
                        break
                if can_move:
                    movable_cars.append(car)
        return movable_cars



    def calculate_reward(self, car, old_positions):
        """Calculate the reward for the current action."""
        if self.env.parking.you_win():
            return 700
        return -1

    def reset_episode(self):
        """Reset the environment for a new episode."""
        self.episode_steps = 0
        self.score = 0
        self.win_message = ""
        self.env.setup()
        self.clear()
        self.on_draw()

    def toggle_mode(self):
        """Switch between manual and auto mode."""
        self.mode = 'auto' if self.mode == 'manual' else 'manual'
        print(f"Switched to {self.mode} mode")

    def on_key_press(self, key, modifiers):
        if key == arcade.key.T:  # 'T' to toggle mode
            self.toggle_mode()
        elif key == arcade.key.S:  # 'S' to save Q-table
            self.agent.save_qtable('qtable.pickle')
            print("Q-table saved")
        elif key == arcade.key.L:  # 'L' to load Q-table
            if self.agent.load_qtable('qtable.pickle'):
                print("Q-table loaded")
            else:
                print("No saved Q-table found")
        elif self.mode == 'manual':
            super().on_key_press(key, modifiers)


    def draw_qtable_button(self):
        # Dessine le bouton "Print Q-table"
        arcade.draw_rectangle_filled(
            SCREEN_WIDTH - 100,
            SCREEN_HEIGHT - 30,
            150,
            30,
            arcade.color.BLUE,
        )
        arcade.draw_text(
            "Print Q-table",
            SCREEN_WIDTH - 150,
            SCREEN_HEIGHT - 40,
            arcade.color.WHITE,
            14,
            bold=True,
        )

    def on_draw(self):
        super().on_draw()
        self.draw_qtable_button()

    def print_qtable(self):
        print("\nQ-Table Content:")
        for state in self.agent.q_table:
            print(f"\nState: {state}")
            for action, value in self.agent.q_table[state].items():
                print(f"Action: {MOVES_NUMBER[action]}, Value: {value:.2f}")
            print("-" * 50)

    def on_mouse_release(self, x, y, button, modifiers):
        # Vérifie si le clic est dans la zone du bouton
        if (SCREEN_WIDTH - 175 <= x <= SCREEN_WIDTH - 25 and
                SCREEN_HEIGHT - 45 <= y <= SCREEN_HEIGHT - 15):
            self.print_qtable()
        else:
            super().on_mouse_release(x, y, button, modifiers)

    def draw_plot_button(self):
        # Dessine le bouton "Show Plot"
        arcade.draw_rectangle_filled(
            SCREEN_WIDTH - 100,
            SCREEN_HEIGHT - 70,
            150,
            30,
            arcade.color.GREEN,
        )
        arcade.draw_text(
            "Show Plot",
            SCREEN_WIDTH - 150,
            SCREEN_HEIGHT - 80,
            arcade.color.WHITE,
            14,
            bold=True,
        )

    def on_draw(self):
        super().on_draw()
        self.draw_qtable_button()
        self.draw_plot_button()

    def show_plot(self):
        plt.figure()
        plt.plot(self.all_scores)
        plt.xlabel("Episodes")
        plt.ylabel("Score")
        plt.title("Score per Episode")
        plt.show()

    def on_mouse_release(self, x, y, button, modifiers):
        # Vérifie si le clic est dans la zone du bouton Q-table
        if (SCREEN_WIDTH - 175 <= x <= SCREEN_WIDTH - 25 and
                SCREEN_HEIGHT - 45 <= y <= SCREEN_HEIGHT - 15):
            self.print_qtable()
        # Vérifie si le clic est dans la zone du bouton Plot
        elif (SCREEN_WIDTH - 175 <= x <= SCREEN_WIDTH - 25 and
              SCREEN_HEIGHT - 85 <= y <= SCREEN_HEIGHT - 55):
            self.show_plot()
        else:
            super().on_mouse_release(x, y, button, modifiers)




def main():
    env = Environment("levels/level3.json")
    env.setup()
    agent = RushHourAgent()
    agent.load_qtable('qtable.pickle')

    window = RushHourGameAI(env, agent, mode='manual')

    arcade.run()

    plt.plot(window.all_scores)
    plt.xlabel("Steps")
    plt.ylabel("Score")
    plt.title("Score per Steps")
    plt.show()


if __name__ == "__main__":
    main()