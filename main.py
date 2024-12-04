import arcade
import random
import json
import pickle
from collections import defaultdict
import time

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

    def __init__(self, color, is_main=False, direction="V", points=[]):
        self.color: arcade.color = get_color_enum(color)
        self.direction = direction
        self.is_main = is_main
        self.points = points



class Parking:
    def __init__(self, cols, rows, exit_point, cars=[]):
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
            arcade.color.AERO_BLUE
            return


        for point in car.points:
            new_x = point.x + move[0]
            new_y = point.y + move[1]
            if not (0 <= new_x < self.cols and 0 <= new_y < self.rows):
                print("Mouvement hors des limites !")
                return


            if self.find_car(new_x, new_y) and self.find_car(new_x, new_y) != car:
                print("Collision détectée !")
                return

        for point in car.points:
            point.x += move[0]
            point.y += move[1]
        print(f"Car {car.color} moved {direction}")


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
        cars = [Car("red", True, direction=self.config["main_car"]["direction"],points=[Point(p["x"], p["y"]) for p in self.config["main_car"]["case"]])]
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
                print(f"Car {car.color} at {col}, {row} moves {MOVES_NUMBER[key]}")
                self.clear()
                self.on_draw()
                if car.is_main   and self.env.parking.you_win():
                    print("Vous avez gagné !")
                    #self.show_win_message()

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
        #arcade.pause(3)
        arcade.close_window()

class RushHourAgent:
    def __init__(self, discount=0.95, learning_rate=0.001, epsilon=1):
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.discount = discount
        self.learning_rate = learning_rate
        #self.epsilon = epsilon
        self.epsilon = max(0.1, epsilon * 0.99) 
        self.available_moves = [65364, 65362, 65361, 65363]  # UP, DOWN, LEFT, RIGHT

    def get_state_key(self, parking):
        state = []
        for car in parking.cars:
            car_state = []
            for point in car.points:
                car_state.append(f"{point.x},{point.y}")
            state.append("|".join(car_state))
        return "_".join(state)

    def choose_action(self, state, parking):
        if random.random() < self.epsilon:
            return random.choice(self.available_moves)

        q_values = self.q_table[state]
        if not q_values:
            return random.choice(self.available_moves)

        return max(q_values.items(), key=lambda x: x[1])[0]

    def learn(self, state, action, reward, next_state):
        best_next_value = max(self.q_table[next_state].values()) if self.q_table[next_state] else 0
        old_value = self.q_table[state][action]
        self.q_table[state][action] = old_value + self.learning_rate * (
                reward + self.discount * best_next_value - old_value)

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

    def update(self, delta_time):
        """Update game state for automatic mode"""
        if self.mode == 'auto' and self.training:
            if self.episode_steps >= self.max_steps:
                self.reset_episode()
                return

            state = self.agent.get_state_key(self.env.parking)
            self.agent_state = state
            # Choose and execute action
            action = self.agent.choose_action(state, self.env.parking)

            # Find a random car that can move in the chosen direction
            movable_cars = self.find_movable_cars(action)
            if movable_cars:
                car = random.choice(movable_cars)
                old_positions = [(p.x, p.y) for p in car.points]

                # Try to move the car
                self.env.parking.move_car(car, action)

                # Calculate reward
                new_state = self.agent.get_state_key(self.env.parking)
                reward = self.calculate_reward(car, old_positions)

                self.score += reward
                # Learn from this action
                self.agent.learn(state, action, reward, new_state)

                if self.env.parking.you_win():
                    print("AI won!")
                    reward = 100
                    self.score += reward
                    self.agent.learn(state, action, reward, new_state)
                    self.reset_episode()

            self.episode_steps += 1

            self.clear()
            self.on_draw()
            #time.sleep(0.1)

    def find_movable_cars(self, move):
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
        if self.env.parking.you_win():
            return 10

        if car.is_main:
            old_distance = self.calculate_distance_to_exit(old_positions)
            new_distance = self.calculate_distance_to_exit([(p.x, p.y) for p in car.points])
            if new_distance < old_distance:
                return -10
            elif new_distance > old_distance:
                return -50

        return -10

    def calculate_distance_to_exit(self, positions):
        exit_x = self.env.parking.exit_point.x
        exit_y = self.env.parking.exit_point.y
        return min(abs(pos[0] - exit_x) + abs(pos[1] - exit_y) for pos in positions)

    def reset_episode(self):
        self.episode_steps = 0
        self.env.setup()
        self.clear()
        self.on_draw()

    def toggle_mode(self):
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


def main():
    env = Environment("levels/level1.json")
    env.setup()
    agent = RushHourAgent()
    agent.load_qtable('qtable.pickle')

    window = RushHourGameAI(env, agent, mode='manual')

    arcade.run()


if __name__ == "__main__":
    main()


