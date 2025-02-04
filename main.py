import arcade
import random
import json
import pickle
import time
from collections import defaultdict
import matplotlib.pyplot as plt
import argparse

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

SCREEN_WIDTH = 800  # Augmenté pour accommoder le dashboard
SCREEN_HEIGHT = 800
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
            return False

        for point in car.points:
            new_x = point.x + move[0]
            new_y = point.y + move[1]
            if not (0 <= new_x < self.cols and 0 <= new_y < self.rows):
                return False

            if self.find_car(new_x, new_y) and self.find_car(new_x, new_y) != car:
                return False

        for point in car.points:
            point.x += move[0]
            point.y += move[1]
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
        cars = [Car("red", True, direction=self.config["main_car"]["direction"],
                    points=[Point(p["x"], p["y"]) for p in self.config["main_car"]["case"]])]
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
        self.all_scores = []

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

    def draw_dashboard(self):
        # Position de départ pour le dashboard
        start_x = SCREEN_WIDTH - 250
        start_y = SCREEN_HEIGHT - 50

        # Fond du dashboard
        arcade.draw_rectangle_filled(
            start_x + 125,
            SCREEN_HEIGHT - 100,
            240,
            500,
            arcade.make_transparent_color(arcade.color.DARK_BLUE, 200)
        )

        # Titre du dashboard
        arcade.draw_text(
            "STATISTIQUES",
            start_x + 50,
            start_y + 20,
            arcade.color.WHITE,
            16,
            bold=True
        )

        # Score actuel
        arcade.draw_text(
            f"Score: {self.score}",
            start_x + 10,
            start_y - 20,
            arcade.color.WHITE,
            14
        )

        if hasattr(self, 'stats'):
            stats_text = [
                f"Parties: {self.total_games}",
                f"Victoires: {self.total_wins}",
                f"Taux victoire: {self.stats['win_rate']:.1f}%",
                f"Score moyen: {self.stats['average_score']:.1f}",
                f"Temps moyen: {self.stats['average_time']:.1f}s",
                f"Meilleur score: {self.best_score}",
                f"Learning rate: {self.agent.learning_rate}",
                f"Discount: {self.agent.discount}",
                f"Epsilon: {self.agent.epsilon:.3f}",
                f"Epsilon decay: {self.agent.epsilon_decay:.3f}",
                f"Epsilon min: {self.agent.epsilon_min}",

            ]

            for i, text in enumerate(stats_text):
                arcade.draw_text(
                    text,
                    start_x + 10,
                    start_y - (50 + i * 25),
                    arcade.color.WHITE,
                    14
                )

    def on_draw(self):
        self.clear()
        arcade.start_render()

        self.draw_grass()
        self.draw_parking()
        self.draw_cars()
        self.draw_exit()
        self.draw_dashboard()


        arcade.draw_rectangle_filled(
            100,
            50,
            150,
            30,
            arcade.color.ORANGE
        )
        arcade.draw_text(
            "Save Q-table",
            50,
            40,
            arcade.color.WHITE,
            14,
            bold=True
        )

        arcade.draw_rectangle_filled(
            300,
            50,
            150,
            30,
            arcade.color.PURPLE
        )
        arcade.draw_text(
            "Load Q-table",
            250,
            40,
            arcade.color.WHITE,
            14,
            bold=True
        )

        # Bouton Reset (position originale)
        arcade.draw_rectangle_filled(
            500,
            50,
            150,
            30,
            arcade.color.RED
        )
        arcade.draw_text(
            "Reset",
            470,
            40,
            arcade.color.WHITE,
            14,
            bold=True
        )

        arcade.draw_rectangle_filled(
            100,
            SCREEN_HEIGHT - 30,
            150,
            30,
            arcade.color.BLUE,
        )
        arcade.draw_text(
            "Print Q-table",
            50,
            SCREEN_HEIGHT - 40,
            arcade.color.WHITE,
            14,
            bold=True,
        )


        arcade.draw_rectangle_filled(
            100,
            SCREEN_HEIGHT - 70,
            150,
            30,
            arcade.color.GREEN,
        )
        arcade.draw_text(
            "Show Plot",
            50,
            SCREEN_HEIGHT - 80,
            arcade.color.WHITE,
            14,
            bold=True,
        )


        arcade.draw_rectangle_filled(
            SCREEN_WIDTH - 100,
            50,
            150,
            30,
            arcade.color.RED if self.paused else arcade.color.GREEN
        )
        arcade.draw_text(
            "PAUSE" if not self.paused else "RESUME",
            SCREEN_WIDTH - 150,
            40,
            arcade.color.WHITE,
            14,
            bold=True
        )

        if self.win_message:
            arcade.draw_text(
                self.win_message,
                SCREEN_WIDTH // 2 - 100,
                SCREEN_HEIGHT // 2,
                arcade.color.YELLOW,
                24,
                bold=True
            )

    def on_mouse_release(self, x, y, button, modifiers):

        if 35 <= y <= 65:
            if 25 <= x <= 175:
                print("Saving Q-table...")
                self.agent.save_qtable(self.name_file_qtable)
                return


            elif 225 <= x <= 375:
                print("Loading Q-table...")
                if self.agent.load_qtable(self.name_file_qtable):
                    print("Q-table loaded successfully")
                else:
                    print("No Q-table file found")
                return


            elif 425 <= x <= 575:
                print("Resetting environment...")
                self.reset_episode()
                return


            elif SCREEN_WIDTH - 175 <= x <= SCREEN_WIDTH - 25:
                self.paused = not self.paused
                print(f"Game {'paused' if self.paused else 'resumed'}")
                return

        # Gestion des boutons de gauche
        if 25 <= x <= 175:
            # Print Q-table
            if SCREEN_HEIGHT - 45 <= y <= SCREEN_HEIGHT - 15:
                self.print_qtable()
                return

            # Show Plot
            elif SCREEN_HEIGHT - 85 <= y <= SCREEN_HEIGHT - 55:
                self.show_plot()
                return

        self.input = (x, y)

    def print_qtable(self):
        print("\nQ-Table Content:")
        for state in self.agent.q_table:
            print(f"\nState: {state}")
            for action, value in self.agent.q_table[state].items():
                print(f"  Action: {action}, Value: {value}")
            print("-" * 50)

    def show_plot(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.all_scores, label='Score')
        plt.xlabel('Episode')
        plt.ylabel('Score')
        plt.title('Evolution des scores')
        plt.legend()
        plt.show()


class RushHourAgent:
    def __init__(self, learning_rate=0.1, discount=0.99, epsilon=1.0, epsilon_decay=0.9995, epsilon_min=0.1):
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.discount = discount
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.available_moves = [65364, 65362, 65361, 65363]

    def get_state_key(self, parking):
        cars_info = []
        main_car = next(car for car in parking.cars if car.is_main)
        cars_info.append(self.get_string_car(main_car))

        other_cars = []
        for car in parking.cars:
            if not car.is_main:
                car_info = self.get_string_car(car)
                other_cars.append(car_info)

        other_cars.sort()
        cars_info.extend(other_cars)
        return str(cars_info)

    def get_string_car(self, car):
        if car.direction == 'H':
            return f"{car.color}({car.points[0].x},{car.points[-1].x}),H,{len(car.points)}"
        else:
            return f"{car.color}({car.points[0].y},{car.points[-1].y}),V,{len(car.points)}"

    def choose_action(self, state, parking):
        available_moves = self.get_available_moves(parking)

        if not available_moves:
            return None

        if random.random() < self.epsilon:
            return random.choice(available_moves)

        q_values = self.q_table[state]
        if not q_values:
            return random.choice(available_moves)

        best_value = float('-inf')
        best_actions = []

        for action in available_moves:
            value = q_values.get(action, 0)
            if value > best_value:
                best_value = value
                best_actions = [action]
            elif value == best_value:
                best_actions.append(action)

        return random.choice(best_actions)

    def get_available_moves(self, parking):
        available_moves = []
        for car in parking.cars:
            for move_key in self.available_moves:
                direction = MOVES_NUMBER[move_key]
                if car.direction == MOVEMENT_AXES[direction]:
                    move = MOVES[direction]
                    can_move = True

                    for point in car.points:
                        new_x = point.x + move[0]
                        new_y = point.y + move[1]

                        if not (0 <= new_x < parking.cols and 0 <= new_y < parking.rows):
                            can_move = False
                            break

                        other_car = parking.find_car(new_x, new_y)
                        if other_car and other_car != car:
                            can_move = False
                            break

                    if can_move:
                        available_moves.append((car.color, direction, move_key))

        return available_moves

    def learn(self, state, action, reward, next_state):
        if next_state is not None:
            best_next_value = max(self.q_table[next_state].values()) if self.q_table[next_state] else 0
            target = reward + self.discount * best_next_value
        else:
            target = reward

        old_value = self.q_table[state][action]
        self.q_table[state][action] = old_value + self.learning_rate * (target - old_value)

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

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
    def __init__(self, env, agent, mode='manual', name_file="qtable.pickle", reward=1000, step_penalty=-1,
                 max_steps=1000, speed=1):
        super().__init__(env)
        self.agent = agent
        self.mode = mode
        self.episode_steps = 0
        self.max_steps = max_steps
        self.current_car = None
        self.last_state = None
        self.last_action = None
        self.name_file_qtable = name_file
        self.reward = reward
        self.step_penalty = step_penalty
        self.last_update = time.time()
        self.speed = speed
        self.total_games = 0
        self.total_wins = 0
        self.start_time = time.time()
        self.current_game_start_time = time.time()
        self.all_completion_times = []
        self.best_score = 0
        self.stats = {
            'average_score': 0,
            'win_rate': 0,
            'average_time': 0,
            'best_time': float('inf'),
            'worst_time': 0,
            'best_score': float('inf'),
        }

    def update(self, delta_time):
        if not self.mode == 'auto' or self.paused:
            return

        if self.env.parking.you_win():
            completion_time = time.time() - self.current_game_start_time
            self.all_completion_times.append(completion_time)
            self.total_wins += 1
            self.total_games += 1
            self.update_statistics(completion_time)
            if self.score < self.best_score:
                self.best_score = self.score
            if self.last_state and self.last_action:
                self.agent.learn(self.last_state, self.last_action, self.reward, None)

            self.all_scores.append(self.score)
            print(f"Victoire ! Score: {self.score}, Steps: {self.episode_steps}, Temps: {completion_time:.2f}s")
            self.reset_episode()
            return

        if self.episode_steps >= self.max_steps:
            self.total_games += 1
            self.update_statistics(None)
            print(f"Maximum steps reached. Score: {self.score}")
            self.reset_episode()
            return

        current_state = self.agent.get_state_key(self.env.parking)
        action = self.agent.choose_action(current_state, self.env.parking)

        if action is None:
            self.reset_episode()
            return

        car_color, direction, move_key = action
        car = self.env.parking.find_car_color(car_color)

        if car:
            old_positions = self.env.parking.get_car_positions()
            moved = self.env.parking.move_car(car, move_key)

            if moved:
                reward = self.calculate_reward(car, old_positions)
                self.score += reward
                next_state = self.agent.get_state_key(self.env.parking)
                self.agent.learn(current_state, action, reward, next_state)
                self.last_state = current_state
                self.last_action = action
                self.episode_steps += 1

                self.clear()
                time.sleep(1 / self.speed)
                self.on_draw()

    def calculate_reward(self, car, old_positions):
        if self.env.parking.you_win():
            return self.reward
        return self.step_penalty

    def update_statistics(self, completion_time):
        if self.all_scores:
            self.stats['average_score'] = sum(self.all_scores) / len(self.all_scores)

        self.stats['win_rate'] = (self.total_wins / self.total_games) * 100 if self.total_games > 0 else 0

        if completion_time is not None:
            if self.all_completion_times:
                self.stats['average_time'] = sum(self.all_completion_times) / len(self.all_completion_times)
                self.stats['best_time'] = min(self.all_completion_times)
                self.stats['worst_time'] = max(self.all_completion_times)

    def reset_episode(self):
        self.episode_steps = 0
        self.score = 0
        self.win_message = ""
        self.env.setup()
        self.current_game_start_time = time.time()
        self.clear()
        self.on_draw()


def main():
    parser = argparse.ArgumentParser(description='Rush Hour Game')
    parser.add_argument('level', type=str, help='Path to level file (ex: level1.json)')
    parser.add_argument('--qtable_load', type=str, default='', help='Path to load qtable')
    parser.add_argument('--qtable_save', type=str, default='qtable.pickle', help='Path to save qtable')
    parser.add_argument('--mode', type=str, choices=['manual', 'auto'], default='manual', help='Game mode')
    parser.add_argument('--learning_rate', type=float, default=0.2, help='Learning rate')
    parser.add_argument('--discount', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--epsilon', type=float, default=0.5, help='Epsilon value')
    parser.add_argument('--epsilon_decay', type=float, default=0.9999, help='Epsilon decay rate')
    parser.add_argument('--epsilon_min', type=float, default=0.05, help='Minimum epsilon value')
    parser.add_argument('--max_steps', type=int, default=1000, help='Maximum steps per episode')
    parser.add_argument('--reward', type=int, default=1000, help='Reward for winning')
    parser.add_argument('--step_penalty', type=int, default=-1, help='Penalty per step')
    parser.add_argument('--speed', type=float, default=100.0,
                        help='Game speed (10.0 = slower, 50 = medium, 100.0 = faster)')

    args = parser.parse_args()

    env = Environment(args.level)
    env.setup()

    agent = RushHourAgent(
        learning_rate=args.learning_rate,
        discount=args.discount,
        epsilon=args.epsilon,
        epsilon_decay=args.epsilon_decay,
        epsilon_min=args.epsilon_min,
    )



    if args.qtable_load != '':
        agent.load_qtable(args.qtable_load)

    RushHourGameAI(
        env,
        agent,
        args.mode,
        args.qtable_save,
        reward=args.reward,
        step_penalty=args.step_penalty,
        max_steps=args.max_steps,
        speed=args.speed,
    )

    arcade.run()


if __name__ == "__main__":
    main()