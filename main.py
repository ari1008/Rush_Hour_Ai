import json
import os
import arcade

SPRITE_SIZE = 64

# Actions
MOVES = {
    'UP': (0, -1),  # Haut : diminue y
    'DOWN': (0, 1),  # Bas : augmente y
    'LEFT': (-1, 0),  # Gauche : diminue x
    'RIGHT': (1, 0)  # Droite : augmente x
}

# Codes des touches directionnelles (par exemple pour Pygame ou Arcade)
MOVES_NUMBER = {
    65364: 'UP',  # Flèche haut
    65362: 'DOWN',  # Flèche bas
    65361: 'LEFT',  # Flèche gauche
    65363: 'RIGHT'  # Flèche droite
}

# Axes associés aux directions (Horizontal ou Vertical)
MOVEMENT_AXES = {
    'UP': 'V',  # Haut = axe vertical
    'DOWN': 'V',  # Bas = axe vertical
    'LEFT': 'H',  # Gauche = axe horizontal
    'RIGHT': 'H'  # Droite = axe horizontal
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

        # Vérification des limites du parking
        for point in car.points:
            new_x = point.x + move[0]
            new_y = point.y + move[1]
            if not (0 <= new_x < self.cols and 0 <= new_y < self.rows):
                print("Mouvement hors des limites !")
                return

            # Vérifiez les collisions avec d'autres voitures
            if self.find_car(new_x, new_y) and self.find_car(new_x, new_y) != car:
                print("Collision détectée !")
                return

        # Mise à jour des positions si aucune collision
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

    def draw_parking(self):
        # Fond gris pour le parking
        arcade.draw_rectangle_filled(
            TILE_SIZE * self.env.parking.cols / 2 + PARKING_OFFSET,
            TILE_SIZE * self.env.parking.rows / 2 + PARKING_OFFSET,
            TILE_SIZE * self.env.parking.cols,
            TILE_SIZE * self.env.parking.rows,
            arcade.color.GRAY,
        )
        # Lignes de la grille
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
        # Indication de la sortie
        arcade.draw_rectangle_filled(x + TILE_SIZE / 2, y, TILE_SIZE, TILE_SIZE, arcade.color.LIGHT_RED_OCHRE)
        arcade.draw_text("EXIT", x + TILE_SIZE / 2 - 20, y - 10, arcade.color.BLACK, 16, bold=True)

    def on_mouse_release(self, x, y, button, modifiers):
        self.input = (x, y)

    def on_key_press(self, key, modifiers):
        print("Key pressed", key)
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
            width=300  # Ajoutez une largeur pour que le texte soit centré
        )
        arcade.pause(3)  # Pause de 3 secondes avant de fermer
        arcade.close_window()


if __name__ == "__main__":
    env = Environment("levels/level2.json")
    env.setup()
    env.to_string()
    window = RushHourGame(env)
    arcade.run()
