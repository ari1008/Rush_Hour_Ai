import json
import os
import arcade

SPRITE_SIZE = 64

# Actions
MOVES = {
    'W': (0, -1),  # Up (diminue y)
    'S': (0, 1),   # Down (augmente y)
    'A': (-1, 0),  # Left (diminue x)
    'D': (1, 0)    # Right (augmente x)
}

MOVES_NUMBER = {
    65362: 'W',
    65364: 'S',
    65361: 'A',
    65363: 'D'
}



class Car:
    def __init__(self, car_data):
        self.x = car_data["x"]
        self.y = car_data["y"]
        self.direction = car_data["direction"]
        self.color = car_data["color"]
        self.x1 = car_data["x1"]
        self.y1 = car_data["y1"]
        self.letter = car_data["direction"]

    def move(self, action, parking_lot):
        dx, dy = MOVES.get(action, (0, 0))
        if self.direction == 'V':
            new_y = self.y + dy
            new_y1 = self.y1 + dy
            if 0 <= new_y < parking_lot.height and 0 <= new_y1 < parking_lot.height:
                # Vérifier les collisions
                blocking_car = any(
                    parking_lot.find_car(self.x, y)
                    for y in range(min(new_y, new_y1), max(new_y, new_y1) + 1)
                    if parking_lot.find_car(self.x, y) is not self
                )
                if not blocking_car:
                    self.y = new_y
                    self.y1 = new_y1

        elif self.direction == 'H':
            new_x = self.x + dx
            new_x1 = self.x1 + dx
            # Vérifier les limites du parking
            if 0 <= new_x < parking_lot.width and 0 <= new_x1 < parking_lot.width:
                # Vérifier les collisions
                blocking_car = any(
                    parking_lot.find_car(x, self.y)
                    for x in range(min(new_x, new_x1), max(new_x, new_x1) + 1)
                    if parking_lot.find_car(x, self.y) is not self
                )
                if not blocking_car:
                    self.x = new_x
                    self.x1 = new_x1


class ParkingLot:
    def __init__(self, layout, cars_data):
        self.height = int(layout["height"])
        self.width = int(layout["width"])
        self.finish = (int(layout["finish"]["x"]), int(layout["finish"]["y"]))
        self.cars = [Car(car) for car in cars_data]

    def find_car(self, x, y):
        return next((car for car in self.cars if car.x <= x <= car.x1 and car.y <= y <= car.y1), None)

    def update(self, action, car_input: Car):
        car = next((c for c in self.cars if c.x == car_input.x and c.y == car_input.y), None)
        if car:
            car.move(action, self)


class Environment:
    def __init__(self, file_name):
        with open(file_name, 'r') as f:
            config = json.load(f)
        self.parking_lot = ParkingLot(config['map'], config['cars'])





class MazeWindow(arcade.Window):
    def __init__(self, env):
        super().__init__(SPRITE_SIZE * env.parking_lot.width, SPRITE_SIZE * env.parking_lot.height, "ESGI Maze")
        self.env = env
        self.walls = arcade.SpriteList()
        self.letters = arcade.SpriteList()
        self.input = None
        self.goal = arcade.Sprite(':resources:images/tiles/signExit.png', 0.5)
        arcade.set_background_color(arcade.color.AMAZON)

    def setup(self):
        self.walls.clear()

        for i in range(self.env.parking_lot.height):
            for j in range(self.env.parking_lot.width):
                car = self.env.parking_lot.find_car(j, i)
                if car:
                    file = self.find_file(car.color)
                    if file:
                        sprite = self.create_sprite(f"resources/{file}", (j, i))
                        self.walls.append(sprite)
                        y, x = j  * SPRITE_SIZE, (self.env.parking_lot.height - i ) * SPRITE_SIZE

        finish_x, finish_y = (self.env.parking_lot.finish[1] + 0.5) * SPRITE_SIZE, (
                    self.env.parking_lot.height - self.env.parking_lot.finish[0] - 0.5) * SPRITE_SIZE
        self.goal.center_x, self.goal.center_y = finish_x, finish_y

    def find_file(self, name):
        for root, dirs, files in os.walk('resources'):
            for file in files:
                if name in file:
                    return file
        return None

    def create_sprite(self, resource, state):
        sprite = arcade.Sprite(resource, 0.5)
        sprite.center_x = (state[1] + 0.5) * SPRITE_SIZE
        sprite.center_y = (self.env.parking_lot.height - state[0] - 0.5) * SPRITE_SIZE
        return sprite

    def update_sprites(self):
        # Met à jour les positions des sprites dans self.letters et self.walls
        for letter_sprite, car in zip(self.letters, self.env.parking_lot.cars):
            letter_sprite.center_x = (car.x + 0.5) * SPRITE_SIZE
            letter_sprite.center_y = (self.env.parking_lot.height - car.y - 0.5) * SPRITE_SIZE

        for wall_sprite, car in zip(self.walls, self.env.parking_lot.cars):
            wall_sprite.center_x = (car.x + 0.5) * SPRITE_SIZE
            wall_sprite.center_y = (self.env.parking_lot.height - car.y - 0.5) * SPRITE_SIZE

    def on_draw(self):
        arcade.start_render()
        self.walls.draw()
        self.goal.draw()

    def on_key_press(self, key, modifiers):
        if key in MOVES_NUMBER and self.input is not None:
            x, y = self.input
            # Conversion des coordonnées de la souris en coordonnées de la grille
            col = x // SPRITE_SIZE
            row = (self.height - y) // SPRITE_SIZE
            car = self.env.parking_lot.find_car(col, row)
            print(f"Car: {car}")
            if car:
                action = MOVES_NUMBER[key]
                self.env.parking_lot.update(action, car)
                self.update_sprites()
                print(f"Action: {action}")
                self.on_draw()

    def on_mouse_release(self, x, y, button, modifiers):
        self.input = (x, y)


if __name__ == "__main__":
    env = Environment("app-config.json")
    window = MazeWindow(env)
    window.setup()
    arcade.run()
