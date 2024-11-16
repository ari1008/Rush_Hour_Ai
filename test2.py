import arcade
import json
import random

SCREEN_WIDTH = 1024
SCREEN_HEIGHT = 768
SCREEN_TITLE = "Rush Hour Game"

# Taille de la grille
GRID_SIZE = 8  # 8x8
CELL_SIZE = 80
GRID_OFFSET_X = (SCREEN_WIDTH - GRID_SIZE * CELL_SIZE) // 2
GRID_OFFSET_Y = (SCREEN_HEIGHT - GRID_SIZE * CELL_SIZE) // 2

class Car(arcade.Sprite):
    def __init__(self, width, height, color, is_main=False):
        # Créer une forme de voiture personnalisée
        super().__init__()
        self.width = width * CELL_SIZE
        self.height = height * CELL_SIZE
        self.color = color
        self.is_main = is_main
        
        # Texture pour la voiture
        texture = arcade.Texture.create_filled("car", (self.width, self.height), color)
        self.textures = [texture]
        self.set_texture(0)

class RushHourGame(arcade.Window):
    def __init__(self):
        super().__init__(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE)
        arcade.set_background_color(arcade.color.DARK_SLATE_GRAY)

        self.car_list = None
        self.main_car = None
        self.grid = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        self._config = None  # Chargera les données depuis le fichier
        self.load_config()

    def load_config(self):
        """Charger la configuration depuis le fichier JSON"""
        with open('config.json', 'r') as f:
            self._config = json.load(f)
        
    def setup(self):
        """Configurer le jeu"""
        self.car_list = arcade.SpriteList()
        
        # Création de la voiture principale (bleue)
        main_car_config = self._config["main_car"]
        self.main_car = Car(
            main_car_config["width"],
            main_car_config["height"],
            arcade.color.BLUE,
            is_main=True
        )
        self.main_car.center_x = GRID_OFFSET_X + (main_car_config["x"] + main_car_config["width"]/2) * CELL_SIZE
        self.main_car.center_y = GRID_OFFSET_Y + (main_car_config["y"] + main_car_config["height"]/2) * CELL_SIZE
        self.car_list.append(self.main_car)
        
        # Création des autres voitures
        for car in self._config["cars"]:
            new_car = Car(
                car["width"],
                car["height"],
                getattr(arcade.color, car["color"].upper())
            )
            new_car.center_x = GRID_OFFSET_X + (car["x"] + car["width"]/2) * CELL_SIZE
            new_car.center_y = GRID_OFFSET_Y + (car["y"] + car["height"]/2) * CELL_SIZE
            self.car_list.append(new_car)
        
    def draw_parking_lot(self):
        """Dessiner le parking avec des détails"""
        # Fond du parking
        arcade.draw_rectangle_filled(
            SCREEN_WIDTH // 2,
            SCREEN_HEIGHT // 2,
            GRID_SIZE * CELL_SIZE + 40,
            GRID_SIZE * CELL_SIZE + 40,
            arcade.color.GRAY
        )
        
        # Lignes de parking
        for row in range(GRID_SIZE + 1):
            y = GRID_OFFSET_Y + row * CELL_SIZE
            arcade.draw_line(
                GRID_OFFSET_X,
                y,
                GRID_OFFSET_X + GRID_SIZE * CELL_SIZE,
                y,
                arcade.color.WHITE,
                2
            )
            
        for col in range(GRID_SIZE + 1):
            x = GRID_OFFSET_X + col * CELL_SIZE
            arcade.draw_line(
                x,
                GRID_OFFSET_Y,
                x,
                GRID_OFFSET_Y + GRID_SIZE * CELL_SIZE,
                arcade.color.WHITE,
                2
            )
            
        # Dessiner l'herbe autour du parking
        self.draw_grass()
        
        # Dessiner la sortie
        exit_x = GRID_OFFSET_X + self._config["exit"]["x"] * CELL_SIZE
        exit_y = GRID_OFFSET_Y + self._config["exit"]["y"] * CELL_SIZE
        
        # Flèche de sortie
        arcade.draw_rectangle_filled(
            exit_x + CELL_SIZE/2,
            exit_y,
            CELL_SIZE,
            CELL_SIZE/2,
            arcade.color.GREEN
        )
        arcade.draw_text(
            "EXIT",
            exit_x + CELL_SIZE/2,
            exit_y,
            arcade.color.WHITE,
            14,
            anchor_x="center"
        )
        
    def draw_grass(self):
        """Dessiner l'herbe et la décoration autour du parking"""
        for i in range(50):  
             x = random.randrange(0, SCREEN_WIDTH) 
             y = random.randrange(0, SCREEN_HEIGHT)
        
             if (x < GRID_OFFSET_X - 20 or x > GRID_OFFSET_X + GRID_SIZE * CELL_SIZE + 20 or
                 y < GRID_OFFSET_Y - 20 or y > GRID_OFFSET_Y + GRID_SIZE * CELL_SIZE + 20):
                 arcade.draw_circle_filled(x, y, 3, arcade.color.DARK_GREEN)
        
    def on_draw(self):
        """Dessiner l'écran de jeu"""
        self.clear()
        
        # Dessiner le parking
        self.draw_parking_lot()
        
        # Dessiner les voitures
        self.car_list.draw()
        
    def on_key_press(self, key, modifiers):
        """Gérer les événements clavier"""
        dx = 0
        dy = 0
        
        if key == arcade.key.LEFT:
            dx = -CELL_SIZE
        elif key == arcade.key.RIGHT:
            dx = CELL_SIZE
        elif key == arcade.key.UP:
            dy = CELL_SIZE
        elif key == arcade.key.DOWN:
            dy = -CELL_SIZE
            
        # Vérifier si le mouvement est valide
        new_x = self.main_car.center_x + dx
        new_y = self.main_car.center_y + dy
        
        # Vérifier les limites de la grille et les collisions
        if self.is_valid_move(new_x, new_y):
            self.main_car.center_x = new_x
            self.main_car.center_y = new_y
            
    def is_valid_move(self, new_x, new_y):
        """Vérifier si le mouvement est valide"""
        if not (GRID_OFFSET_X <= new_x <= GRID_OFFSET_X + (GRID_SIZE-1) * CELL_SIZE and
                GRID_OFFSET_Y <= new_y <= GRID_OFFSET_Y + (GRID_SIZE-1) * CELL_SIZE):
            return False
            
        main_car_rect = self.main_car.get_adjusted_hit_box()
        
        for car in self.car_list:
            if car != self.main_car:
                if arcade.check_for_collision(self.main_car, car):
                    return False
                    
        return True

def main():
    window = RushHourGame()
    window.setup()
    arcade.run()

if __name__ == "__main__":
    main()