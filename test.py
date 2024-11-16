import arcade

# Constantes
SCREEN_WIDTH = 700
SCREEN_HEIGHT = 700
SCREEN_TITLE = "Rush Hour"
TILE_SIZE = 80  
PARKING_ROWS = 6
PARKING_COLS = 6
PARKING_OFFSET = 100  
EXIT_ROW = 2  

class RushHourGame(arcade.Window):
    def __init__(self):
        super().__init__(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE)
        arcade.set_background_color(arcade.color.DARK_SPRING_GREEN)

        self.parking = []  # Liste des voitures
        self.exit_position = (PARKING_COLS, EXIT_ROW)  # Position de la sortie 

    def setup(self):
        """Initialise les éléments du jeu."""
        self.parking = [
            (1, 2, arcade.color.RED),  # Voiture principale
            (0, 0, arcade.color.BLUE),
            (3, 5, arcade.color.GREEN),
            (4, 1, arcade.color.YELLOW),
        ]

    def on_draw(self):
        """Affiche les éléments du jeu."""
        arcade.start_render()

        self.draw_grass()

        self.draw_parking()

        self.draw_cars()

        self.draw_exit()

    def draw_grass(self):
        grass_thickness = 20
        arcade.draw_rectangle_filled(
            SCREEN_WIDTH / 2,
            SCREEN_HEIGHT / 2,
            SCREEN_WIDTH,
            SCREEN_HEIGHT,
            arcade.color.DARK_SPRING_GREEN,
        )

    def draw_parking(self):
        # Fond gris pour le parking
        arcade.draw_rectangle_filled(
            SCREEN_WIDTH / 2,
            SCREEN_HEIGHT / 2,
            TILE_SIZE * PARKING_COLS,
            TILE_SIZE * PARKING_ROWS,
            arcade.color.GRAY,
        )

        # Lignes de la grille
        for row in range(PARKING_ROWS + 1):
            y = row * TILE_SIZE + PARKING_OFFSET
            arcade.draw_line(
                PARKING_OFFSET,
                y,
                PARKING_OFFSET + PARKING_COLS * TILE_SIZE,
                y,
                arcade.color.BLACK,
                2,
            )
        for col in range(PARKING_COLS + 1):
            x = col * TILE_SIZE + PARKING_OFFSET
            arcade.draw_line(
                x,
                PARKING_OFFSET,
                x,
                PARKING_OFFSET + PARKING_ROWS * TILE_SIZE,
                arcade.color.BLACK,
                2,
            )

    def draw_cars(self):
        for col, row, color in self.parking:
            x = col * TILE_SIZE + TILE_SIZE / 2 + PARKING_OFFSET
            y = row * TILE_SIZE + TILE_SIZE / 2 + PARKING_OFFSET
            arcade.draw_rectangle_filled(x, y, TILE_SIZE - 10, TILE_SIZE - 10, color)

    def draw_exit(self):
        x = self.exit_position[0] * TILE_SIZE + PARKING_OFFSET
        y = self.exit_position[1] * TILE_SIZE + TILE_SIZE / 2 + PARKING_OFFSET
        # Indication de la sortie
        arcade.draw_rectangle_filled(x + TILE_SIZE / 2, y, TILE_SIZE, TILE_SIZE, arcade.color.LIGHT_RED_OCHRE)
        arcade.draw_text("EXIT", x + TILE_SIZE / 2 - 20, y - 10, arcade.color.BLACK, 16, bold=True)

def main():
    game = RushHourGame()
    game.setup()
    arcade.run()

if __name__ == "__main__":
    main()
