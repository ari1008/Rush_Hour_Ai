# RushHour

Un jeu de puzzle où vous devez sortir la voiture rouge d'un embouteillage.

## Installation

```bash
# Clonez le dépôt
git clone https://github.com/votre-repo/rushHour.git

# Accédez au dossier
cd rushHour
```

## Utilisation

### Mode Manuel
Pour jouer manuellement :
```bash
python main.py level1.json --mode manual
```

### Mode Automatique (IA)
Pour laisser l'IA jouer avec une Q-table existante :
```bash
python main.py level1.json --mode auto --qtable_load qtables/level1.pickle
```

## Structure du Projet
```
rushHour/
├── levels/          # Fichiers de niveaux (.json)
├── qtables/         # Q-tables pré-entrainées (.pickle)
└── main.py         # Programme principal
```

## Arguments Disponibles

| Argument | Description | Valeur par défaut |
|----------|-------------|-------------------|
| `level` | Chemin du fichier niveau (.json) | Requis            |
| `--mode` | Mode de jeu (`manual`/`auto`) | `manual`          |
| `--qtable_load` | Charger une Q-table existante | rien              |
| `--qtable_save` | Sauvegarder la Q-table | `qtable.pickle`   |
| `--learning_rate` | Taux d'apprentissage | 0.2               |
| `--discount` | Facteur gamma | 0.99              |
| `--epsilon` | Epsilon initial | 0.5               |
| `--epsilon_decay` | Décroissance d'epsilon | 0.9999            |
| `--epsilon_min` | Epsilon minimum | 0.05              |
| `--max_steps` | Pas maximum par épisode | 1000              |
| `--reward` | Récompense de victoire | 1000              |
| `--step_penalty` | Pénalité par mouvement | -1                |
| `--speed` | Vitesse de jeu (10=lent, 50=moyen, 100=rapide) | 50                |

## Exemples

### Entraîner une nouvelle Q-table
```bash
python main.py levels/level1.json --mode auto --qtable_save nouvelle_qtable.pickle
```

### Utiliser une Q-table existante
```bash
python main.py levels/level2.json --mode auto --qtable_load qtables/level2.pickle
```

### Mode manuel avec vitesse personnalisée
```bash
python main.py levels/level3.json --mode manual --speed 75
```