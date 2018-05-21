
import retro
for game in retro.list_games():
    if "Sonic" in game:
        print(game, retro.list_states(game))