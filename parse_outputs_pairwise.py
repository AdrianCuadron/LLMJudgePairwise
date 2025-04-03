import json
import pandas as pd
import argparse
from collections import defaultdict
import trueskill

def process_json_to_table(input_file, output_file):
    # Cargar el archivo JSON
    with open(input_file, "r", encoding="utf-8") as file:
        data = json.load(file)

    suma_winners = defaultdict(int)

    # create all players
    players = {
        "GPT4": trueskill.Rating(),
        "OpenBioLLM": trueskill.Rating(),
        "Llama3": trueskill.Rating(),
        "Gold": trueskill.Rating(),
        "Cheater": trueskill.Rating(),
        "RAG": trueskill.Rating(),
        "empty": trueskill.Rating(),
        "Random": trueskill.Rating(),
    }

    comparaciones = []

    for inst in data:
        comb = inst["combination"]
        #print(comb)

        model_eval = inst["model_evaluation"]
        #print(model_eval)
        winner = model_eval.split("Chosen argument: ")[-1].strip()
        #print(winner)

        if winner=="A":
            suma_winners[comb[0]] += 1
            winner_model = comb[0]
            loser_model = comb[1]

        else:
            suma_winners[comb[1]] += 1
            winner_model = comb[1]
            loser_model = comb[0]
        

        comparaciones.append((winner_model.split('_')[-1], loser_model.split('_')[-1]))


    # Aplicar TrueSkill a cada comparación
    for ganador, perdedor in comparaciones:
        players[ganador], players[perdedor] = trueskill.rate_1vs1(players[ganador], players[perdedor])
    
    # Ordenar por la media del ranking
    ranking = sorted(players.items(), key=lambda x: -x[1].mu)


    sorted_args = sorted(suma_winners.items(), key=lambda x: x[1], reverse=True)
    print(sorted_args)

    
    print("\nRanking de las respuestas:")
    for i, (modelo, cont) in enumerate(sorted_args, start=1):
        print(f"{i}. {modelo}: {cont}")

    # Guardar el ranking en un archivo de salida
    output_filename = f"{output_file}_ranking.txt"
    with open(output_filename, "w") as file:
        file.write("Ranking base:\n")
        for i, (modelo, cont) in enumerate(sorted_args, start=1):
            file.write(f"{i}. {modelo.split('_')[-1]}: {cont}\n")
        
        # Mostrar ranking final
        file.write("\nRanking TrueSkill:\n")
        for i, (name, rating) in enumerate(ranking, 1):
            file.write(f"{i}. {name}: {rating.mu:.2f} ± {rating.sigma:.2f}\n")
    

    print(f"\nEl ranking ha sido guardado en {output_filename}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Procesar un archivo JSON y generar una tabla CSV con ratings de modelos.")
    parser.add_argument("input_file", help="Ruta al archivo JSON de entrada")
    parser.add_argument("output_file", help="Ruta al archivo CSV de salida")

    args = parser.parse_args()

    process_json_to_table(args.input_file, args.output_file)
