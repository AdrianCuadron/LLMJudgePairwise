import argparse
import os
import sys
import json
import re
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

import random
import getpass


random.seed(42)

def obtener_feedback(texto):
    clave = "AI: Feedback:::"
    # Buscar la posicion de la clave
    posicion = texto.find(clave)
    if posicion != -1:
        # Extraer todo lo que viene despues de la clave
        return texto[posicion + len(clave):].strip()
    else:
        return "Error: No se ha encontrado el feedback. Texto: "+texto


def obtener_pregunta(texto):
    patron_question = r"question:\s*(.+)"
    match = re.search(patron_question, texto)
    return match.group(1) if match else None


def obtener_nli(texto):
    patron_question = r"statement:\s*(.+)"
    match = re.search(patron_question, texto)
    return match.group(1) if match else None


def obtener_respuesta(texto):
    patron_answer = r"argument:\s*(.+)"
    match = re.search(patron_answer, texto)
    return match.group(1) if match else None


def main():

    data_path = "" # add the data path here
    prompt_path = ""
    output_path = ""
    empty = False

    #####################   Parsear argumentos y hacer comprobaciones    ##################

    
    parser = argparse.ArgumentParser(description="Script para utilizar LLM-as-judge utilizando un prompt de plantilla")
    parser.add_argument("template", help="Plantilla para el prompt")
    parser.add_argument("task", help="QA/MISS/NLI", choices=["QA", "MISS", "NLI"])
    parser.add_argument("model", help="llama/gemma/mistral/aloe", choices=["llama", "gemma", "mistral", "aloe"])
    parser.add_argument("--empty", action='store_true', help="Opcion para hacer los prompts con respuestas vacias (control case)")
    parser.add_argument("--dev", action='store_true', help="Opcion para utilizar el dev para probar el prompt ")
    parser.add_argument("--batch", help="batch size para la inferencia", type=int)
    args = parser.parse_args()    

    if not os.getenv("HUGGINGFACEHUB_API_TOKEN"):
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = getpass.getpass("Enter your token: ")

    split = "test"
    if args.dev:
        split = "dev"

    # Comprobar si el archivo existe
    if not os.path.isfile(args.template):
        print(f"Error: El archivo {args.template} no existe.")
        sys.exit(1)
    else:
        prompt_path = args.template
        with open(prompt_path, 'r') as archivo:
            content = archivo.read()
            contentJson = json.loads(content)
            system_template = contentJson["system"]
            user_template = contentJson["user"]
    
    if args.task=="QA":
        data_path += f"QA/{split}_QA_all_arguments.jsonl"
    elif args.task=="NLI":
        data_path += f"NLI/{split}_NLI_all_arguments.jsonl"
    else:
        data_path += f"MISS/{split}_missinformation_all_arguments.jsonl"

    if not os.path.isfile(data_path):
        print(f"Error: El archivo '{data_path}' no existe.")
        sys.exit(1)

    if args.empty:
        empty = True
    
    #######################     instanciar el modelo   #######################
    if args.batch is not None:
        batch_size = args.batch
    else: 
        batch_size = 24
    # batch_size = 35
    # batch_size = 10
    # batch_size = 20
    

    if args.model=="llama":
        model_id = "meta-llama/Llama-3.2-3B-Instruct"
    elif args.model=="gemma":
        model_id = "google/gemma-2-9b-it"
    elif args.model == "mistral":
        model_id = "mistralai/Mistral-7B-Instruct-v0.3"
    elif args.model == "aloe":
        model_id = "HPAI-BSC/Llama3.1-Aloe-Beta-8B"
    else:
        exit(1)


    # model_id = "meta-llama/Llama-3.1-8B-Instruct"
    # model_id = "meta-llama/Llama-3.3-70B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side='left')
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto" )
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", temperature=0, do_sample=True)

    pipe = pipeline(
        "text-generation", 
        model=model, 
        tokenizer=tokenizer, 
        max_new_tokens=300, 
        do_sample=True,
        model_kwargs={
            'device_map': 'auto',
            'batch_size':batch_size,
            "temperature": 0 
            }, 
        batch_size=batch_size 
    )
    # hf = HuggingFacePipeline(pipeline=pipe, batch_size=batch_size)
    hf = HuggingFacePipeline(pipeline=pipe, batch_size=batch_size, model_kwargs={"temperature": 0})


    ##############    Generar los prompts    ###################
    
    #i = 0
    # Abrir el archivo en modo lectura
    with open(data_path, 'r') as dataset:
        prompts = []
        combination_keys = []
        for instancia_ in dataset:
            instancia = json.loads(instancia_.strip())

            # hay que tratar de forma diferente los jsons de las diferentes tareas
            
            if args.task=="QA":    
                question = instancia["question"]
                options = instancia["Options"]
                correct_option = instancia["correct_option"]
                quest_opt_correct = question + options + "Correct option: "+ str(correct_option)

                # obtener una lista con todos los argumentos 
                arguments = {key: value for key, value in instancia.items() if key.startswith("Argument")}

                if empty:
                    # añadir argumento vacio 
                    arguments["empty"] = ""

                items = list(arguments.items())
                random.shuffle(items)
                arguments = dict(items)

                print(arguments)
                

                # crear los promps para todos los pares posibles 
                argument_pairs = [(item1, item2) 
                            for i, item1 in enumerate(arguments.items()) 
                            for item2 in list(arguments.items())[i+1:]]
                
                for argA, argB in argument_pairs:
                    # con cada par de argumentos crear el prompt y añadirlo a la lista de prompts
                    keyA, valueA = argA
                    keyB, valueB = argB
                    combination_keys.append([keyA, keyB])
            
                    # Plantilla del prompt que se va a utilizar
                    prompt_template = ChatPromptTemplate.from_messages([
                        ("system", system_template),
                        ("user",   user_template),
                        ("assistant", "{assistant_response}")
                    ]
                    )

                    # completar plantilla del prompt con los datos 
                    prompt = prompt_template.invoke(
                        {"question": quest_opt_correct,
                        "argumentA": valueA, # ARGUMENTO A
                        "argumentB": valueB, # ARGUMENTO B
                        "assistant_response": "Feedback:::\nEvaluation: "
                        })
                    
                    prompts.append(prompt) # guardar el prompt en la lista de prompts
                
            if args.task=="MISS":
                map = {
                    0: "Supported",
                    1: "Not evidence",
                    2: "Refuted"
                }
                question = instancia["Question"]
                label = instancia["Label"]
                if int(label)==1:
                    # saltar instancias con label de not evidence
                    continue 
                quest_label = question + ' Label is: ' + map[int(label)]

                # obtener una lista con todos los argumentos 
                arguments = {key: value for key, value in instancia.items() if key.startswith("Argumentation")}

                if empty:
                    # añadir argumento vacio 
                    arguments["empty"] = ""

                items = list(arguments.items())
                random.shuffle(items)
                arguments = dict(items)

                # crear los promps para todos los pares posibles 
                argument_pairs = [(item1, item2) 
                            for i, item1 in enumerate(arguments.items()) 
                            for item2 in list(arguments.items())[i+1:]]
                
                for argA, argB in argument_pairs:
                    # con cada par de argumentos crear el prompt y añadirlo a la lista de prompts
                    keyA, valueA = argA
                    keyB, valueB = argB
                    combination_keys.append([keyA, keyB])
            
                    # Plantilla del prompt que se va a utilizar
                    prompt_template = ChatPromptTemplate.from_messages([
                        ("system", system_template),
                        ("user",   user_template),
                        ("assistant", "{assistant_response}")
                    ]
                    )

                    # completar plantilla del prompt con los datos 
                    prompt = prompt_template.invoke(
                        {"question": quest_label,
                        "argumentA": valueA, # ARGUMENTO A
                        "argumentB": valueB, # ARGUMENTO B
                        "assistant_response": "Feedback:::\nEvaluation: "
                        })
                    
                    prompts.append(prompt) # guardar el prompt en la lista de prompts


            if args.task=="NLI": 
                statement = instancia["Statement"]
                label = instancia["Label"]
                map = {
                    0: "Entailment",
                    1: "Contradiction"
                }
                state_label = statement + '. Label: '+ map[int(label)]

                # obtener una lista con todos los argumentos 
                arguments = {key: value for key, value in instancia.items() if key.startswith("Evidence")}

                if empty:
                    # añadir argumento vacio 
                    arguments["empty"] = ""

                items = list(arguments.items())
                random.shuffle(items)
                arguments = dict(items)

                # crear los promps para todos los pares posibles 
                argument_pairs = [(item1, item2) 
                            for i, item1 in enumerate(arguments.items()) 
                            for item2 in list(arguments.items())[i+1:]]
                
                for argA, argB in argument_pairs:
                    # con cada par de argumentos crear el prompt y añadirlo a la lista de prompts
                    keyA, valueA = argA
                    keyB, valueB = argB
                    combination_keys.append([keyA, keyB])
            
                    # Plantilla del prompt que se va a utilizar
                    prompt_template = ChatPromptTemplate.from_messages([
                        ("system", system_template),
                        ("user",   user_template),
                        ("assistant", "{assistant_response}")
                    ]
                    )

                    # completar plantilla del prompt con los datos 
                    prompt = prompt_template.invoke(
                        {"question": state_label,
                        "argumentA": valueA, # ARGUMENTO A
                        "argumentB": valueB, # ARGUMENTO B
                        "assistant_response": "Feedback:::\nEvaluation: "
                        })
                    
                    prompts.append(prompt) # guardar el prompt en la lista de prompts


    #######################   Pasar los prompts al modelo y guardar los resultados #################

    # sacar un ejemplo por terminal para probar
    print("prompt")
    print(prompts[0])
    response1 = hf.invoke(prompts[0]) 
    print("response")
    print(response1)
    print("combination")
    print(combination_keys[0])
    print("FEEDBACK")
    print(obtener_feedback(response1))
    print("output de nli")
    print(obtener_nli(response1))
    print("output del resto)")
    print(obtener_respuesta(response1))
    #input()
    #exit()

    # pasar todos los prompts a la vez al modelo
    responses = hf.batch(prompts) 

    # guardar los output en un fichero TODO cambiar como se guarda dependiendo de la tarea
    output = []
    i=0
    for response in responses:
        json_output = {}
        if args.task=="QA" or args.task == "MISS" :
            json_output["question"] = obtener_pregunta(response)
        else:
            json_output["description"] = obtener_nli(response)

        json_output["combination"]   = combination_keys[i]
        i+=1
        json_output["model_evaluation"] = obtener_feedback(response)
        output.append(json_output)
    
    with open('/gaueko1/users/asagasti036/'+model_id.replace('/', '_')+'_'+args.task+'_'+split+'_pairwise.json', "w", encoding="utf-8") as file:
        json.dump(output, file, ensure_ascii=False, indent=4)
        
        

if __name__ == "__main__":
    main()