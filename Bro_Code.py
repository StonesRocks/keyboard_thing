import numpy as np
import networkx as nx
import json

json_file = 'keyboard-layout_1.json'
ken_array = np.empty

with open(json_file, encoding="utf8") as f:
    json_dict = json.load(f)

    adjy = 0
    skipy = 0

    ken_list = []

    for i in json_dict:
        width = 0
        skipx = 0
        for j in i:
            if isinstance(j, str):
                x = skipx+(width/2)
                y = float(adjy + skipy)
                ken_list.append([x, y])

            elif isinstance(j, dict):
                for key in j:
                    if key == 'x':
                        # print("add ", j['x'], " horizontal space")
                        skipx += float(j['x'])
                    elif key == 'y':
                        # print("add ", j['y'], " vertical space")
                        skipy += float(j['y'])
                    elif key == 'w':
                        # print("add key width by ", j['w'])
                        width = float(j['w']) - 1
                    elif key == 'h':
                        "add key height(downwards) by , j['h'])"
                        # adjy += float(j['h']) / 2
                    elif key == 'x2' or key == 'y2' or key == 'w2' or key == 'h2':
                        "uh oh"
                    elif key == 'a' or key == 'f' or key == 'f2' or key == 'p' or key == 's':
                        "something about font"
        skipy += 1
    ken_array = np.array(list(ken_list))

matrix_size = len(ken_array)
max = 0
may = 0
if np.sqrt(matrix_size) == int(np.sqrt(matrix_size)):
    max = np.sqrt(matrix_size)
    may = max
else:
    max = int(np.sqrt(matrix_size)) + 1
    may = int(np.sqrt(matrix_size))

print(ken_array)
