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
    #puts every kex and key into a numpy array where each index represents a key.
    ken_array = np.array(list(ken_list))

#looks up the smallest matrix available, if matrix is square then columns and row will match, otherwise add extra column
ken_size = len(ken_array)
man_size = 0 #value given after
max = 0
may = 0
if np.sqrt(ken_size) == int(np.sqrt(ken_size)):
    max = np.sqrt(ken_size)
    may = max
else:
    max = int(np.sqrt(ken_size)) + 1
    may = int(np.sqrt(ken_size))
man_size = max * may
print(man_size)
#create an array with all rows and columns.
man_array = np.empty
rows = []
k = 0
for i in range(may):
    onerow = []
    onerow.clear()
    for j in range(0,max):
        onerow.append(k)
        k += 1
    rows.append(onerow)
man_array = np.array(list(rows))

#should i create a class for the individual in population?
#the class would include the list of ken2man and fitness.

#ken2man or man2ken?
#we need man2ken to be able to calculat the length/fitness
#we need ken2man to be able know which man exist in keyboard.
    #why cant we just have a list of the existing man and then use each index as the ken?
    #we look up which index the man have and take all the index to calculate kex and key?

class Keyboard:
    def __init__(self):
        self.fitness = 0
        self.man = np.random.choice(man_size, size=ken_size, replace=False)




