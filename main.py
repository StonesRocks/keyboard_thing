import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import libpysal as ps
import json
from pathlib import Path
from numpy.random import default_rng as rng
"""
Objective: Model a keyboard circuit to determine the optimal circuit design
Special: use OOP to model the keyboard circuit
"""
# define keyboard class
class Keyboard:
    def __init__(self, pins, json_file):
        self.keyboard = nx.Graph()
        self.json_file = json_file
        self.length = float(0)
        self.fitness = float(1)
        self.relative_fitness = float(0)
        self.processed_fitness = float(0)
        self.node_dict = {}
        with open(json_file, encoding="utf8") as f:
            self.json_dict = json.load(f)
            #print(type(self.json_dict))
            #print(len(self.json_dict))
            node = 0
            adjy = 0
            skipy = 0
            my_list = []
            full_list = []
            for i in self.json_dict:
                #print("This is a new row!")
                width = 0
                skipx = 0
                for j in i:
                    #print(j)
                    if isinstance(j, str):
                        #print(j.rstrip(), "is a str")
                        self.keyboard.add_node('SW' + str(node))
                        self.keyboard.nodes['SW' + str(node)]['coord'] = (skipx+(width/2), float(adjy + skipy))
                        this_list = [node,
                                     self.keyboard.nodes['SW' + str(node)]['coord'][0],
                                     self.keyboard.nodes['SW' + str(node)]['coord'][1]]

                        self.node_dict["node" + str(node)] = {
                                'node' : this_list[0],
                                'nodex': this_list[1],
                                'nodey': this_list[2]
                        }
                        '''        
                        adding dict
                        self.node_dict['node0']['newkey'] = 6
                        using dict
                        print(self.node_dict['node0']['newkey'])
                        '''

                        #creating edges between nodes within the same row
                        if (node != 0 and
                                self.keyboard.nodes['SW' + str(node-1)]['coord'][1] ==
                                self.keyboard.nodes['SW' + str(node)]['coord'][1]):

                            theweight = np.abs(self.keyboard.nodes['SW' + str(node)]['coord'][0] -
                                               self.keyboard.nodes['SW' + str(node-1)]['coord'][0])
                            self.keyboard.add_edge('SW' + str(node-1), 'SW' + str(node), weight=theweight)

                        if width != 0:
                            skipx += width

                        skipx += 1
                        width = 0
                        #[node, node.x, node.y]
                        my_list.append(this_list)
                        node += 1

                    #checks modifier
                    elif isinstance(j, dict):
                        #print("This a dict with ", len(j), "items")
                        for key in j:
                            if key == 'x':
                                #print("add ", j['x'], " horizontal space")
                                skipx += float(j['x'])
                            elif key == 'y':
                                #print("add ", j['y'], " vertical space")
                                skipy += float(j['y'])
                            elif key == 'w':
                                #print("add key width by ", j['w'])
                                width = float(j['w'])-1
                            elif key == 'h':
                                "add key height(downwards) by , j['h'])"
                                #adjy += float(j['h']) / 2
                            elif key == 'x2' or key == 'y2' or key == 'w2' or key == 'h2':
                                "uh oh"
                            elif key == 'a' or key == 'f' or key == 'f2' or key == 'p' or key == 's':
                                "something about font"
                self.num_nodes = node
                #print(self.num_nodes)
                skipy += 1
                full_list.append(my_list.copy())
                #clearing list to reuse
                my_list.clear()
        #creating edges from one node to nodes in the row below.
        for i in range(0, len(full_list)-1):
            #print("this is ", full_list[i])
            for j in range(0, len(full_list[i])):
                #print("comparing this node: ", full_list[i][j])
                for k in range(0, len(full_list[i+1])):
                    #print("with ", full_list[i+1][k])
                    if abs(float(full_list[i][j][1])-float(full_list[i+1][k][1])) <= 1.5:
                        #print(full_list[i][j], full_list[i+1][k])
                        self.keyboard.add_edge('SW' + str(int(full_list[i][j][0])),
                                               'SW' + str(int(full_list[i+1][k][0])))
                        theweight = (
                                abs(self.keyboard.nodes['SW' + str(int(full_list[i][j][0]))]['coord'][0] -
                                    self.keyboard.nodes['SW' + str(int(full_list[i+1][k][0]))]['coord'][0]) +
                                abs(self.keyboard.nodes['SW' + str(int(full_list[i][j][0]))]['coord'][1] -
                                    self.keyboard.nodes['SW' + str(int(full_list[i+1][k][0]))]['coord'][1])
                        )
                        self.keyboard.add_edge('SW' + str(int(full_list[i][j][0])),
                                               'SW' + str(int(full_list[i+1][k][0])),
                                               weight=theweight)

        #creates matrix for pins
        if (pins / 2) % 2 == 0:
            pincols = pins / 2
            pinrows = pins / 2
        else:
            pincols = int(pins / 2 + 0.5)
            pinrows = int(pins / 2 - 0.5)

        #These are the matrix/switches
        self.pinrows = int(pinrows)
        self.pincols = int(pincols)
        self.matrix = int(pinrows * pincols)
        if self.matrix < node:
            print("Pins are too low, maximum matrix: ", self.matrix, " while keyboard requires atleast: ", node)

        #each list in this list represents connected switches either through column or row.
        self.switch_lines = []
        switch_num = 0
        for i in range(self.pinrows):
            line_list = []
            for j in range(self.pincols):
                line_list.append(switch_num)
                switch_num += 1
            self.switch_lines.append(line_list)
        for j in range(self.pincols):
            line_list = []
            for i in range(self.pinrows):
                line_list.append(self.switch_lines[i][j])
            self.switch_lines.append(line_list)






        #my_dict is used for drawing
        self.my_dict ={}
        for i in full_list:
            for j in i:
                self.my_dict['SW' + str(j[0])] = j[1],j[2]



        rand = np.random.choice(self.matrix, size=self.num_nodes, replace=False)
        #print(rand)
        self.list_of_nodes = []
        self.list_of_unused = []
        self.list_of_used = []
        #This creates a list of node, switch pairs. [node, switch]
        for i in range(self.num_nodes):
            pair_list = [i, rand[i]]
            self.list_of_used.append(rand[i])
            self.list_of_nodes.append(pair_list)
            self.node_dict['node' + str(i)]['switch'] = rand[i]
        self.list_of_nodes = sorted(self.list_of_nodes, key=lambda x: x[1])
        #print(self.list_of_nodes)
        #print(self.node_dict)
        #print(self.list_of_nodes)




        #This creates a list with all unused switches for later usage
        for i in range(self.matrix):
            if i not in self.list_of_used:
                self.list_of_unused.append(i)
        sorted(self.list_of_unused)
        #print("length of list of unused", len(self.list_of_unused)) answer: 5
        #print("list of used switches", self.list_of_used)
        #print("list of unused switches", self.list_of_unused)
        #print(self.list_of_nodes)

    def for_other_parent(self):
        return [self.list_of_nodes, self.list_of_unused]


    def rewireswitch(self):
        self.list_of_nodes.clear()
        for node in self.node_dict:
            node_switch = [self.node_dict[node]['node'], self.node_dict[node]['switch']]
            self.list_of_nodes.append(node_switch)
        self.list_of_nodes = sorted(self.list_of_nodes, key=lambda x: x[1])
        full_matrix= list(range(0, self.matrix))
        for node in self.list_of_nodes:
            try:
                full_matrix.pop(full_matrix.index(node[1]))

            except ValueError:
                pass
        self.list_of_unused = full_matrix
        if len(self.list_of_unused) > 5:
            print("list of unused", len(self.list_of_unused), self.list_of_nodes)

    def calculate_fitness(self):
        self.length = 0
        node_list_copy = self.list_of_nodes
        nodec, switchc = map(list, zip(*node_list_copy))
        #print(node_list_copy)
        #print(nodec, switchc)
        for line in self.switch_lines:
            node_line = []
            for switch in line:
                if switch in switchc:
                    node_line.append(nodec[switchc.index(switch)])
            line_list = []
            for node in node_line:
                node_coord = [
                    self.node_dict['node' + str(node)]['node'],
                    self.node_dict['node' + str(node)]['nodex'],
                    self.node_dict['node' + str(node)]['nodey']
                ]
                line_list.append(node_coord)
            line_list = sorted(line_list, key=lambda x: x[1])
            for node in range(len(line_list)-1):
                self.length += abs(line_list[node+1][1]-line_list[node][1]) + \
                               abs(line_list[node+1][2]-line_list[node][2])







# define population class
class Population:
    def __init__(self, count):
        # size of population
        self.count = count
        self.best = 100000
        self.top_keeb = []
        self.gen_fitness = float(0)
        # initialize keyboard population
        self.population = [Keyboard(21, 'keyboard-layout_1.json') for i in range(self.count)]
        #print(self.population)



    def copulate(self):
        sum_fitness = float(0)
        self.keeb_and_probability = []
        for i in range(self.count):
            self.keeb_and_probability.append(0)
        number = 0
        ranking = []
        for keeeb in self.population:
            keeeb.calculate_fitness()
            ranking.append([number, keeeb.length])
            if keeeb.length < self.best:
                print("new best")
                best_nodes_list = ["length", keeeb.length, keeeb.list_of_nodes]
                self.top_keeb.append(best_nodes_list)
                while len(self.top_keeb) > 3:
                    self.top_keeb.pop(0)
                    print(self.top_keeb[0])
                    print(self.top_keeb[1])
                    print(self.top_keeb[2])
                    self.best = self.top_keeb[2][1]
            number += 1
        ranking = sorted(ranking, key=lambda x: -x[1])

        if ranking[-1][1] < self.best:
            self.best = ranking[-1][1]
        #print("best:", self.best, "this is ranking", sorted(ranking, key=lambda x: x[1]))


        #linear rank probability
        k = 2/(self.count**2)
        sum_prob = 0
        for rank in ranking:
            self.keeb_and_probability[rank[0]] = (k*(ranking.index(rank)))
            sum_prob += (k*(ranking.index(rank)))

        probs = np.array(self.keeb_and_probability)
        probs /= probs.sum()


        #grab dict of highest fitness and replace lowest fitness in next generation.

        pairs = []
        for i in range(int(self.count/2+0.5)):
            #last condition is the probability "p=self.keeb_and_probability"
            pairs.append(np.random.choice(pop.population, size=2, replace=False, p=probs))
        #print(pairs)

        for this_pair in pairs:
            '''print("this is a pair ", this_pair)
            print("this is first parent ", this_pair[0].node_dict)
            print("this is second parent ", this_pair[1].node_dict)
            '''
            split = np.random.randint(0, pop.population[0].matrix)
            #print("split here ",split)
            parent1 = this_pair[0].for_other_parent()
            parent2 = this_pair[1].for_other_parent()
            #print(parent1[0][1])

            split_here1 = 0
            split_here2 = 0
            for pair in parent1[0]:
                if pair[1] < split:
                    split_here1 += 1
            for pair in parent2[0]:
                if pair[1] < split:
                    split_here2 += 1

            #the flip part is broken. needs fixing.
            #flip_kept_given = np.random.randint(2)
            flip_kept_given = 0
            if flip_kept_given == 0:
                parent1_kept = parent1[0][:split_here1]
                parent2_kept = parent2[0][:split_here2]
                parent1_given = parent1[0][split_here1:]
                parent2_given = parent2[0][split_here2:]
            else:
                parent1_kept = parent1[0][split_here1:]
                parent2_kept = parent2[0][split_here2:]
                parent1_given = parent1[0][:split_here1]
                parent2_given = parent2[0][:split_here2]
                #print("flipped")

            #print("parent 1 kept:", parent1_kept, "given", parent1_given)
            #print("parent 2 kept:", parent2_kept, "given", parent2_given)

            ''' pretty sure this section doesnt matter.
            #creates a list to match unused switches from parents where [parent1, parent2]
            parent1_removed = []
            parent2_removed = []
            for unused1 in parent1[1]:
                for unused2 in parent2[1]:
                    if unused1 == unused2:
                        parent1_removed.append(parent1[1].pop(parent1[1].index(unused1)))
                        parent2_removed.append(parent2[1].pop(parent2[1].index(unused1)))
                        #parent1[1].remove(unused1)
                        #parent2[1].remove(unused1)
            '''

            #checks for first step of merge conflict, if amount of nodes kept are different for the pairs
            if len(parent1_kept) != len(parent2_kept):
                while len(parent1_kept) > len(parent2_kept):
                    #print(len(parent2_kept), len(parent1_kept), parent2_given[0], parent2_given)
                    #print("this is parent2 given:", parent2_given)
                    replace_node = parent2_given.pop(0)
                    #print("this is replace_node", replace_node)
                    replace_node[1] = parent2[1].pop(0)
                    #print("this is replace_node after fix", replace_node)
                    #print("this is parent2_kept before", parent2_kept)
                    parent2_kept.append(replace_node)
                    #print("this is parent2_kept after", parent2_kept)
                while len(parent1_kept) < len(parent2_kept):
                    #print(len(parent2_kept), len(parent1_kept), parent1_given[0], parent1_given)
                    replace_node = parent1_given.pop(0)
                    replace_node[1] = parent1[1].pop(0)
                    parent1_kept.append(replace_node)

            #Create copies of nodes and reapply them to the others genes.
            parent1_given = sorted(parent1_given, key=lambda x: x[0])
            parent2_given = sorted(parent2_given, key=lambda x: x[0])
            #loop to extract the nodes from switches and then swapping them.
            parent1_node = []
            parent1_switch = []
            parent2_node = []
            parent2_switch = []
            for pairs in parent1_given:
                parent1_node.append(pairs[0])
                parent1_switch.append(pairs[1])
            for pairs in parent2_given:
                parent2_node.append(pairs[0])
                parent2_switch.append(pairs[1])

            parent1_given.clear()
            parent2_given.clear()
            for i in range(len(parent1_node)):
                pair = [parent1_node[i], parent2_switch[i]]
                parent1_kept.append(pair)
                pair = [parent2_node[i], parent1_switch[i]]
                parent2_kept.append(pair)
            this_pair[0].list_of_nodes = parent1_kept
            this_pair[1].list_of_nodes = parent2_kept

            for i in range(len(parent1_kept)):
                this_pair[0].node_dict['node' + str(parent1_kept[i][0])]['switch'] = parent1_kept[i][1]
                this_pair[1].node_dict['node' + str(parent2_kept[i][0])]['switch'] = parent2_kept[i][1]
            #print(this_pair[0].list_of_unused)
            this_pair[0].rewireswitch()
            this_pair[1].rewireswitch()
            #print(this_pair[0].list_of_unused)
            #print("this is first child ", this_pair[0].node_dict)
            #print("this is second child ", this_pair[1].node_dict)


        # RULES FOR GENERATING INDIVIDUALS
        # 1. grab copy of switches.
        # 2. grab copy of keyboard.
        # 3. start at first node and randomize one of the item in switches list, continue until every node has one switch.
        # 4. this is now a generated keeb.

        #RULES FOR CROSSOVER
        # 1. randomize where to split the switches, draw a line between first switch and n switch.
        # 2. decide which part will be kept and which will mutate(since kept/locked switches will have different locations on each keeb we'll mutate the other parents gene to fit it)
        # 3. mutate gene to fit the other keyboard.
        # 3.1. lock switches to nodes that will not mutate/will be kept, if switch x is used on parent1 but not parent2 then parent2 will have an equivalent switch y.
        # 3.2. grab all unlocked switches from parent1 and place them as close as possible to their node position on parent1 but on parent2, the node on parent2 might be occupied by a locked switch and therefore will look for a closer unlocked switch, this will be the mutation.
        # 3.3. repeat step 3.2 on parent1 with switches from parent2
        # 3.4. two new children has been generated.

        #RULES FOR FITNESS
        # 1. summarize all unique rows and columns, this pin score is more important than the distance score.
        # 2. sort used switches by row
        # 3. grab all switches with same row and calculate distance score of each row by shortest distance to connect all switches of same row.
        # 3.2. note that the distance is based on the nodes and not switch matrix, we also need an algorithm to find shortest path.
        # 4. do step 2 but with columns instead of rows and add them all together, the smaller the score the better.
        # 5. do some math where fewer pins used(fewer unique rows and columns) is better fitness as well as lower distance score, note that pin score weights more than distance score, this should avoid the algorithm to apply a unique row and column for each switch(giving it a 0 distance score) when amount of pins>=switches.




#keebo = Keyboard(21, 'keyboard-layout_1.json')
'''
all_the_keebs = []
for i in range(10):
    keeb = Keyboard(21, 'keyboard-layout_1.json')
    all_the_keebs.append(keeb)
'''
pop = Population(1000)
for i in range(1000000):
    pop.copulate()
'''
nx.draw(keebo.keyboard, node_size=50, font_size=10, pos=nx.get_node_attributes(keebo.keyboard, 'coord'), with_labels=True)
labels = {}
for u,v,data in keebo.keyboard.edges(data=True):
    labels[(u,v)] = data['weight']
nx.draw_networkx_edge_labels(keebo.keyboard,
                             pos=keebo.my_dict,
                             label_pos=0.8,
                             edge_labels=labels,
                             font_size=8,
                             )
plt.gca().invert_yaxis()
plt.show()


print(keebo.list_of_nodes)
'''
