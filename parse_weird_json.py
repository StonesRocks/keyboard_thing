import json


class WeirdJson:
    # initialise by reading file contents
    def __init__(self, json_file):
        self.json_file = json_file
        with open(json_file, encoding="utf8") as f:
            self.json_dict = json.load(f)
            #print(type(self.json_dict))
            #print(len(self.json_dict))
            for i in self.json_dict:
                print("This is a new row!")
                for j in i:
                    #print(j)
                    if isinstance(j,str):
                        j.rstrip()
                        print(j.rstrip(), "is a str")
                    elif isinstance(j, dict):
                        #print("This a dict with ", len(j), "items")
                        for key in j:
                            if key == 'x':
                                print("add ", j['x'], " horizontal space")

                            elif key == 'y':
                                print("add ", j['y'], " vertical space")
                            elif key == 'w':
                                print("add key width by ", j['w'])
                            elif key == 'h':
                                print("add key height(downwards) by ", j['h'])
                            elif key == 'x2' or key == 'y2' or key == 'w2' or key == 'h2':
                                print("uh oh")
                            elif key == 'a' or key == 'f' or key == 'f2' or key == 'p' or key == 's':
                                print("something about font")



                    else:
                        print(j, "is ",type(j))
                #print(type(line))
                #print(line)
                '''
                if line.find("{") == 4 or line.find("}") == 4:
                    print("skip")
                elif line.find("x") != -1:
                    #print(line.find(":"))
                    #print(len(line))
                    print(line[line.find(":")+2:len(line)])
                '''





if __name__ == '__main__':
    wj = WeirdJson('keyboard-layout_1.json')