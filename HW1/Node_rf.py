
class Node:
    
    def __init__(self, data):
        self.left = None
        self.right = None
        self.data = data
        self.threshold_idx = -1
        self.threshold = -1
        self.leaf = True
        self.pure = True
        self.label = -1
        self.check_prune = False
        # determine whether this set data is pure or not
        if len(data) > 1:
            check_label = data[0][-1]

            # set majority class as pre-pruning
            for i in range(1,len(data)):
                    if check_label != data[i][-1]:
                        self.pure = False

            if len(data) < 10 and not(self.pure):
                #print "pre-pruning!!"
                #pprint.pprint(self.data)
                label_count = [0,0,0]
                for label in self.data:
                    if label[-1] == 'setosa':
                        label_count[0] += 1
                    elif label[-1] == 'versicolor':
                        label_count[1] += 1
                    elif label[-1] == 'virginica' :
                        label_count[2] += 1
                    
                max_idx = max(enumerate(label_count),key=lambda x: x[1])[0]
                    
                if max_idx == 0:
                    self.label = 'setosa'
                elif max_idx == 1:
                    self.label = 'versicolor'
                else:
                    self.label = 'virginica'
                self.pure = True
                self.check_prune = True


        # if all elements are same label, set the node.label
        if self.pure and not(self.check_prune):
            #print "here"
            #print data[0]
            self.label = data[0][-1]


    def set_threshold_idx(self, index):
        self.threshold_idx = index

    def set_threshold(self, value):
        self.threshold = value

    def set_right(self,data):
        # contain set
        self.leaf = False
        self.right = data

    def set_left(self,data):
        # contain set
        self.leaf = False
        self.left = data  



