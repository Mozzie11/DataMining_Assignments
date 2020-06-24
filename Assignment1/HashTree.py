class Node:
    def __init__(self, container=None):
        self.container = container
        self.leafNode = True
        self.children = []
        self.depth = 0
        self.next = None  

class ListNode:
    def __init__(self, value=None):
        self.value = value
        self.next = None

def new_hash(data):
    left = [1, 3, 7]
    middle = [2, 4, 8]
    right = [5, 6, 9]
    if data in left:
        return 0
    elif data in middle:
        return 1
    elif data in right:
        return 2

class HashTree:
    def __init__(self, max_leaf_size=3):
        self.root = Node([])
        self.max_leaf_size = max_leaf_size

    def initTree(self, datasets):
        for data in datasets:
            self.insert(data, self.root)
        return

    def insert(self, data, node):
        if node.leafNode:
            if len(node.container) < self.max_leaf_size:
                node.container.append(data)
                return
            elif node.depth == 3:  # put them together when exausted
                self.linkList(data, node)
                return
            else:
                self.generate(data, node)
                return
        else:
            hv = new_hash(data[node.depth])
            self.insert(data, node.children[hv])
            return

    def linkList(self, data, node):
        linked_node = ListNode(data)
        while node.next is not None:
            node = node.next
        node.next = linked_node
        return

    def generate(self, data, node):
        node.leafNode = False
        for i in range(self.max_leaf_size):
            children_node = Node([])
            children_node.depth = node.depth + 1
            node.children.append(children_node)
        for items in node.container:
            hv = new_hash(items[node.depth])
            self.insert(items, node.children[hv])

        hv = new_hash(data[node.depth])
        self.insert(data, node.children[hv])
        node.container = []
        return

    def scan(self, node):
        res = []
        if node.leafNode:
            if node.next is None:
                return node.container
            else:
                res += node.container
                while node.next is not None:
                    node = node.next
                    res += [node.value]
        else:
            for child in node.children:
                res += [self.scan(child)]
        return res

if __name__ == "__main__":
    myTree = HashTree()

    datasets = [[1, 2, 3], [1, 3, 9], [1, 4, 5], [1, 4, 6], [1, 5, 7], [1, 5, 9], [1, 6, 8], [1, 6, 9], [1, 8, 9], \
                [2, 3, 9], [2, 5, 6], [2, 5, 7], [2, 5, 9], [2, 6, 7], [2, 6, 8], [2, 6, 9], [2, 7, 8], [2, 7, 9], [2, 8, 9], \
                [3, 4, 6], [3, 4, 8], [3, 7, 8], \
                [4, 5, 6], [4, 5, 8], [4, 5, 9], [4, 7, 8], [4, 7, 9], [4, 8, 9], \
                [5, 6, 7], [5, 6, 8], [5, 7, 8], [5, 7, 9], [5, 8, 9], \
                [6, 7, 9], [6, 8, 9], [7, 8, 9]]

    myTree.initTree(datasets)
    print(myTree.scan(myTree.root))
