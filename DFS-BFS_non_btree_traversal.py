 # DFS Traversal algorithm for non-binary tree using recursion
 # Preorder Traversal (root-left-right)

# BFS Traversal algorithm for non-binary tree using queue

class Node(object):
    
    def __init__(self, name, children=[]):
        self.name = name
        self.children = children

    def dfs(self, node, lst):

        lst.append(node.name)

        if len(node.children) != 0:
            for idx in range(len(node.children)):
                self.dfs(node.children[idx],lst)


    def bfs(self, node, lst):

        queue = []
        queue.append(node)

        while len(queue) > 0:
            lst.append(queue[0].name)
            node = queue.pop(0)

            if len(node.children) != 0:
                for idx in range(len(node.children)):
                    queue.append(node.children[idx])



def create_tree():
    return Node('root', [Node("PYTHON", [Node("I"),Node("Love"),Node("Python")]),Node("Hello", [Node("Hello", [Node("Hello", [Node("World")])])])])

'''
                    root
                  /      \
               PYTHON    Hello
             /   |   \      \
            I  Love Python  Hello
                              \
                             Hello
                                \
                                World
'''



if __name__ == '__main__':
    lst = []
    root = create_tree()
    # root2 = Node(1, [Node(2, [Node(5, [Node(10)]), Node(6, [Node(11), Node(12), Node(13)])]), Node(3), Node(4, [Node(7), Node(8), Node(9)])])
    root3 = Node(1, [Node(2, [Node(4, [Node(8), Node(9)]), Node(5, [Node(10), Node(11)])]), Node(3, [Node(6, [Node(12), Node(13)]), Node(7, [Node(14), Node(15)])])])
    root3.dfs(root3, lst)
    print(lst)