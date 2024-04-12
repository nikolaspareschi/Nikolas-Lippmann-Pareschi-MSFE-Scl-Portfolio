# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 23:53:53 2018

@author: Nikolas
"""

class Node:
    def __init__(self, val):
        self.left = None
        self.right = None
        self.value = val


class Tree:
    def __init__(self):
        self.root = None

    def DeleteTree(self):
        self.root = None
 
    def IsEmpty(self):
        return self.root == None

    def PrintTree(self):
        if(self.root != None):
            self._PrintTree(self.root)

    def _PrintTree(self, node):
        if(node != None):
            self._PrintTree(node.left)
            print(str(node.value) + ' ')
            self._PrintTree(node.right)

    def find(self, val):
        if(self.root != None):
            x = self.root
            return self.findx(val, self.root)
        else:
            return None


    def findx(self, val, node):
        if(val == node.value):
            print node.value, 'Is in our B-tree'
            return node.value
        elif(val < node.value):
            if (node.left != None):
                return self.findx(val, node.left)
            else:
                print val, 'Is not in our B-tree' 
        elif(val > node.value):
            if (node.right != None):
                return self.findx(val, node.right)
            else:
                print val, 'Is not in our B-tree'

    
    def search(self, k, x=None):
        """Search the B-Tree for the key k.
        
        args
        =====================
        k : Key to search for
        x : (optional) Node at which to begin search. Can be None, in which case the entire tree is searched.
        
        """
        if isinstance(x, BTreeNode):
            i = 0
            while i < len(x.keys) and k > x.keys[i]: 
                print x.keys[i] 
                i += 1
            if i < len(x.keys) and k == x.keys[i]:      
                return (x, i)
            elif x.leaf:                                
                return None
            else:                                       
                return self.search(k, x.c[i])
        else:                                           
            return self.search(k, self.root)


    def add(self, val):
        if(self.root == None):
            self.root = Node(val)
        else:
            self._add(val, self.root)


    def _add(self, val, node):
        if(val < node.value):
            if(node.left != None):
                self._add(val, node.left)
            else:
                node.left = Node(val)
        else:
            if(node.right != None):
                self._add(val, node.right)
            else:
                node.right = Node(val)

    def getRoot(self):
        return self.root



tree = Tree()
tree.add(0)
tree.add(1)
tree.add(2)
tree.add(3)
tree.add(4)
tree.add(5)
tree.add(6)
tree.add(7)
tree.add(8)
tree.add(9)
tree.PrintTree()
