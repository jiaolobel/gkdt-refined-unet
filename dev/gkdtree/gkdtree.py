from typing import Union

import numpy as np

class GKDTree:
    class Split:
        def __init__(self) -> None:
            self.cutDim = np.int32(0)
            self.cutVal = np.float32(0.0)
            self.minVal = np.float32(0.0)
            self.maxVal = np.float32(0.0)

            self.left = None
            self.right = None 

    class Leaf:
        def __init__(self) -> None:
            self.nid = int(0)
            self.position = None
            # self.position = np.ndarray((GKDTree.dimensions, ), dtype=np.float32) 

    def __init__(self, kd: np.int32, pos: np.ndarray[np.float32, np.float32], n: np.int32, sBound: np.float32) -> None:
        self.dimensions = kd
        self.sizeBound = sBound
        self.leaves = np.int32(0)
        
        self.weights = np.zeros((64, ), dtype=np.float32)
        self.ids = np.zeros((64, ), dtype=np.int32)

        self.kdtreeMins = np.zeros((self.dimensions, ), dtype=np.float32)
        self.kdtreeMins.fill(-np.inf)

        self.kdtreeMaxs = np.zeros((self.dimensions, ), dtype=np.float32)
        self.kdtreeMaxs.fill(np.inf)

        self.root = self.build(pos, n) # pos is 2d
        self.computeBounds(self.root)

    def nLeaves(self) -> np.int32:
        return self.leaves
    
    # def lookup(self, query: np.ndarray[np.float32], ids_idx: np.int32, weights_idx: np.int32, nSamples: np.int32) -> np.int32:
        # return self.lookup(self.root, query, ids_idx, weights_idx, nSamples, 1)
    def lookup_tree(self, query: np.ndarray[np.float32], ids_idx: np.int32, weights_idx: np.int32, nSamples: np.int32) -> np.int32:
        return self.lookup(self.root, query=query, ids_idx=ids_idx, weights_idx=weights_idx, nSamples=nSamples, p=1.0)
    
    def lookup(self, node: Union[Split, Leaf], query: np.ndarray[np.float32], ids_idx:np.int32, weights_idx: np.int32, nSamples: np.int32, p: np.float32) -> np.int32:
        if node.__class__.__name__ == "Split":
            # print("lookup in Split")
            def gCDF(x: np.float32) -> np.float32:
                x *= 0.81649658092772592 # 81649658092772592

                if x < -2:
                    return 0
                elif x < -1:
                    x += 2
                    x *= x
                    x *= x
                    return x
                elif x < 0:
                    return 12 + x * (16 - x * x * (8 + 3 * x))
                elif x < 1:
                    return 12 + x * (16 - x * x * (8 - 3 * x))
                elif x < 2:
                    x -= 2
                    x *= x
                    x *= x
                    return 24 - x
                else:
                    return 24
            
            def pLeft(value: np.float32) -> np.float32:
                val = gCDF(node.cutVal - value)
                minBound = gCDF(node.minVal - value)
                maxBound = gCDF(node.maxVal - value)
                return (val - minBound) / (maxBound - minBound)
            
            val = pLeft(query[node.cutDim])

            if nSamples == 1:
                if np.random.rand() < val:
                    return self.lookup(node.left, query, 0, 0, 1, p * val)
                else:
                    return self.lookup(node.right, query, 0, 0, 1, p * (1 - val))
                
            leftSamples = np.int32(np.floor(val * nSamples))
            rightSamples = np.int32(np.floor((1 - val) * nSamples))

            if leftSamples + rightSamples != nSamples:
                fval = np.float32(val * nSamples - leftSamples)
                if np.random.rand() < fval:
                    leftSamples += 1
                else:
                    rightSamples += 1

            samplesFound = np.int32(0)

            if leftSamples > 0:
                samplesFound += self.lookup(node.left, query=query, ids_idx=0, weights_idx=0, nSamples=leftSamples, p=p * val)
            
            if rightSamples > 0:
                samplesFound += self.lookup(node.right, query=query, ids_idx=samplesFound, weights_idx=samplesFound, nSamples=rightSamples, p=p * (1 - val))

            return samplesFound
        else:
            # print("lookup in Leaf")
            q = np.float32(0)
            for i in range(node.position.size):
                delta = query[i] - node.position[i]
                q += delta * delta

            q = np.exp(-q)

            self.weights[weights_idx] = nSamples * q / p
            self.ids[ids_idx] = node.nid

            # print("new weights[weights_idx]", self.weights[weights_idx])
            # print("new ids[ids_idx]", self.ids[ids_idx])

            return 1 

    def computeBounds(self, node: Union[Split, Leaf]) -> None:
        if node.__class__.__name__ == "Split":
            node.minVal = self.kdtreeMins[node.cutDim]
            node.maxVal = self.kdtreeMaxs[node.cutDim]

            self.kdtreeMaxs[node.cutDim] = node.cutVal
            self.computeBounds(node.left)
            self.kdtreeMaxs[node.cutDim] = node.maxVal

            self.kdtreeMins[node.cutDim] = node.cutVal
            self.computeBounds(node.right)
            self.kdtreeMins[node.cutDim] = node.minVal 
    
    def build(self, pos: np.ndarray[np.float32, np.float32], n: np.int32):
        mins = np.ndarray((self.dimensions, ), dtype=np.float32)
        mins.fill(np.inf)
        maxs = np.ndarray((self.dimensions, ), dtype=np.float32)
        maxs.fill(-np.inf)

        for i in range(n):
            for j in range(self.dimensions):
                if pos[i][j] < mins[j]:
                    mins[j] = pos[i][j]
                if pos[i][j] > maxs[j]:
                    maxs[j] = pos[i][j]

        longest = np.int32(0)
        for i in range(1, self.dimensions):
            if maxs[i] - mins[i] > maxs[longest] - mins[longest]:
                longest = i

        if maxs[longest] - mins[longest] > self.sizeBound:
            node = self.Split()
            node.cutDim = longest
            node.cutVal = (maxs[longest] + mins[longest]) / 2.

            pivot = np.int32(0)
            for i in range(n):
                if pos[i][longest] >= node.cutVal:
                    continue

                if i == pivot:
                    pivot += 1
                    continue

                # print("before swap: ", pos[i], pos[pivot])
                # pos[i], pos[pivot] = pos[pivot], pos[i]
                pos[[i, pivot]] = pos[[pivot, i]]
                # print("after swap: ", pos[i], pos[pivot])
                pivot += 1

            node.left = self.build(pos, pivot)
            node.right = self.build(pos[pivot:], n - pivot)

            return node
        
        else:
            node = self.Leaf()
            node.nid = self.leaves
            # print("leaf.nid:", node.nid)
            self.leaves += 1
            node.position = np.ndarray((self.dimensions, ), dtype=np.float32)
            for i in range(self.dimensions):
                node.position[i] = (mins[i] + maxs[i]) / 2.

            return node 


def gkdtree_filter(pos: np.ndarray[np.float32], pd: np.int32, val: np.ndarray[np.float32], vd: np.int32, n: np.int32) -> np.ndarray:
    points = pos.copy().reshape((n, pd)) # 2d
    tree = GKDTree(pd, points, points.shape[0], np.sqrt(2.0))
    # print("tree.ids:", tree.ids)
    # print("tree.weights: ", tree.weights)

    # indices = np.ndarray((64, ), dtype=np.int32)
    # weights = np.ndarray((64, ), dtype=np.float32)
    leafValues = np.zeros((tree.nLeaves() * vd), dtype=np.float32)

    # Splatting
    for i in range(n):
        print("splat lookup")
        results = tree.lookup_tree(query=pos[i * pd:], ids_idx=0, weights_idx=0, nSamples=4) # 1d vectors, refs to indices and weights
        for j in range(results):
            for k in range(vd):
                leafValues[tree.ids[j] * vd + k] += val[i * vd + k] * tree.weights[j]

    out = np.zeros((n * vd, ), dtype=np.float32)

    # Slicing
    for i in range(n):
        print("slice lookup")
        results = tree.lookup_tree(query=pos[i * pd:], ids_idx=0, weights_idx=0, nSamples=64)
        for j in range(results):
            for k in range(vd):
                out[i * vd + k] += leafValues[tree.ids[j] * vd + k] * tree.weights[j]

    return out