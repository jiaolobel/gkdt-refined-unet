"""
Gaussian KD Tree in Python

restart gkdtree.py

bugfix, type hint removed.

transform recursion to loop. old version refers to gkdtree - copy (2).py
"""

import time
import numpy as np

# from numba import jit

class GKDTree:
    class Node:
        def __init__(self):
            self.ntype = None # 0 for Split, 1 for Leaf

        def lookup(self, query: np.ndarray, ids: np.ndarray, weights: np.ndarray, nSamples: np.int32, p: np.float32) -> np.int32:
            pass

        def computeBounds(self, mins: np.ndarray, maxs: np.ndarray) -> None:
            pass

    class Split(Node):
        def __init__(self) -> None:
            self.ntype = 0 # 0 for Split

            self.cutDim: np.int32 = 0
            self.cutVal: np.float32 = 0.0
            self.minVal: np.float32 = 0.0
            self.maxVal: np.float32 = 0.0

            self.left = None
            self.right = None 

        def pLeft(self, value: np.float32) -> np.float32:
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
            
            val: np.float32 = gCDF(self.cutVal - value)
            minBound: np.float32 = gCDF(self.minVal - value)
            maxBound: np.float32 = gCDF(self.maxVal - value)
            return (val - minBound) / (maxBound - minBound) 

        def lookup(self, query: np.ndarray, ids: np.ndarray, weights: np.ndarray, nSamples: np.int32, p: np.float32):
            # print(query, self.cutDim)
            val: np.float32 = self.pLeft(query[self.cutDim])
            # print("query[cutDim]", query[self.cutDim], "val: ", val)

            if nSamples == 1:
                if np.random.rand() < val:
                    return self.left.lookup(query, ids, weights, 1, p * val)
                else:
                    return self.right.lookup(query, ids, weights, 1, p * (1 - val))
                
            leftSamples: np.int32 = np.int32(np.floor(val * nSamples))
            rightSamples: np.int32 = np.int32(np.floor((1 - val) * nSamples))

            if leftSamples + rightSamples != nSamples:
                fval: np.float32 = np.float32(val * nSamples - leftSamples)
                if np.random.rand() < fval:
                    leftSamples += 1
                else:
                    rightSamples += 1

            samplesFound: np.int32 = 0

            if leftSamples > 0:
                samplesFound += self.left.lookup(query, ids, weights, leftSamples, p * val)
            
            if rightSamples > 0:
                samplesFound += self.right.lookup(query, ids[samplesFound:], weights[samplesFound:], rightSamples, p * (1 - val))

            return samplesFound
                
        def computeBounds(self, mins: np.ndarray, maxs: np.ndarray):
            # print("mins:", mins)
            # print("maxs:", maxs)
            self.minVal = mins[self.cutDim]
            self.maxVal = maxs[self.cutDim]

            maxs[self.cutDim] = self.cutVal
            self.left.computeBounds(mins, maxs)
            maxs[self.cutDim] = self.maxVal

            mins[self.cutDim] = self.cutVal
            self.right.computeBounds(mins, maxs)
            mins[self.cutDim] = self.minVal 

    class Leaf(Node):
        def __init__(self) -> None:
            self.ntype = 1 # 1 for Leaf

            self.nid: np.int32 = 0
            self.position = None
            # self.position = np.ndarray((GKDTree.dimensions, ), dtype=np.float32) 

        def lookup(self, query: np.ndarray, ids: np.ndarray, weights: np.ndarray, nSamples: np.int32, p: np.float32) -> np.int32:
            # q: np.float32 = 0.0
            # for i in range(self.position.shape[0]):
            #     delta: np.float32 = query[i] - self.position[i]
            #     q += delta * delta
            # q = np.exp(-q)
            
            q = np.exp(-np.sum((query - self.position) ** 2))

            weights[0] = np.float32(nSamples) * q / p
            ids[0] = self.nid

            return 1 

    def __init__(self, kd: np.int32, pos: np.ndarray, n: np.int32, sBound: np.float32) -> None:
        self.dimensions: np.int32 = kd
        self.sizeBound: np.float32 = sBound
        self.leaves: np.int32 = 0
        
        # self.weights = np.zeros((64, ), dtype=np.float32)
        # self.ids = np.zeros((64, ), dtype=np.int32)

        self.root: GKDTree.Node = self.build(pos, n) # pos is 2d

        kdtreeMins = np.zeros((self.dimensions, ), dtype=np.float32)
        kdtreeMins.fill(-np.inf)

        kdtreeMaxs = np.zeros((self.dimensions, ), dtype=np.float32)
        kdtreeMaxs.fill(np.inf)

        self.root.computeBounds(kdtreeMins, kdtreeMaxs)

    def nLeaves(self) -> np.int32:
        return self.leaves
    
    def lookup(self, query: np.ndarray, ids: np.ndarray, weights: np.ndarray, nSamples: np.int32) -> np.int32:
        class Param:
            def __init__(self, node, nSamples, p):
                self.node: GKDTree.Node = node
                self.nSamples: np.int32 = nSamples
                self.p: np.float32 = p
        
        samplesFound: np.int32 = 0
        param_list: list = list()
        param_list.append(Param(self.root, nSamples, 1))

        while param_list: 
            param: Param = param_list.pop(0)

            if param.node.ntype == 0: # 0 for Split
                val: np.float32 = param.node.pLeft(param.query[param.node.cutDim])

                if param.nSamples == 1:
                    if np.random.rand() < val:
                        param_list.insert(0, Param(param.node.left, param.query, param.ids, param.weights, 1, param.p * val))
                        continue
                    else:
                        param_list.insert(0, Param(param.node.right, param.query, param.ids, param.weights, 1, param.p * (1 - val)))
                        continue
                
                leftSamples: np.int32 = np.int32(np.floor(val * param.nSamples))
                rightSamples: np.int32 = np.int32(np.floor((1 - val) * param.nSamples))

                if leftSamples + rightSamples != param.nSamples:
                    fval: np.float32 = np.float32(val * param.nSamples - leftSamples)
                    if np.random.rand() < fval:
                        leftSamples += 1
                    else:
                        rightSamples += 1

                if leftSamples > 0:
                    param_list.insert(0, Param(param.node.left, param.query, param.ids, param.weights, leftSamples, param.p * val))
                
                if rightSamples > 0: 
                    param_list.insert(0, Param(param.node.right, param.ids[samplesFound]))

                    
        return self.root.lookup(query, ids, weights, nSamples, 1)
    

    
    # def lookup(self, node: Union[Split, Leaf], query: np.ndarray[np.float32], ids_idx:np.int32, weights_idx: np.int32, nSamples: np.int32, p: np.float32) -> np.int32:
    #     if node.__class__.__name__ == "Split":
    #         # print("lookup in Split")
    #         def gCDF(x: np.float32) -> np.float32:
    #             x *= 0.81649658092772592 # 81649658092772592

    #             if x < -2:
    #                 return 0
    #             elif x < -1:
    #                 x += 2
    #                 x *= x
    #                 x *= x
    #                 return x
    #             elif x < 0:
    #                 return 12 + x * (16 - x * x * (8 + 3 * x))
    #             elif x < 1:
    #                 return 12 + x * (16 - x * x * (8 - 3 * x))
    #             elif x < 2:
    #                 x -= 2
    #                 x *= x
    #                 x *= x
    #                 return 24 - x
    #             else:
    #                 return 24
            
    #         def pLeft(value: np.float32) -> np.float32:
    #             val = gCDF(node.cutVal - value)
    #             minBound = gCDF(node.minVal - value)
    #             maxBound = gCDF(node.maxVal - value)
    #             return (val - minBound) / (maxBound - minBound)
            
    #         val = pLeft(query[node.cutDim])

    #         if nSamples == 1:
    #             if np.random.rand() < val:
    #                 return self.lookup(node.left, query, 0, 0, 1, p * val)
    #             else:
    #                 return self.lookup(node.right, query, 0, 0, 1, p * (1 - val))
                
    #         leftSamples = np.int32(np.floor(val * nSamples))
    #         rightSamples = np.int32(np.floor((1 - val) * nSamples))

    #         if leftSamples + rightSamples != nSamples:
    #             fval = np.float32(val * nSamples - leftSamples)
    #             if np.random.rand() < fval:
    #                 leftSamples += 1
    #             else:
    #                 rightSamples += 1

    #         samplesFound = np.int32(0)

    #         if leftSamples > 0:
    #             samplesFound += self.lookup(node.left, query=query, ids_idx=0, weights_idx=0, nSamples=leftSamples, p=p * val)
            
    #         if rightSamples > 0:
    #             samplesFound += self.lookup(node.right, query=query, ids_idx=samplesFound, weights_idx=samplesFound, nSamples=rightSamples, p=p * (1 - val))

    #         return samplesFound
    #     else:
    #         # print("lookup in Leaf")
    #         q = np.float32(0)
    #         for i in range(node.position.size):
    #             delta = query[i] - node.position[i]
    #             q += delta * delta

    #         q = np.exp(-q)

    #         self.weights[weights_idx] = nSamples * q / p
    #         self.ids[ids_idx] = node.nid

    #         # print("new weights[weights_idx]", self.weights[weights_idx])
    #         # print("new ids[ids_idx]", self.ids[ids_idx])

    #         return 1 

    # def computeBounds(self, node: Union[Split, Leaf]) -> None:
    #     if node.__class__.__name__ == "Split":
    #         node.minVal = self.kdtreeMins[node.cutDim]
    #         node.maxVal = self.kdtreeMaxs[node.cutDim]

    #         self.kdtreeMaxs[node.cutDim] = node.cutVal
    #         self.computeBounds(node.left)
    #         self.kdtreeMaxs[node.cutDim] = node.maxVal

    #         self.kdtreeMins[node.cutDim] = node.cutVal
    #         self.computeBounds(node.right)
    #         self.kdtreeMins[node.cutDim] = node.minVal 
    
    def build(self, pos: np.ndarray, n: np.int32) -> Node:
        # mins = np.ndarray((self.dimensions, ), dtype=np.float32)
        # mins.fill(np.inf)
        # maxs = np.ndarray((self.dimensions, ), dtype=np.float32)
        # maxs.fill(-np.inf)

        # for i in range(n):
        #     for j in range(self.dimensions):
        #         if pos[i][j] < mins[j]:
        #             mins[j] = pos[i][j]
        #         if pos[i][j] > maxs[j]:
        #             maxs[j] = pos[i][j]

        # print(pos.shape[0], n)

        mins, maxs = pos.min(axis=0), pos.max(axis=0)
        # mins, maxs = pos.min(axis=0), pos.max(axis=0)

        # if not np.allclose(mins, mins2):
        #     print("min", mins, mins2, mins.dtype, mins2.dtype)
        # if not np.allclose(maxs, maxs2):
        #     print("max", maxs, maxs2)
        # # print(np.allclose(mins, mins2), np.allclose(maxs, maxs2))

        # longest: np.int32 = 0
        # for i in range(1, self.dimensions):
        #     if (maxs[i] - mins[i]) > (maxs[longest] - mins[longest]):
        #         longest = i

        longest = (maxs - mins).argmax()

        # if not longest == longest2:
        #     print("longest: ", longest, "longest2: ", longest2, longest == longest2)

        if (maxs[longest] - mins[longest]) > self.sizeBound:
            node = self.Split()
            node.cutDim = longest
            node.cutVal = (maxs[longest] + mins[longest]) / 2.

            pivot: np.int32 = 0
            for i in np.arange(n):
                if pos[i][longest] >= node.cutVal:
                    continue

                if i == pivot:
                    pivot += 1
                    continue

                # print("before swap {} and {}: ".format(i, pivot), pos[i])
                pos[[i, pivot]] = pos[[pivot, i]]
                # print("after swap {} and {}: ".format(i, pivot), pos[i])
                pivot += 1

            node.left = self.build(pos[:pivot], pivot)
            node.right = self.build(pos[pivot:], n - pivot)

            return node
        
        else:
            node = self.Leaf()
            node.nid = self.leaves
            # print("leaf.nid:", node.nid)
            self.leaves += 1
            node.position = np.ndarray((self.dimensions, ), dtype=np.float32)
            node.position[:] = (mins + maxs) / 2.
            # for i in range(self.dimensions):
            #     node.position[i] = (mins[i] + maxs[i]) / 2.

            # print("building leaf: ", node.nid, node.position)

            return node 

# @jit(nopython=True)
def gkdtree_filter(pos: np.ndarray, pd: np.int32, val: np.ndarray, vd: np.int32, n: np.int32) -> np.ndarray:
    # points = pos.copy().reshape((n, pd)) # to 2d
    pos = pos.reshape((n, pd))
    points = pos.copy()
    val = val.reshape((n, vd)) # to 2d

    # print("Before creating tree")
    # for i in range(n):
    #     print("pos ", i, points[i])

    print("Build tree")
    start = time.time()
    tree = GKDTree(pd, points, n, np.sqrt(2.0))
    print("Time", time.time() - start)

    # print("After creating tree")
    # for i in range(n):
    #     print("pos ", i, points[i])
    # print("tree.ids:", tree.ids)
    # print("tree.weights: ", tree.weights)

    # points = points.reshape((-1, ))

    # print("all close:", np.allclose(points, pos))

    indices = np.ndarray((64, ), dtype=np.int32)
    weights = np.ndarray((64, ), dtype=np.float32)
    leafValues = np.zeros((tree.nLeaves(), vd, ), dtype=np.float32)

    # Splatting
    # print("Splat")
    # for i in range(n):
    #     # print("splat lookup")
    #     results = tree.lookup(pos[i * pd:], indices, weights, 4) # 1d vectors, refs to indices and weights
    #     for j in range(results):
    #         for k in range(vd):
    #             leafValues[indices[j] * vd + k] += val[i * vd + k] * weights[j]

    print("Splat")
    start = time.time()
    for i in np.arange(n):
        results = tree.lookup(pos[i], indices, weights, 4)
        for j in np.arange(results):
            leafValues[indices[j]] += val[i] * weights[j]
    print("Time", time.time() - start)


    out = np.zeros((n, vd, ), dtype=np.float32)

    # # Slicing
    # print("Slice")
    # for i in range(n):
    #     results = tree.lookup(pos[i * pd:], indices, weights, 64)
    #     for j in range(results):
    #         for k in range(vd):
    #             out[i * vd + k] += leafValues[indices[j] * vd + k] * weights[j]

    print("Slice")
    start = time.time()
    for i in np.arange(n):
        results = tree.lookup(pos[i], indices, weights, 64)
        for j in np.arange(results):
            out[i] += leafValues[indices[j]] * weights[j]
    print("Time", time.time() - start)

    return out