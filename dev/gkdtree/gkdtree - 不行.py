import numpy as np

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
    

class Split:
    def __init__(self) -> None:
        self.cutDim = np.int32(0)
        self.cutVal = np.float32(0.0)
        self.minVal = np.float32(0.0)
        self.maxVal = np.float32(0.0)

        self.left = None
        self.right = None

    def lookup(self, query: np.ndarray[np.float32], ids: np.ndarray[np.int32], weights: np.ndarray[np.float32], nSamples: np.int32, p: np.float32) -> np.int32:
        def pLeft(value: np.float32) -> np.float32:
            val = gCDF(self.cutVal - value)
            minBound = gCDF(self.minVal - value)
            maxBound = gCDF(self.maxVal - value)
            return (val - minBound) / (maxBound - minBound)
        
        val = pLeft(query[self.cutDim])

        if nSamples == 1:
            if np.random.rand() < val:
                return self.left.lookup(query, ids, weights, 1, p * val)
            else:
                return self.right.lookup(query, ids, weights, 1, p * (1 - val))
            
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
            samplesFound += self.left.lookup(query, ids, weights, leftSamples, p * val)
        
        if rightSamples > 0:
            samplesFound += self.right.lookup(query, ids[samplesFound:], weights[samplesFound:], rightSamples, p * (1 - val))

        return samplesFound
    
    def computeBounds(self, mins: np.ndarray[np.float32], maxs: np.ndarray[np.float32]) -> None:
        self.minVal = mins[self.cutDim]
        self.maxVal = maxs[self.cutDim]

        maxs[self.cutDim] = self.cutVal
        self.left.computeBounds(mins, maxs)
        maxs[self.cutDim] = self.maxVal

        mins[self.cutDim] = self.cutVal
        self.right.computeBounds(mins, maxs)
        mins[self.cutDim] = self.minVal

class Leaf:
    def __init__(self) -> None:
        self.id = int(0)
        self.position = np.ndarray((1, ), dtype=np.float32)

    def lookup(self, query: np.ndarray[np.float32], ids: np.ndarray[np.int32], weights: np.ndarray[np.float32], nSamples: np.int32, p: np.float32) -> np.int32:
        q = np.float32(0)
        for i in range(self.position.size):
            delta = query[i] - self.position[i]
            q += delta * delta

        q = np.exp(-q)

        weights[0] = nSamples * q / p
        ids[0] = self.id

        return 1
    
    def computeBounds(self, mins: np.ndarray[np.float32], maxs: np.ndarray[np.float32]) -> None:
        pass

class GKDTree:
    def __init__(self, kd: np.int32, pos: np.ndarray[np.float32], n: np.int32, sBound: np.float32) -> None:
        self.dimensions = kd
        self.sizeBound = sBound
        self.leaves = np.int32(0)

        self.root = self.build(pos, n) # pos is 2d

        self.kdtreeMins = np.zeros((self.dimensions, ), dtype=np.float32)
        self.kdtreeMins.fill(-np.inf)

        self.kdtreeMaxs = np.zeros((self.dimensions, ), dtype=np.float32)
        self.kdtreeMaxs.fill(np.inf)

        self.root.computeBounds(self.kdtreeMins, self.kdtreeMaxs)

    def nLeaves(self) -> np.int32:
        return self.leaves
    
    def lookup(self, query: np.ndarray[np.float32], ids: np.ndarray[np.int32], weights: np.ndarray[np.float32], nSamples: np.int32) -> np.int32:
        return self.root.lookup(query, ids, weights, nSamples, 1)
    
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
            node = Split()
            node.cutDim = longest
            node.cutVal = (maxs[longest] + mins[longest]) / 2.

            pivot = np.int32(0)
            for i in range(n):
                if pos[i][longest] >= node.cutVal:
                    continue

                if i == pivot:
                    pivot += 1
                    continue

                pos[i], pos[pivot] = pos[pivot], pos[i]

                pivot += 1

            node.left = self.build(pos, pivot)
            node.right = self.build(pos[pivot:], n - pivot)

            return node
        
        else:
            node = Leaf()
            node.id = self.leaves
            self.leaves += 1
            node.position.resize((self.dimensions, ))
            for i in range(self.dimensions):
                node.position[i] = (mins[i] + maxs[i]) / 2.

            return node


def gkdtree_filter(pos: np.ndarray[np.float32], pd: np.int32, val: np.ndarray[np.float32], vd: np.int32, n: np.int32) -> np.ndarray:
    points = pos.reshape((n, pd)) # 2d
    tree = GKDTree(pd, points, points.shape[0], np.sqrt(2.0))

    indices = np.ndarray((64, ), dtype=np.int32)
    weights = np.ndarray((64, ), dtype=np.float32)
    leafValues = np.zeros((tree.nLeaves() * vd), dtype=np.float32)

    # Splatting
    for i in range(n):
        results = tree.lookup(pos[i * pd:], indices, weights, 4) # 1d vectors, refs to indices and weights
        for j in range(results):
            for k in range(vd):
                leafValues[indices[j] * vd + k] += val[i * vd + k] * weights[j]

    out = np.zeros((n * vd, ), dtype=np.float32)

    # Slicing
    for i in range(n):
        results = tree.lookup(pos[i * pd:], indices, weights, 64)
        for j in range(results):
            for k in range(vd):
                out[i * vd + k] += leafValues[indices[j] * vd + k] * weights[j]

    return out