import sys
import os

graph = {}
pie = {0: []}
active = {0: []}

try:
    epsilon = int(sys.argv[1])
    input_file = sys.argv[2]
    output_file = sys.argv[3]
except IndexError:
    print "Usage :: python %s <epsilon> <graph_file> <output_file>" % sys.argv[0]
    sys.exit(1)

num_nodes_enron = 8657
for i in xrange(0, num_nodes_enron):
    graph[i] = []

def create_graph():
    # supports directed graph
    # fg = open(input_file, 'r')
    #print "Starting allocation of memory for Graph"
    for line in open(input_file):
        # source, destination, weight
        line = line.strip()
        line = line.split(',')
        source = int(line[0])
        dest = int(line[1])
        wt = int(line[2])
        graph[source].append(dest)
    #print "Graph %s populated in memory " % input_file
    degree = 0
    zero_nodes = 0
    for node in graph.keys():
        if len(graph[node]) == 0:
            zero_nodes += 1
        degree += len(graph[node])
    print 'num nodes: %s, zero deg nodes: %s, total degree: %s, avg degree: %.2f' % (num_nodes_enron, zero_nodes,
                                                                                     degree,
                                                                                     float(degree) / (num_nodes_enron-zero_nodes))
    return graph


def initialize(pie, active):
    for index in xrange(len(graph)):
        pie[0].append(index)
    for val in pie[0]:
        active[0].append(val)
    return


def degree_dist(activeCell):
    # print graph
    fofU = [0 for i in xrange(len(graph))]
    for index in graph.keys():
        fofU[index] = len(set(graph[index]) & set(activeCell))
        # print fofU
    return fofU


#outDir = 'partitions/ep' + str(epsilon) + '/'
def display(pie):
    fo = open(output_file, "w")
    for key in pie.keys():
        cell = pie[key]
        for v in sorted(cell):
            fo.write(str(v) + " ")
        fo.write("\n")
    fo.close()


graph = create_graph()
#print "Graph Creation Complete"

initialize(pie, active)
#print "Active Partition Initialized"

iteration = 0


def split(cell, fofU, epsilon):
    def align_splitted_cell(splitted_cell):
        idx = 1
        aligned_splitted_cells = {}
        for key in sorted(splitted_cell.keys()):
            aligned_splitted_cells[idx] = []
            for v in splittedCells[key]:
                aligned_splitted_cells[idx].append(v)
            idx += 1
        return aligned_splitted_cells


    splittedCells = {}
    alignedSplittedCells = {}

    for vertex in cell:
        key = fofU[vertex]
        if key not in splittedCells:
            splittedCells[key] = [vertex]
        else:
            splittedCells[key].append(vertex)
    # the partition till now in splittedCells is per Equitable Partition definition

    sortedSplittedCellsKeys = sorted(splittedCells.keys())
    if len(sortedSplittedCellsKeys) > 1:
        splitBoundaries = {}
        boundaryIndex = 1
        startKey = sortedSplittedCellsKeys[0]
        splitBoundaries[boundaryIndex] = [startKey]
        for i in xrange(1, len(sortedSplittedCellsKeys)):
            endKey = sortedSplittedCellsKeys[i]
            if (endKey - startKey) <= epsilon:
                splitBoundaries[boundaryIndex].append(endKey)
            else:
                boundaryIndex += 1
                splitBoundaries[boundaryIndex] = [endKey]
                startKey = endKey
        for k in sorted(splitBoundaries.keys()):
            alignedSplittedCells[k] = []
            for b in splitBoundaries[k]:
                for v in splittedCells[b]:
                    alignedSplittedCells[k].append(v)
    else:
        alignedSplittedCells = align_splitted_cell(splittedCells)
    return alignedSplittedCells


def is_cell_in_active(sortedCell):
    for key in active.keys():
        cellInActive = sorted(active[key])
        if cellInActive == sortedCell:
            return key
    return -1


maxIndexInPie = 0
maxIndexInActive = 0
noOfVertices = len(graph)
noIters = 0

while active and (len(pie) != noOfVertices):
    noIters += 1
    activeKeys = sorted(active.keys())
    minIdx = min(activeKeys)
    activeCell = active[minIdx]
    del (active[minIdx])
    fofU = degree_dist(activeCell)

    for key in pie.keys():
        cell = pie[key]
        splittedCells = split(cell, fofU, epsilon)
        splittedCellsKeys = splittedCells.keys()
        if len(splittedCellsKeys) == 1:
            continue
        s = max(splittedCellsKeys)  # s in paper
        t = 1  # t in paper
        tFinder = {}
        for k in sorted(splittedCellsKeys):
            l = len(splittedCells[k])
            if l > s:
                break
            else:
                if l not in tFinder:
                    tFinder[l] = [k]
                else:
                    tFinder[l].append(k)
        tFinderKeys = sorted(tFinder.keys())
        if len(tFinderKeys) > 0:
            t = tFinder[max(tFinderKeys)][0]  # Let t be tbe smallest integer such |X_t| is max(1 <= t <= s)
        else:
            t = 1

        del (pie[key])
        for s in splittedCells.keys():
            maxIndexInPie += 1
            pie[maxIndexInPie] = splittedCells[s]

        doesCellBelongToActive = is_cell_in_active(sorted(cell))  # check if cell in a member of active
        if doesCellBelongToActive != -1:
            del (active[doesCellBelongToActive])
            active[doesCellBelongToActive] = splittedCells[t]

        for i in xrange(1, t):
            maxIndexInActive += 1
            active[maxIndexInActive] = splittedCells[i]
        for i in xrange(t + 1, s + 1):
            maxIndexInActive += 1
            active[maxIndexInActive] = splittedCells[i]
        #print pie
    if noIters % 100 == 0:
        print 'No iters = %s, Size of active = %s, Size of pie = %s' % (noIters, len(active), len(pie))
display(pie)
#print 'No iters = ', noIters

# TODO: 1. Sort Active according to size (lowest to highest) 2. Introduce FoFUDistributed and run time experiments