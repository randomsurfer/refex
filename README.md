#ReFeX#
An implementation of Recursive Feature Extraction (ReFeX). 

Described in the paper "It's Who You Know: Graph Mining using Recursive Structural Features" by Henderson <i>et al.</i> 

Link to the <a href="http://dl.acm.org/citation.cfm?id=2020512">paper</a>.

<b>Disclaimer:</b> I have implemented this paper purely for research, educational experimentation and non-profit use. Hence doesn't come with any guarantees. Please contact the authors for any other purpose.

**Running the Code:**

* Requires `python 2.7`.

* Install package dependencies using
`pip install -r requirements.txt`

* Run the code using `python refex.py`

* Run `python refex.py -h` for help and other program options.

* Input graph (directed graphs are supported) is specified with comma separated edge information in each line.
    * Line `1,2,3` in the input graph depicts `source, destination, edge-weight`.
* The epsilon equitable partition (eEP) based features are also supported. The `-r/--rider` flag enables this.
    * The multiple eEPs are expected in a directory specified using `-rd/--rider-dir`.
    * Each line of the eEP file represents a cell/block of the partition, the member nodes of the cell are separated with space.
    * We compute recursive (sum and means) egonet features for eEP based primitive features.
* The `test/resources` directory has examples of sample input graph and epsilon equitable partitions.
* This code might (potentially) differ from the original paper, please refer to the inline text comments for more details.
* The list of primitive features and their nomenclature is captured from the original code implementation provided by the authors. We are really grateful to them for sharing the code.  
    