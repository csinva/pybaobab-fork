# pybaobabdt Package

The pybaobabdt package provides a python implementation for the visualization of decision trees. 
The technique is based on the scientific paper <a href="https://ieeexplore.ieee.org/document/6102453" target="_blank">BaobabView: Interactive construction and analysis of decision trees</a> developed by the TU/e.
A typical decision tree is visualized using a standard node link diagram:

<img src="https://gitlab.tue.nl/20040367/pybaobab/-/raw/main/images/vehicle_dt.png" width="100%" align="center">

The problem, however, is that information is not easily extracted from this. Which classes are 
easy to separate for example, which classes are similar, where does the main flow of items go etc.
Therefore, we developed techniques to answer these questions with a scalable visualization:

<img src="https://gitlab.tue.nl/20040367/pybaobab/-/raw/main/images/tree.png" width="100%" align="center"/>

Note, this is the same decision tree as the standard node-link diagram above. Each class is represented by a
color, the width of the link represents the number of items flowing from one node to the other.

## Installation

Currently it is supported on Python3.6 onwards. The package can be installed through pip:

```py
$ pip install pybaobabdt
```

### Requirements

This implementation requires <a href="https://graphviz.org/" target="_blank">Graphviz</a>. Graphviz can be installed using:

```py
$ sudo apt-get install graphviz graphviz-dev

``` 

Furthermore it depends on the following python packages (sklearn, numpy, pygraphviz, matplotlib, scipy, pandas), which can be installed through pip:

```py
$ python3 -m pip install -r requirements.txt
```

## Usage
The following example illustrates the ease of use of this package. First build (or load) a decision tree classifier
with sklearn: 

```py
import pybaobabdt
import pandas as pd
from scipy.io import arff
from sklearn.tree import DecisionTreeClassifier

data = arff.loadarff('winequality-red.arff')
df   = pd.DataFrame(data[0])

y = list(df['class'])
features = list(df.columns)
features.remove('class')
X = df.loc[:, features]

clf = DecisionTreeClassifier().fit(X,y)
```
Next, use pybaobab to visualize it:
```py
ax = pybaobabdt.drawTree(clf, size=10, dpi=72, features=features)
```

<img src="https://gitlab.tue.nl/20040367/pybaobab/-/raw/main/images/tree_example.png" width="100%" align="center"/>

You can then save it to a file with for example:
```py
ax.get_figure().savefig('tree.png', format='png', dpi=300, transparent=True)
```

Also, trees from a RandomForest classifier can be visualized and saved to a high-resolution image for inspection:

```py
import pybaobabdt
import pandas as pd
from scipy.io import arff
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

data = arff.loadarff('vehicle.arff')

df = pd.DataFrame(data[0])
y = list(df['class'])
features = list(df.columns)
features.remove('class')
X = df.loc[:, features]

clf = RandomForestClassifier(n_estimators=20, n_jobs=-1, random_state=0)
clf.fit(X, y)
```

Save to image:

```py
size = (15,15)
plt.rcParams['figure.figsize'] = size
fig = plt.figure(figsize=size, dpi=300)

for idx, tree in enumerate(clf.estimators_):
    ax1 = fig.add_subplot(5, 4, idx+1)
    pybaobabdt.drawTree(tree, model=clf, size=15, dpi=300, features=features, ax=ax1)
    
fig.savefig('random-forest.png', format='png', dpi=1200, transparent=True)
```
<img src="https://gitlab.tue.nl/20040367/pybaobab/-/raw/main/images/random-forest.png" width="100%" align="center"/>

### Options

There are several different options that can be used in the drawTree function.

* colormap='plasma' (all matplotlib <a href="https://matplotlib.org/stable/tutorials/colors/colormaps.html" target="_blank">colormaps</a> are supported)
  
You can also define your own colormap, which could be useful to highlight a specific class for example:
```py
#colors = [[1,0,0], [0,1,0], [0,0,1], [1,1,0]]
colors = ["gray", "gray", "purple", "gray"]
colorMap = ListedColormap(colors)

ax = pybaobabdt.drawTree(clf, size=10, dpi=72, features=features, colormap=colorMap)
```

<img src="https://gitlab.tue.nl/20040367/pybaobab/-/raw/main/images/tree_oneclass.png" width="100%" align="center"/>

* maxdepth=3 (set the maximum depth of the tree to render, this can be useful for large trees, to inspect only the top splits.)


<img src="https://gitlab.tue.nl/20040367/pybaobab/-/raw/main/images/tree_maxdepth.png" width="100%" align="center"/>

* ratio=0.5 (sets the aspect ratio of the tree, default = 1)

<img src="https://gitlab.tue.nl/20040367/pybaobab/-/raw/main/images/tree_ratio.png" width="100%" align="center"/>

Note that examples can be found in the 'notebooks' folder containing jupyter notebook examples.

## License
<a href="https://choosealicense.com/licenses/gpl-3.0/#" target="_blank">GNU General Public License v3.0</a>

## Reference

If you need to reference this work please use the following bibtex entry:

```bibtex
@INPROCEEDINGS{Elzen2011,
  author={van den Elzen, Stef and van Wijk, Jarke J.},
  booktitle={2011 IEEE Conference on Visual Analytics Science and Technology (VAST)}, 
  title={BaobabView: Interactive construction and analysis of decision trees}, 
  year={2011},
  pages={151-160},
  doi={10.1109/VAST.2011.6102453}}
```

S. van den Elzen and J. J. van Wijk, "BaobabView: Interactive construction and analysis of decision trees," 2011 IEEE Conference on Visual Analytics Science and Technology (VAST), 2011, pp. 151-160, doi: 10.1109/VAST.2011.6102453.