# Dataset:   pga2004.dat

# Source: sportsillustrated.cnn.com

# Description: Performance statistics and winnings for 196 PGA participants
# during 2004 season.

import tensorflow as tf
import numpy as np
myarray = np.fromfile('pga2004.dat',dtype=float);
print myarray