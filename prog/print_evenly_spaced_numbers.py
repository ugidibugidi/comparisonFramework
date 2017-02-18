import numpy as np;
import sys;

tmp = np.linspace(0.0, np.float(sys.argv[1]), num=int(sys.argv[2]), endpoint=False);
print('\n'.join('{0:.8f}'.format(k).rstrip('0').rstrip('.') for k in tmp))