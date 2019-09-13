import numpy as np
from math import sin, cos, pi, pow, log
import matplotlib.pyplot as plt

N=256;
logN=int(log(N,2));

indices = np.zeros(N, int); ##bit reversed indices

twiddleIndices = np.zeros((logN, N, 4),float); ## N x log N vec4 texture

## the twiddle factors texture stores the computational structure of fft
## each column of the twiddles texture is an fft stage, and the number of lines corresponds to the size of the 1D input frequency array
## we test if we are in STAGE 0 to save bit reversed indices in the B A channels of vec4, otherwise we save the butterfly span
## we also check if we are at the top wing or bottom wing
## the butterfly span doubles in each stage

def mod(a,b):
	return a % b;

## nBits = number bits necessary to store the largest index
## ex: number = 256, nBits = 8
def bitReversed(number, nBits):
	res = 0;
	inp = number;
	for i in range(0,nBits):
		ppw = int(pow(2, nBits - 1 - i));
		d = inp // ppw;
		m = inp % ppw;
		res += d * pow(2, i);
		inp = m;
	return res;

def imageStore(tex):
	fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6),
				 sharex=True, sharey=True);
	ax.cla();
	ax.imshow(np.rot90(np.repeat(tex,N/logN, axis=0)));
	ax.axis('off');
	ax.set_title('twiddles', fontsize=20);

	fig.tight_layout();
	plt.show();

for i in range(0,N):
	indices[i]=bitReversed(i,logN);

def twiddleCalc():
	## for each stage
	for x in range(0,logN): ## N x log N twiddles texture
		for y in range(0,N):
			k = mod(y * (float(N)/pow(2,x+1)) , N); ##float

			twiddle = (cos(2.0 * pi * k/float(N)), sin(2.0 * pi * k/float(N))); ##complex

			## span doubles each stage
			butterflyspan = int(pow(2,x)); ##int

			butterflywing = 0; ##int

			## which wing are we calculating (one instance for every wing or y value. Every stage has N y's values. These can be computed in parallel)
			if( mod(y, pow(2, x + 1)) < pow(2,x) ):
				butterflywing = 1;
			else:
				butterflywing = 0;

			## first stage, bit reversed indices
			if(x == 0):
				## top wing
				if(butterflywing == 1):
					twiddleIndices[x,y] = [twiddle[0], twiddle[1],indices[y],indices[y+1]];
				else: ## bottom wing
					twiddleIndices[x,y] = [twiddle[0], twiddle[1],indices[y-1], indices[y]];
			else: ## second to log(N) stage
				if(butterflywing == 1): ## top
					twiddleIndices[x,y] = [twiddle[0], twiddle[1],y,y + butterflyspan];
				else: ## bottom
					twiddleIndices[x,y] = [twiddle[0], twiddle[1], y - butterflyspan, y];

	imageStore(twiddleIndices);
