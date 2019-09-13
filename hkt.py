"""
	This is a time dependent frequency components texture, based on the initial components pre-computed in h0(k) and h0(-k)
	In conjunction with the twiddle factors, we can now Generate an FFT based ocean animation
"""
import h0k
import numpy as np
from h0k import N, L, g, imageStore
from math import sin, cos, pi, sqrt

import matplotlib.pyplot as plt

tilde_hkt_dy = np.zeros((N,N,4)); ## ouput height displacement of # vec4
tilde_h0k, tilde_h0minusk = h0k.staticFrequencies(); ## input read-only static textures

## tilde hkt will be the input for the fft and it is time dependent
## hkt needs to be generated for every frame
## but it's computationally efficient because is's only composed of two complex mult and a complex add
## 


def mul(c0,c1):
	return c0[0]*c1[0]-c0[1]*c1[1],c0[0]*c1[1]+c0[1]*c1[0];

def add(c0,c1):
	return c0[0]+c1[0], c0[1]+c1[1];

def conj(c):
	return c[0]-c[1];

def complex(c):
	return c[0],c[1];


def timeDependentFrequencies(t):
	for x in range(0,N):
		for y in range(0,N):

			## Wave vector
			k = (2.0 * pi * x/L, 2.0 * pi * y/L); #vec2
			
			length = sqrt(k[0] * k[0] + k[1] * k[1]); #float
			mag = length;
			## To prevent division by zero
			## in Phillips formula
			if(mag < 0.001): mag = 0.001;

			## float
			w = sqrt(g*mag);

			exp_iwt = (cos(w*t), sin(w*t));
			exp_iwt_inv = (cos(w*t), -sin(w*t));

			## dy ## complex
			h_k_t_dy = add(mul(complex(tilde_h0k[x,y]), exp_iwt), mul(complex(tilde_h0minusk[x,y]), exp_iwt_inv));
			tilde_hkt_dy[x,y] = [h_k_t_dy[0], h_k_t_dy[1], 0, 1];

if __name__ ==  '__main__':
	#fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5),
				 #sharex=True, sharey=True)

	fig, ax = plt.subplots();

	timeDependentFrequencies(0);

	for t in np.arange(0,30,0.05):
		#ax = axes.ravel();
		ax.cla();
		ax.imshow(np.rot90(tilde_hkt_dy));
		timeDependentFrequencies(t);
		#ax[0].axis('off');
		ax.set_title("hkt t={:.2f}".format(t), fontsize=20);
		plt.pause(0.001);
		fig.tight_layout();

	#plt.show();

