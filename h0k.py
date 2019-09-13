"""
	This program will output two static textures h0(k) and h0(-k), the initial frequency components of the ocean displacement texture.
	These textures store the ocean initial state.

	run in console to see the result
	$	python h0k.py 
"""
import numpy as np
from math import sin, cos, pi, sqrt, exp, pow, log
import random

import sys
sys.path.append('/usr/lib/python3.6/site-packages')
from skimage import data
import matplotlib.pyplot as plt

N=256 # fft resolution ## int
L=1000 # horizontal dimension of the ocean ## int
A=20.0 # amplitude ## float
w=(1.0,0.0) # wind direction ## vec2
windspeed=26.0 # m/s ## float

l=L/2000.0 ##? ## float
g=9.81 # gravity constant

## output static textures
tilde_h0k = np.zeros((N,N,4),float); ## matrix NxN of vec4
tilde_h0minusk = np.zeros((N,N,4),float); ## matrix NxN of vec4

## two static noise textures, h0(k) and h0(-k), are needed in the beginning, to generate the time dependent frequency texture h(k,t)
## we need two independent gaussian random numbers matrices for each h0(k), h0(-k), so a total of four random matrices are necessary

# should return a vec4 (a tuple) - Box-Muller-Method in shader
def gaussRND():
	## initialization of noise textures with zeros
	noise04 = np.zeros(1,float);
	noise01 = np.zeros(1,float);
	noise02 = np.zeros(1,float);
	noise03 = np.zeros(1,float);

	## to generate a N by N noise texture - from 0 to 1 random decimal values - minimum value cannot be zero, to avoid division by zero in further calculations
	noise01 = random.uniform(0.001,1.0);
	noise02 = random.uniform(0.001,1.0);
	noise03 = random.uniform(0.001,1.0);
	noise04 = random.uniform(0.001,1.0);

	u0 = 2.0 * pi * noise01;
	v0 = sqrt(-2.0 * log(noise02));
	u1 = 2.0 * pi * noise03;
	v1 = sqrt(-2.0 * log(noise03));

	return v0*cos(u0),v0*sin(u0),v1*cos(u1),v1*sin(u1);

def normalize(k):
	k = np.array([k[0],k[1]])
	knorm = np.linalg.norm(k)
	k = k / knorm ## careful if knorm outputs zero
	return k[0],k[1]

def dot(v1,v2):
	return v1[0]*v2[0] + v1[1]*v2[1];

def neg(v):
	return (-v[0],-v[1]);

def imageStore(tex1,tex2):
	fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5),
				 sharex=True, sharey=True)
	ax = axes.ravel()

	ax[0].imshow(np.rot90(tex1))
	ax[0].axis('off')
	ax[0].set_title('h0k', fontsize=20)

	ax[1].imshow(np.rot90(tex2))
	ax[1].axis('off')
	ax[1].set_title('h0minusk', fontsize=20)

	fig.tight_layout()
	plt.show()

## shader main function
def staticFrequencies():
	for x in range(0,N): ## to N x N texture (absent in the shader)
		for y in range(0,N):
			## Wave vector
			k = (2.0 * pi * x/L, 2.0 * pi * y/L); #vec2

			## L from PhillipSpectrum
			L_ = (windspeed * windspeed)/g; #float

			## We calculate magnitude of k wave vector
			length = sqrt(k[0] * k[0] + k[1] * k[1]); #float
			mag = length;
			## To prevent division by zero
			## in Phillips formula
			if(mag < 0.001): mag = 0.001;
			magSq = mag * mag;

			## sqrt(Ph(k))/sqrt(2) ##float
			h0k = sqrt((A/(magSq*magSq)) * pow(dot(normalize(k), normalize(w)), 2.0) * exp(-(1.0/(magSq * L_ * L_))) * exp(-magSq*pow(l,2.0)))/ sqrt(2.0);
			h0k = np.clip(h0k,-4000,4000);

			## sqrt(Ph(-k))/sqrt(2) ##float
			h0minusk = sqrt((A/(magSq*magSq)) * pow(dot(normalize(neg(k)), normalize(w)), 2.0) * exp(-(1.0/(magSq * L_ * L_))) * exp(-magSq*pow(l,2.0)))/ sqrt(2.0);
			h0minusk = np.clip(h0minusk,-4000,4000);

			gauss_random = gaussRND(); #vec4

			## first two components of vec4 of random number
			tilde_h0k[x,y] = [gauss_random[0]*h0k, gauss_random[1]*h0k, 0,1] #r,g,b,a
			## last two components
			tilde_h0minusk[x,y] = [gauss_random[2]*h0minusk, gauss_random[3]*h0minusk, 0,1] #r,g,b,a
	return tilde_h0k, tilde_h0minusk

if __name__ ==  '__main__':
	staticFrequencies();
	imageStore( tilde_h0k, tilde_h0minusk ); #pseudo shader function call to display texture

