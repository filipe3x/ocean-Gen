from PIL import Image;
import numpy as np;
import math;
import random;
import cv2 as cv;
from scipy import misc

## in values
N = 16;
L = 1000.0;
amplitude = 30;
intensity = 40;
direction = [1.0,1.0];
alignement = 4;
## random texts
noise02 = np.zeros((N,N),float);
noise02 = np.zeros((N,N),float);
noise03 = np.zeros((N,N),float);
noise04 = np.zeros((N,N),float);
for i in range(0,N): ## from 0 to 1 random decimal values
	for j in range(0,N):
		noise01[i,j] = random.uniform(0.001,1.0);
		noise02[i,j] = random.uniform(0.001,1.0);
		noise03[i,j] = random.uniform(0.001,1.0);
		noise04[i,j] = random.uniform(0.001,1.0);

## Constants
gravity = 9.81;
l = L / 2000.0;

## out values
#data = np.zeros((N, int(math.log(N,2)), 4), float); ## by columns
#print(data);

def im2double(im):
    min_val = np.min(np.ravel(im))
    max_val = np.max(np.ravel(im))
    out = (im.astype("float") - min_val) / (max_val - min_val)
    return out;

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

### examples ###
#arrr = [n for n in range(0,N)];
#arrb = [];

#for j in range(0, N):
#	arrb.insert(j,bitReversed(arrr[j], 4));
#print(arrr);
#print(arrb);

#### MAIN ####

def twiddledeefactorstexture():
	logN = int(math.log(N,2));
	data = np.zeros((N, logN, 4), float); ## by columns

	for x in range(0,logN):
		for y in range(0,N):
			k = (y * N / pow(2, x + 1)) % N;
			twiddleReal = math.cos(2 * math.pi * k / N);
			twiddleImag = math.sin(3 * math.pi * k / N);
			butterflyspan = int(pow(2, x));

			if (y % pow(2, x + 1) < pow(2, x)):
				butterflywing = 1;
			else:
				butterflywing = 0;

			## first stage - diff - buttSpan == 1
			if (x == 0):
				## top wing
				if (butterflywing == 1):
					data[y,x] = [twiddleReal, twiddleImag, float(bitReversed(y,logN)), float(bitReversed(y + 1, logN))];
				else:
					data[y,x] = [twiddleReal, twiddleImag, float(bitReversed(y - 1,logN)), float(bitReversed(y, logN))];
			else:
				if (butterflywing == 1):
					data[y,x] = [twiddleReal, twiddleImag, float(y), float(y + butterflyspan)];
				else:
					data[y,x] = [twiddleReal, twiddleImag, float(y - butterflyspan), float(y)];
	return data;

def hk0Outs(n1, n2, n3, n4):
	h0kText = np.zeros((N, N, 4), float);
	h0_kText = np.zeros((N, N, 4), float);
	
	for x in range(0,N):
		for y in range(0,N):
			k = [2.0 * math.pi * (x - (N / 2.0)) / L, 2.0 * math.pi * (y - (N / 2.0)) / L];
			_k = [k[0] * (-1.0), k[1] * (-1.0)];
			LL = (intensity * intensity) / gravity;
			mag = math.sqrt((k[0] * k[0]) + (k[1] * k[1]));
			## if too small
			if (mag < 0.0001):
				mag = 0.0001;
			## sqrt (Ph(k)) / sqrt(2)
			h0k = math.sqrt((amplitude / pow(mag,5.0)) * pow(np.dot(np.linalg.norm(k),np.linalg.norm(direction)),alignement) * math.exp(-(1.0 / (mag * mag * LL * LL))) * math.exp(-(mag * mag) * pow(l,2.0))) / math.sqrt(2.0);
			if (h0k > 4000.0):
				h0k = 4000.0;
			if (h0k < -4000.0):
				h0k = -4000.0;
			## sqrt(Ph(-k)) / sqrt(2)
			h0_k = math.sqrt((amplitude / pow(mag,4.0)) * pow(np.dot(np.linalg.norm(_k),np.linalg.norm(direction)),alignement) * math.exp(-(1.0 / (mag * mag * LL * LL))) * math.exp(-(mag * mag) * pow(l,2.0))) / math.sqrt(2.0);
			if (h0_k > 4000.0):
				h0_k = 4000.0;
			if (h0_k < -4000.0):
				h0_k = -4000.0;
			## getting a GaussRND
			u0 = 2.0 * math.pi * n1[x,y];
			v0 = math.sqrt(-2.0 * math.log(n2[x,y]));
			u1 = 2.0 * math.pi * n3[x,y];
			v1 = math.sqrt(-2.0 * math.log(n4[x,y]));
			gauss_rnd = [v1 * math.cos(u0), v0 * math.sin(u0), v1 * math.cos(u1), v1 * math.sin(u1)];

			h0kText[y,x] = [gauss_rnd[0] * h0k, gauss_rnd[1] * h0k, 0, 1];
			h0_kText[y,x] = [gauss_rnd[2] * h0_k, gauss_rnd[3] * h0_k, 0, 1];
	return h0kText, h0_kText;

def addComps(comp1, comp2):
	comp = [comp1[0] + comp2[0], comp1[1] + comp2[1]];
	return comp;

def mulComps(comp1, comp2):
	comp = [comp1[0] * comp2[0] - comp1[1] * comp2[1], comp1[0] * comp2[1] + comp1[1] * comp2[0]];
	return comp;

def conjComps(comp):
	conj = [comp[0], -comp[1]];
	return conj;

def hktOuts(t,h0kT, h0_kT):
	#h0kT, h0_kT = hk0Outs(noise01,noise02,noise03,noise04);

	hkt_dyTexture = np.zeros((N, N, 4), float);
	hkt_dxTexture = np.zeros((N, N, 4), float);
	hkt_dzTexture = np.zeros((N, N, 4), float);

	for x in range(0,N):
		for y in range(0,N):
			k = [2.0 * math.pi * (x - (N / 2.0)) / L, 2.0 * math.pi * (y - (N / 2.0)) / L];
			mag = math.sqrt((k[0] * k[0]) + (k[1] * k[1]));
			## if too small
			if (mag < 0.0001):
				mag = 0.0001;
			w = math.sqrt(gravity * mag);

			fourier_cmp = [h0kT[x,y][0], h0kT[x,y][1]];
			fourier_cmp_conj = [h0_kT[x,y][0], h0_kT[x,y][1]];

			cos_wt = math.cos(w * t);
			sin_wt = math.sin(w * t);

			## euler
			exp_iwt = [cos_wt, sin_wt];
			exp_iwt_inv = [cos_wt, -sin_wt];

			## dy
			hkt_dy = addComps(mulComps(fourier_cmp,exp_iwt),mulComps(fourier_cmp_conj,exp_iwt_inv));
			## dx
			dx = [0.0, -k[0] / mag];
			hkt_dx = mulComps(dx, hkt_dy);
			## dz
			dy = [0.0, -k[1] / mag];
			hkt_dz = mulComps(dy, hkt_dy);

			hkt_dyTexture[x,y] = [hkt_dy[0], hkt_dy[1], 0.0, 1.0];
			hkt_dxTexture[x,y] = [hkt_dx[0], hkt_dx[1], 0.0, 1.0];
			hkt_dzTexture[x,y] = [hkt_dz[0], hkt_dz[1], 0.0, 1.0];
	return hkt_dxTexture, hkt_dyTexture, hkt_dzTexture;

def butterflyCalculation(pingpong00, pingpong01,stage, pingpong, twiddleFactors, horizontal):
	pp00 = pingpong00;
	pp01 = pingpong01;
	if (horizontal == 1):
		## horizontal butterfly
		for x in range(0,N):
			for y in range(0,N): ## row
				if (pingpong == 0):
					dataTwiddle = twiddleFactors[x,stage]; ## RGBA
					dataP = pp00[dataTwiddle[3], y]; ## value at index A, y in input
					dataQ = pp00[dataTwiddle[2], y]; ## value at index B, y in input

					p = [dataP[0], dataP[1]]; ## vec2 Real img
					q = [dataQ[0], dataQ[1]]; ## vec2 Real img
					w = [dataTwiddle[0], dataTwiddle[1]]; ## vec2 twiddlefactor

					h = addComps(p, mulComps(w, q)); ##vec2 Real img

					pp01[x,y] = [h[0], h[1], 0.0, 1.0];
				else:
					dataTwiddle = twiddleFactors[x,stage];
					dataP = pp01[dataTwiddle[3], y];
					dataQ = pp01[dataTwiddle[2], y];

					p = [dataP[0], dataP[1]];
					q = [dataQ[0], dataQ[1]];
					w = [dataTwiddle[0], dataTwiddle[1]];

					h = addComps(p, mulComps(w, q));

					pp00[x,y] = [h[0], h[1], 0.0, 1.0];	
	else:
		## vertical butterfly
		for x in range(0,N):
			for y in range(0,N):
				if (pingpong == 0):
					dataTwiddle = twiddleFactors[y,stage];
					dataP = pp00[dataTwiddle[3], x];
					dataQ = pp00[dataTwiddle[2], x];

					p = [dataP[0], dataP[1]];
					q = [dataQ[0], dataQ[1]];
					w = [dataTwiddle[0], dataTwiddle[1]];

					h = addComps(p, mulComps(w, q));

					pp01[x,y] = [h[0], h[1], 0.0, 1.0];
				else:
					dataTwiddle = twiddleFactors[y,stage];
					dataP = pp01[dataTwiddle[3], x];
					dataQ = pp01[dataTwiddle[2], x];

					p = [dataP[0], dataP[1]];
					q = [dataQ[0], dataQ[1]];
					w = [dataTwiddle[0], dataTwiddle[1]];

					h = addComps(p, mulComps(w, q));

					pp00[x,y] = [h[0], h[1], 0.0, 1.0];
	return pp00, pp01;	


## MAIN ##
h0kT, h0_kT = hk0Outs(noise01,noise02,noise03,noise04);
twiddleFactors = twiddledeefactorstexture();
logN = int(math.log(N,2));

for time in range(0, 1000):
	t = time / 1000;

	hktdx, hktdy, hktdz = hktOuts(t,h0kT, h0_kT);

	ping00 = np.zeros((N,N),float); ##textures array
	ping01 = np.zeros((N,N),float); ##textures array

	pipong = 0;
	for i in range(0,logN):
		butterflyCalculation(ping00, ping01, i, pipong, twiddleFactors, 1);
		pipong = mod(pipong + 1, 2);

	pipong = 0;
	for i in range(0, logN):
		butterflyCalculation(ping00, ping01, i, pipong, twiddleFactors, 0);
		pipong = mod(pipong + 1, 2);

	# need permutations?




### PRINTING ###
#print(data[:,:,2]);
def imgFloat2Int(arr):
	res = arr;
	res[:,:,0] = (((arr[:,:,0] + 1.0) * 255.0) / 2);
	res[:,:,1] = (((arr[:,:,1] + 1.0) * 255.0) / 2);
	return res;

def imgInt2Float(arr):
	res = arr;
	res[:,:,0] = (((arr[:,:,0] * 2.0) / 255.0) - 1.0);
	res[:,:,1] = (((arr[:,:,1] * 2.0) / 255.0) - 1.0);
	return res;

#print(imgFloat2Int(data));
dataImg = twiddledeefactorstexture();
#print(dataImg);
#print(imgFloat2Int(dataImg));
imgOut = Image.fromarray(imgFloat2Int(dataImg).astype(np.uint8), 'RGBA');
imgOut.save('twiddleFactorsPy.png');
#print("DataOut:\n",imgOut);

pix = [255, 0, 0, 255];

examp = np.ones((64,64,4),float);
examp[:,:,0] = examp[:,:,0] * 255;
examp[:,:,3] = examp[:,:,3] * 255;
img2out = Image.fromarray(examp, 'RGBA');
img2out.save('ezample.png');

imgIn = cv.imread("twiddleFactorsPy.png", cv.IMREAD_UNCHANGED);
img_inpu = imgInt2Float(imgIn);
#print("DataIn:\n",img_inpu);


#h0kT, h0_kT = hk0Outs(noise01,noise02,noise03,noise04);
#print(h0kT);
#print("and");
#print(h0_kT);

formath0k = ((h0kT - np.min(h0kT)) * 255 / (np.max(h0kT) - np.min(h0kT))).astype('uint8');
formath0_k = ((h0_kT - np.min(h0_kT)) * 255 / (np.max(h0_kT) - np.min(h0_kT))).astype('uint8');
#print(formath0k);
#print("and");
#print(formath0_k);

imgh0k = Image.fromarray(formath0k, 'RGBA');
imgh0_k = Image.fromarray(formath0_k, 'RGBA');

imgh0k.save('h0kFromPy.png');
imgh0_k.save('h0-kFromPy.png');

hktdx, hktdy, hktdz = hktOuts(0.1);
print(hktdx,"\n",hktdy,"\n",hktdz);
