#include <stdio.h>
#include <math.h>
#include <time.h>

//A TRIER
#define EPS 0.0000001f
#define r 0.1f
#define N 3			// Size of the problem's matrix
#define NB 4 		// Number of blocks
#define NTPB 3		// Number of threads per block
#define MIN 1
#define MAX 10

void printVect(float *v, int n){
	for(int i = 0; i < n; i++){
		printf("%f ", v[i]);
	}
	printf("\n\n");
}

// Used 
float randFloat(){
	return ((MAX - MIN) * ((float)rand() / RAND_MAX)) + MIN;
}

// Function that catches the error 
void testCUDA(cudaError_t error, const char *file, int line)  {
	if (error != cudaSuccess) {
	printf("There is an error in file %s at line %d\n", file, line);
	printf("%s\n", cudaGetErrorName(error));
	printf("%s\n", cudaGetErrorString(error));
	exit(EXIT_FAILURE);
	} 
}
#define testCUDA(error) (testCUDA(error, __FILE__ , __LINE__))

void iniTest(float* a,float* b, float* c, float* y, int n){
	// Initialisation d'un système de taille n pour les tests
	for(int i = 0; i < n; i++){
		// POur éviter l'overflow
		a[i] = randFloat();
		c[i] = randFloat();
		b[i] = randFloat();
		y[i] = randFloat();
	}
}

// CREDITS TO Lokman A. Abbas-Turki FOR THIS CODE
__device__ void PCR_d(float* sa, float* sd, float* sc, float* sy, int* sl, int n) {
	int i, lL, d, tL, tR;
	float aL, dL, cL, yL;
	float aLp, dLp, cLp, yLp;

	d = (n / 2 + (n % 2)) * (threadIdx.x % 2) + (int)threadIdx.x / 2;

	tL = threadIdx.x - 1;
	if (tL < 0) tL = 0;
	tR = threadIdx.x + 1;
	if (tR >= n) tR = 0;

	for (i = 0; i < (int)(logf((float)n) / logf(2.0f)) + 1; i++) {
		lL = (int)sl[threadIdx.x];

		aL = sa[threadIdx.x];
		dL = sd[threadIdx.x];
		cL = sc[threadIdx.x];
		yL = sy[threadIdx.x];

		dLp = sd[tL];
		cLp = sc[tL];

		if (fabsf(aL) > EPS) {
			aLp = sa[tL];
			yLp = sy[tL];
			dL -= aL * cL / dLp;
			yL -= aL * yLp / dLp;
			aL = -aL * aLp / dLp;
			cL = -cLp * cL / dLp;
		}

		cLp = sc[tR];
		if (fabsf(cLp) > EPS) {
			aLp = sa[tR];
			dLp = sd[tR];
			yLp = sy[tR];
			dL -= cLp * aLp / dLp;
			yL -= cLp * yLp / dLp;
		}
		__syncthreads();

		if (i < (int)(logf((float)n) / logf(2.0f))) {
			sa[d] = aL;
			sd[d] = dL;
			sc[d] = cL;
			sy[d] = yL;
			sl[d] = (int)lL;
			__syncthreads();
		}
	}

	sy[(int)sl[threadIdx.x]] = yL / dL;
}

__global__ void PCR(float* sa, float* sd, float* sc, float* sy, int n){
	
	/* Allocating space in the shared memomry because accesses to the 
	 * shared memory are faster than  accesses to the global memory */
	extern __shared__ float tab[];
	float* ssa = tab;
	float* ssd = ssa + NTPB;
	float* ssc = ssd + NTPB;
	float* ssy = ssc + NTPB;
	int* ssl = (int*)(ssy + NTPB);

	/* Actually copying the values to the shared memory */
	for (int i = 0; i < n; i++){	
		ssa[threadIdx.x] = sa[threadIdx.x + blockIdx.x*blockDim.x];
		ssd[threadIdx.x] = sd[threadIdx.x + blockIdx.x*blockDim.x];
		ssc[threadIdx.x] = sc[threadIdx.x + blockIdx.x*blockDim.x];
		ssy[threadIdx.x] = sy[threadIdx.x + blockIdx.x*blockDim.x];
		ssl[threadIdx.x] = threadIdx.x;
	}
	__syncthreads();
	PCR_d(ssa, ssd, ssc, ssy, ssl, n);

	/* Here we need to copy the result in global memory so that 
	 * the host can access it later on outside of the kernel
	 * (reminder : shared memory doesn't exist outside of a block)*/
	for(int i = 0; i < n; i++){
		sy[threadIdx.x + blockIdx.x*blockDim.x] = ssy[threadIdx.x];
	}
	__syncthreads();
}

/*
*/
__global__ void Thomas(float* aGPU, float* bGPU, float* cGPU, float* yGPU, float* zGPU, int n){
	int idx = threadIdx.x + blockIdx.x*blockDim.x;
	//Forward step
	cGPU[(n-1)*idx] = cGPU[(n-1)*idx]/bGPU[n*idx];
	yGPU[n*idx] = yGPU[n*idx]/bGPU[n*idx];

	for(int i = 1; i < n; i++){
		if(i < n-1) cGPU[i + (n-1)*idx] = cGPU[i + (n-1)*idx]/(bGPU[i + n*idx] - aGPU[i - 1 + (n-1)*idx]*cGPU[i-1 + (n-1)*idx]);
		yGPU[i + n*idx] = (yGPU[i + n*idx] - aGPU[i - 1 + (n-1)*idx]*yGPU[i - 1 + n*idx])/(bGPU[i + n*idx] - aGPU[i - 1+ (n-1)*idx]*cGPU[i - 1 + (n-1)*idx]);
	}

	//Backward step
	zGPU[n-1 + n*idx] = yGPU[n-1 + n*idx];
	for(int i = n-2; i >= 0; i--){
		zGPU[i + n*idx] = yGPU[i + n*idx] - cGPU[i + (n-1)*idx] * zGPU[i+1 + n*idx];
	}
}

__global__ void PDE_1(){}

void Thomas_wrap(float* a, float* b, float* c, float* y, float* z, int n){
	// Déclaration des variables utilisées
	float *aGPU, *bGPU, *cGPU, *yGPU, *zGPU;

	// Allocation des vecteurs dans la mémoire GPU
	testCUDA(cudaMalloc(&aGPU, NB*NTPB*(n-1)*sizeof(float)));
	testCUDA(cudaMalloc(&cGPU, NB*NTPB*(n-1)*sizeof(float)));
	testCUDA(cudaMalloc(&bGPU, NB*NTPB*n*sizeof(float)));
	testCUDA(cudaMalloc(&yGPU, NB*NTPB*n*sizeof(float)));
	testCUDA(cudaMalloc(&zGPU, NB*NTPB*n*sizeof(float)));

	// Copie des données dans les vecteurs sur GPU pour chaque block
	for(int i = 0; i < NB*NTPB; i++){
		testCUDA(cudaMemcpy(aGPU + i*(n-1), &(a[1]), (n-1)*sizeof(float), cudaMemcpyHostToDevice));
		testCUDA(cudaMemcpy(bGPU + i*n, b, n*sizeof(float), cudaMemcpyHostToDevice));
		testCUDA(cudaMemcpy(cGPU + i*(n-1), &(c[1]), (n-1)*sizeof(float), cudaMemcpyHostToDevice));
		testCUDA(cudaMemcpy(yGPU + i*n, y, n*sizeof(float), cudaMemcpyHostToDevice));
	}

	Thomas<<<NB, NTPB>>>(aGPU, bGPU, cGPU, yGPU, zGPU, n);
	cudaDeviceSynchronize();

	for(int i = 0; i < NB*NTPB; i++){
		testCUDA(cudaMemcpy(z, zGPU + i*n, n*sizeof(float), cudaMemcpyDeviceToHost));
		printVect(z, n);
	}

	// Libération des vecteurs GPU
	testCUDA(cudaFree(aGPU));
	testCUDA(cudaFree(bGPU));
	testCUDA(cudaFree(cGPU));
	testCUDA(cudaFree(yGPU));
	testCUDA(cudaFree(zGPU));
}

void PCR_wrap(float* a, float* b, float* c, float* y, int n){
	
	// Déclaration des variables utilisées
	float *aGPU, *bGPU, *cGPU, *yGPU;

	// Allocation des vecteurs dans la mémoire GPU
	testCUDA(cudaMalloc(&aGPU, NB*n*sizeof(float)));  		//sa
	testCUDA(cudaMalloc(&cGPU, NB*n*sizeof(float)));  		//sc
	testCUDA(cudaMalloc(&bGPU, NB*n*sizeof(float)));		//sd
	testCUDA(cudaMalloc(&yGPU, NB*n*sizeof(float)));		//sy (contains the solution after calling PCR_d)

	// Copie des données dans les vecteurs sur GPU pour chaque block
	for(int i = 0; i < NB; i++){
		testCUDA(cudaMemcpy(aGPU + i*n, a, n*sizeof(float), cudaMemcpyHostToDevice));
		testCUDA(cudaMemcpy(bGPU + i*n, b, n*sizeof(float), cudaMemcpyHostToDevice));
		testCUDA(cudaMemcpy(cGPU + i*n, c, n*sizeof(float), cudaMemcpyHostToDevice));
		testCUDA(cudaMemcpy(yGPU + i*n, y, n*sizeof(float), cudaMemcpyHostToDevice));
	}
	

	PCR<<<NB, NTPB, 5*NTPB*sizeof(float)>>>(aGPU, bGPU, cGPU, yGPU, n);
	cudaDeviceSynchronize();

	for(int i = 0; i < NB; i++){
		testCUDA(cudaMemcpy(y, yGPU + i*n, n*sizeof(float), cudaMemcpyDeviceToHost));	// !!! SOLUTION IN yGPU !!!
		printVect(y, n);
	}

	// Libération des vecteurs GPU
	testCUDA(cudaFree(aGPU));
	testCUDA(cudaFree(bGPU));
	testCUDA(cudaFree(cGPU));
	testCUDA(cudaFree(yGPU));
}

void PDE_1_wrap(int M, int P1, int P2){
	/*
	int i;

	// threadIdx.x = le i dans la formule d'induction de Crank-Nicolson
	int u = threadIdx.x + 1;											
	int m = threadIdx.x;
	int d = threadIdx.x - 1;

	//Constants used in the computation
    float sig = sigmin + dsig*blockIdx.x;
	float mu = r - 0.5f*sig*sig;										//CHECKED
	float pu = 0.25f*(sig*sig*dt/(dx*dx) + mu*dt/dx);					//CHECKED
	float pm = 1.0f - 0.5*sig*sig*dt/(dx*dx);							//CHECKED
	float pd = 0.25f*(sig*sig*dt/(dx*dx) - mu*dt/dx);					//CHECKED
	float qu = -0.25f * (sig * sig * dt / (dx * dx) + mu * dt / dx);	//CHECKED
	float qm = 1.0f + 0.5 * sig * sig * dt / (dx * dx);					//CHECKED
	float qd = -0.25f * (sig * sig * dt / (dx * dx) - mu * dt / dx);	//CHECKED
	*/
}

int main(void){
	int n = N;

	/***********************************
	************ QUESTION 1 ************
	************************************/
	float *a, *b, *c, *y, *z;

	// Allocation des vecteurs pour initialisation du système
	a = (float *) malloc(n* sizeof(float) );
	b = (float *) malloc(n * sizeof(float) );
	c = (float *) malloc(n * sizeof(float) );
	y = (float *) malloc(n * sizeof(float) );
	z = (float *) malloc(n * sizeof(float) );

	iniTest(a, b, c, y, n);

	a[0] = 0.f;
	c[0] = 0.f;

	printf("Vecteur a :\n");
	printVect(a, n);
	printf("Vecteur b :\n");
	printVect(b, n);
	printf("Vecteur c :\n");
	printVect(c, n);
	printf("Vecteur y :\n");
	printVect(y, n);

	//Test Thomas
	Thomas_wrap(a, b, c, y, z, n);

	//Test PCR
	PCR_wrap(a, b, c, y, n);

	// Libération des vecteurs sur RAM
	free(a);
	free(b);
	free(c);
	free(y);
	free(z);

	/***********************************
	******** END OF QUESTION 1 *********
	************************************/

	/***********************************
	************ QUESTION 2 ************
	************************************/

	/***********************************
	******** END OF QUESTION 2 *********
	************************************/



	return 0;
}