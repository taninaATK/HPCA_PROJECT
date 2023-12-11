#include <stdio.h>
#include <math.h>
#include <time.h>

// Variables pour toutes les questions
#define NB 4 		// Number of blocks
#define NTPB 3		// Number of threads per block

// Used to generate random matrices for tests
#define MIN 1
#define MAX 10

// Pour la question 2
#define N 10000
#define EPS 0.0000001f
#define r 0.1f
#define B 120
#define M 100
#define K 100
#define P1 10
#define P2 50

typedef float MyTab[NB][NTPB];

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
	ssa[threadIdx.x] = sa[threadIdx.x + blockIdx.x*blockDim.x];
	ssd[threadIdx.x] = sd[threadIdx.x + blockIdx.x*blockDim.x];
	ssc[threadIdx.x] = sc[threadIdx.x + blockIdx.x*blockDim.x];
	ssy[threadIdx.x] = sy[threadIdx.x + blockIdx.x*blockDim.x];
	ssl[threadIdx.x] = threadIdx.x;

	/* making sure the values are all set before beginning the computation*/
	__syncthreads();

	PCR_d(ssa, ssd, ssc, ssy, ssl, n);

	/* Here we need to copy the result in global memory so that 
	 * the host can access it later on outside of the kernel
	 * (reminder : shared memory doesn't exist outside of a block)*/
	sy[threadIdx.x + blockIdx.x*blockDim.x] = ssy[threadIdx.x];
}

void PCR_wrap(float* a, float* b, float* c, float* y, int n){

	// Pour les tests
	float TimeExec;									// GPU timer instructions
	cudaEvent_t start, stop;						// GPU timer instructions
	testCUDA(cudaEventCreate(&start));				// GPU timer instructions
	testCUDA(cudaEventCreate(&stop));				// GPU timer instructions
	testCUDA(cudaEventRecord(start,0));				// GPU timer instructions

	
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

	// Temps
	testCUDA(cudaEventRecord(stop,0));				// GPU timer instructions
	testCUDA(cudaEventSynchronize(stop));			// GPU timer instructions
	testCUDA(cudaEventElapsedTime(&TimeExec, start, stop));							// GPU timer instructions
	testCUDA(cudaEventDestroy(start));				// GPU timer instructions
	testCUDA(cudaEventDestroy(stop));				// GPU timer instructions

	printf("GPU time execution for PCR: %f ms\n", TimeExec);

	/*
	for(int i = 0; i < NB; i++){
		testCUDA(cudaMemcpy(y, yGPU + i*n, n*sizeof(float), cudaMemcpyDeviceToHost));	// !!! SOLUTION IN yGPU !!!
		printVect(y, n);
	}
	*/

	// Libération des vecteurs GPU
	testCUDA(cudaFree(aGPU));
	testCUDA(cudaFree(bGPU));
	testCUDA(cudaFree(cGPU));
	testCUDA(cudaFree(yGPU));
}

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

void Thomas_wrap(float* a, float* b, float* c, float* y, float* z, int n){

	// Pour les tests
	float TimeExec;									// GPU timer instructions
	cudaEvent_t start, stop;						// GPU timer instructions
	testCUDA(cudaEventCreate(&start));				// GPU timer instructions
	testCUDA(cudaEventCreate(&stop));				// GPU timer instructions
	testCUDA(cudaEventRecord(start,0));				// GPU timer instructions

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

	// Temps
	testCUDA(cudaEventRecord(stop,0));				// GPU timer instructions
	testCUDA(cudaEventSynchronize(stop));			// GPU timer instructions
	testCUDA(cudaEventElapsedTime(&TimeExec, start, stop));							// GPU timer instructions
	testCUDA(cudaEventDestroy(start));				// GPU timer instructions
	testCUDA(cudaEventDestroy(stop));				// GPU timer instructions

	printf("GPU time execution for Thomas : %f ms\n", TimeExec);

	/*
	for(int i = 0; i < NB*NTPB; i++){
		testCUDA(cudaMemcpy(z, zGPU + i*n, n*sizeof(float), cudaMemcpyDeviceToHost));
		printVect(z, n);
	}
	*/

	// Libération des vecteurs GPU
	testCUDA(cudaFree(aGPU));
	testCUDA(cudaFree(bGPU));
	testCUDA(cudaFree(cGPU));
	testCUDA(cudaFree(yGPU));
	testCUDA(cudaFree(zGPU));
}


// Solves the PDE on [T(M-1), T]
__global__ void PDE_partial(float dt, float dx, float sig, float pmin, float pmax, MyTab *pt_GPU, int iteration){
	// threadIdx.x = le i dans la formule d'induction de Crank-Nicolson
	int u = threadIdx.x + 1;											
	int m = threadIdx.x;
	int d = threadIdx.x - 1;

	//Constants used in the computation
	float mu = r - 0.5f*sig*sig;										//CHECKED
	float pu = 0.25f*(sig*sig*dt/(dx*dx) + mu*dt/dx);					//CHECKED
	float pm = 1.0f - 0.5*sig*sig*dt/(dx*dx);							//CHECKED
	float pd = 0.25f*(sig*sig*dt/(dx*dx) - mu*dt/dx);					//CHECKED
	float qu = -0.25f * (sig * sig * dt / (dx * dx) + mu * dt / dx);	//CHECKED
	float qm = 1.0f + 0.5 * sig * sig * dt / (dx * dx);					//CHECKED
	float qd = -0.25f * (sig * sig * dt / (dx * dx) - mu * dt / dx);	//CHECKED

	extern __shared__ float A[];

	int i_local;
	int i;

	float* sa = A;
	float* sd = sa + NTPB;
	float* sc = sd + NTPB;
	float* sy = sc + NTPB;
	int* sl = (int*)sy + 2*NTPB;

	sy[m] = pt_GPU[0][blockIdx.x][m];
	__syncthreads();

	// le nombre de pas de T à 0 soit M * dt
	for (i_local = 1; i_local < N/M; i_local++) {
		i = i_local + iteration * N/M;
		if (m == 0) {
			sy[NTPB*(i%2) + m] = pmin;
		}
		else {
			if (m == NTPB - 1) {
				sy[NTPB*(i%2) + m] = pmax;
			}
			else {
				sy[NTPB*(i%2) + m] = pu*sy[NTPB * ((i+1) % 2) + u] + pm*sy[NTPB * ((i+1) % 2) + m] + pd*sy[NTPB * ((i+1) % 2) + d];
			}
		}
		sd[m] = qm;
		if (m < NTPB - 1) {
			sc[m + 1] = qu;
		}
		if (m > 0) {
			sa[m] = qd;
		}
		if (m == 0) {
			sa[0] = 0.f;
			sc[0] = 0.f;
		}
		sl[m] = m;

		__syncthreads();
		PCR_d(sa, sd, sc, sy + NTPB * (i % 2), sl, NTPB);
		__syncthreads();

		if (m == 0) {
			sy[NTPB * (i % 2)] = pmin;
			sy[NTPB * (i % 2) + NTPB - 1] = pmax;
		}
		__syncthreads();
	}
	
	pt_GPU[0][blockIdx.x][m] = sy[m + NTPB*(N % 2)];
}

__global__ void limit_PDE(MyTab *pt_GPU_save, MyTab *pt_GPU, float xmin, float dx, int iteration){										
	int m = threadIdx.x;
	int P1k;

	extern __shared__ float A[];
	float* sy = A;

	sy[m] = pt_GPU_save[0][blockIdx.x][m];
	__syncthreads();

	// Ici, on prend en compte la discontinuité à l'approche de T(M - k)
	int x = xmin + dx*m;
	if(blockIdx.x == P2){
		// Condition équivalente à l'indicatrice
		if(x >= B){
			sy[m] = sy[m];
		} else {
			sy[m] = 0.0f;
		}
	}

	P1k = max(P1 - iteration, 0);

	if(blockIdx.x == (P1k - 1)){
		// Condition équivalente à l'indicatrice
		if(x <= B){
			sy[m] = sy[m];
		} else {
			sy[m] = 0.0f;
		}
	}

	if((blockIdx.x < P2) || (blockIdx.x >= P1k)){
		float tmp1, tmp2;

		// First half of the 3rd limit
		if(x >= B){
			tmp1 = sy[m];
		} else {
			tmp1 = 0.0f;
		}

		// Second half of the 3rd limit
		if(x < B){
			tmp2 = pt_GPU_save[0][blockIdx.x + 1][m];
		} else {
			tmp2 = 0.0f;
		}
		sy[m] = tmp1 + tmp2;
	}

	// Applying the changes to the global matrix
	pt_GPU[0][blockIdx.x][m] = sy[m];
}

void PDE_partial_wrap(float dt, float dx, float sig, float pmin, float pmax, MyTab *pt_CPU){
	float TimeExec;									// GPU timer instructions
	cudaEvent_t start, stop;						// GPU timer instructions
	testCUDA(cudaEventCreate(&start));				// GPU timer instructions
	testCUDA(cudaEventCreate(&stop));				// GPU timer instructions
	testCUDA(cudaEventRecord(start,0));				// GPU timer instructions

	MyTab *GPUTab;
	MyTab *GPUTabBackup;
	testCUDA(cudaMalloc(&GPUTab, sizeof(MyTab)));
	testCUDA(cudaMalloc(&GPUTabBackup, sizeof(MyTab)));
	
	testCUDA(cudaMemcpy(GPUTab, pt_CPU, sizeof(MyTab), cudaMemcpyHostToDevice));

	// Computing the partial PDE
	PDE_partial<<<NB, NTPB, 6*NTPB*sizeof(float)>>>(dt, dx, sig, pmin, pmax, GPUTab, 0);
	cudaDeviceSynchronize();

	float TimeCpy;									// GPU timer instructions
	cudaEvent_t start1, stop1;						// GPU timer instructions
	testCUDA(cudaEventCreate(&start1));				// GPU timer instructions
	testCUDA(cudaEventCreate(&stop1));				// GPU timer instructions
	testCUDA(cudaEventRecord(start1,0));				// GPU timer instruction

	testCUDA(cudaMemcpy(GPUTabBackup, GPUTab, sizeof(MyTab), cudaMemcpyDeviceToDevice));

	testCUDA(cudaEventRecord(stop1,0));						// GPU timer instructions
	testCUDA(cudaEventSynchronize(stop1));					// GPU timer instructions
	testCUDA(cudaEventElapsedTime(&TimeCpy, start1, stop1));	// GPU timer instructions
	testCUDA(cudaEventDestroy(start1));						// GPU timer instructions
	testCUDA(cudaEventDestroy(stop1));						// GPU timer instructions

	// Getting the limit acknowledged
	limit_PDE<<<NB, NTPB, NTPB*sizeof(float)>>>(GPUTabBackup, GPUTab);
	cudaDeviceSynchronize();

	testCUDA(cudaMemcpy(pt_CPU, GPUTab, sizeof(MyTab), cudaMemcpyDeviceToHost));

	testCUDA(cudaEventRecord(stop,0));						// GPU timer instructions
	testCUDA(cudaEventSynchronize(stop));					// GPU timer instructions
	testCUDA(cudaEventElapsedTime(&TimeExec, start, stop));	// GPU timer instructions
	testCUDA(cudaEventDestroy(start));						// GPU timer instructions
	testCUDA(cudaEventDestroy(stop));						// GPU timer instructions

	printf("GPU time execution for partial PDE diffusion: %f ms\n", TimeExec);
	printf("Time spent on the memCpy : %f ms\n", TimeCpy);

	testCUDA(cudaFree(GPUTab));	
	testCUDA(cudaFree(GPUTabBackup));	
}

int main(void){
	int n = NTPB;

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

	// printf("Vecteur a :\n");
	// printVect(a, n);
	// printf("Vecteur b :\n");
	// printVect(b, n);
	// printf("Vecteur c :\n");
	// printVect(c, n);
	// printf("Vecteur y :\n");
	// printVect(y, n);

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

	float T = 1.0f;
	float dt = (float)T/N;
	float xmin = log(K/3);
	float xmax = log(3*K);
	float dx = (xmax-xmin)/NTPB;
	float pmin = 0.0f;
	float pmax = 2.0f * K;
	float sig = 0.2f;

	MyTab *pt_CPU;
	testCUDA(cudaHostAlloc(&pt_CPU, sizeof(MyTab), cudaHostAllocDefault));
	for(int i=0; i<NB; i++){
	   for(int j=0; j<NTPB; j++){
		if(j <= P2 && j >= P1){
	      pt_CPU[0][i][j] = max(0.0, exp(xmin + dx*j) - K);	
		} else {
	      pt_CPU[0][i][j] = 0.0f;
		}
	   }
	}

	testCUDA(cudaFreeHost(pt_CPU));

	/***********************************
	******** END OF QUESTION 2 *********
	************************************/



	return 0;
}