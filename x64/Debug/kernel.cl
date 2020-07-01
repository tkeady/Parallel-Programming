//kernels file

//serial scan using atomic additions
kernel void scan_add_atomic(global int* A, global int* B) 
{
	int id = get_global_id(0);
	int N = get_global_size(0);
	for (int i = id + 1; i < N; i++)

		//atomic addition
		atomic_add(&B[i], A[id]);
}

//bin value of A goes into histogram H
kernel void histogram(global const uchar* A, global int* H)
{
	int id = get_global_id(0);

	// atomic increment of bin index
	atomic_inc(&H[A[id]]);
}

// additional buffer B to avoid data overwrite
kernel void scan_hs(global int* input, global int* output) 
{
	int N = get_global_size(0);
	int id = get_global_id(0);
	printf("%i, ", id);

	//A, B, C from example changed to temp, input, output
	global int* temp;
	for (int stride=1;stride<N; stride*=2) {
		output[id] = input[id];
		while (id >= stride) {
			output[id] += input[id - stride];
		}

		//sync the step
		barrier(CLK_GLOBAL_MEM_FENCE);

		//swap between steps
		temp = input;
		input = output;
		output = temp;
	}
}

//a double-buffered version of the Hillis-Steele inclusive scan
//requires two additional input arguments which correspond to two local buffers
kernel void scan_add(__global const int* A, global int* B, local int* scratch_1, local int* scratch_2) 
	{
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);
	local int* scratch_3;//used for buffer swap

	//cache all N values from global memory to local memory
	scratch_1[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory

	for (int i = 1; i < N; i *= 2) 
	{
		if (lid >= i)
			scratch_2[lid] = scratch_1[lid] + scratch_1[lid - i];
		else
			scratch_2[lid] = scratch_1[lid];

		barrier(CLK_LOCAL_MEM_FENCE);

		//buffer swap
		scratch_3 = scratch_2;
		scratch_2 = scratch_1;
		scratch_1 = scratch_3;
	}

	//copy the cache to output array
	B[id] = scratch_1[lid];
}

//normalise the cumulative histogram
kernel void normalisationBins(global const int* A, global int* B) 
{
	int id = get_global_id(0);

	//calculation of the bin values. 255 is the number of bins. 699392 is number of pixels
	float calculation = (float)255/(float)699392;

	// normalise the value
	B[id] = A[id] * calculation;
}

//store the intensity value into cimg
kernel void lut(global  uchar* A, global uchar* B, global int* C) 
{
	int id = get_global_id(0);
	B[id] = A[id];
	B[id] = C[A[id]];
}