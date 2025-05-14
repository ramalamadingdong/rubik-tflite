#include <assert.h>
#include <stdio.h>
#include <time.h> // Add this for timing

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <tensorflow/lite/version.h>
#include <tensorflow/lite/c/c_api.h>
#include <tensorflow/lite/c/c_api_experimental.h>
#include <tensorflow/lite/delegates/external/external_delegate.h>

// Guesses the width, height, and channels of a tensor if it were an image. Returns false on failure.
bool tensor_image_dims(const TfLiteTensor* tensor, int* w, int* h, int* c) {
	int n = TfLiteTensorNumDims(tensor);
	int cursor = 0;

	for (int i = 0; i < n; i++) {
		int dim = TfLiteTensorDim(tensor, i);
		if (dim == 0) return false;
		if (dim == 1) continue;

		switch (cursor++) {
			case 0: if(w) *w = dim; break;
			case 1: if(h) *h = dim; break;
			case 2: if(c) *c = dim; break;
			default: return false;  break;
		}
	}

	// Ensure that we at least have the width and height.
	if (cursor < 2) return false;
	// If we don't have the number of channels, then assume there's only one.
	if (cursor == 2 && c) *c = 1;
	// Ensure we have no more than 4 image channels.
	if (*c > 4) return false;
	// The tensor dimension appears coherent.
	return true;
}

void print_tensor_info(const TfLiteTensor* tensor) {
  	size_t tensor_size = TfLiteTensorByteSize(tensor);

	printf("INFO:   Size: %lu bytes\n", tensor_size);

	int num_dims = TfLiteTensorNumDims(tensor);

	printf("INFO:   Dimension: ");

	for (int i = 0; i < num_dims; i++) printf("%d%s", TfLiteTensorDim(tensor, i), i == num_dims - 1 ? "" : "x");

	printf("\n");

	switch (TfLiteTensorType(tensor)) {
		case kTfLiteFloat16: printf("INFO:   Type: f16\n"); break;
		case kTfLiteFloat32: printf("INFO:   Type: f32\n"); break;
		case kTfLiteUInt8:   printf("INFO:   Type: u8 \n"); break;
		case kTfLiteUInt32:  printf("INFO:   Type: u32\n"); break;
		case kTfLiteInt8:    printf("INFO:   Type: i8 \n"); break;
		case kTfLiteInt32:   printf("INFO:   Type: i32\n"); break;
		default:             printf("INFO:   Type: ???\n"); break;
	}
}

int main(int argc, char** argv) {
	int result = 1;

	printf("INFO: TFLite version: %d.%d.%d\n", TF_MAJOR_VERSION, TF_MINOR_VERSION, TF_PATCH_VERSION);

	if (argc < 4) {
		printf("Usage: %s <model.tflite> <input.jpg> <output.png>\n", argv[0]);
		return 1;
	}

	const char* arg_model = argv[1];
	const char* arg_input = argv[2];
	const char* arg_output = argv[3];

	TfLiteModel* model = NULL;
	TfLiteDelegate* delegate = NULL;
	TfLiteInterpreter* interpreter = NULL;
	unsigned char* image = NULL;

	/* Setup the interpreter. */

	if ((model = TfLiteModelCreateFromFile(arg_model)) == NULL) {
		printf("ERROR: Failed to load model file '%s'\n", arg_model);
		goto defer;
	}

	printf("INFO: Loaded model file '%s'\n", arg_model);

	TfLiteExternalDelegateOptions delegateOpts = TfLiteExternalDelegateOptionsDefault("libQnnTFLiteDelegate.so");
	TfLiteExternalDelegateOptionsInsert(&delegateOpts, "backend_type", "htp");

	if ((delegate = TfLiteExternalDelegateCreate(&delegateOpts)) == NULL) {
		printf("ERROR: Failed to create delegate\n");
		goto defer;
	}

	printf("INFO: Loaded external delegate\n");

	TfLiteInterpreterOptions* interpreterOpts = TfLiteInterpreterOptionsCreate();

	TfLiteInterpreterOptionsAddDelegate(interpreterOpts, delegate);

	if ((interpreter = TfLiteInterpreterCreate(model, interpreterOpts)) == NULL) {
		printf("ERROR: Failed to create interpreter\n");
		TfLiteInterpreterOptionsDelete(interpreterOpts);
		goto defer;
	}

	TfLiteInterpreterOptionsDelete(interpreterOpts);

	if (TfLiteInterpreterModifyGraphWithDelegate(interpreter, delegate) != kTfLiteOk) {
		printf("ERROR: Failed to modify graph with delegate\n");
		goto defer;
	}

	if (TfLiteInterpreterAllocateTensors(interpreter) != kTfLiteOk) {
		printf("ERROR: Failed to allocate tensors\n");
		goto defer;
	}

	/* Get the tensor info. */

	// Input:

	unsigned int n_tensors_in = TfLiteInterpreterGetInputTensorCount(interpreter);

	if (n_tensors_in != 1) {
		printf("ERROR: expected only 1 input tensor, got %d\n", n_tensors_in);
		goto defer;
	}

	TfLiteTensor* input = TfLiteInterpreterGetInputTensor(interpreter, 0);

	printf("INFO: Input tensor:\n");
	print_tensor_info(input);

	int in_w, in_h, in_c;

	if (!tensor_image_dims(input, &in_w, &in_h, &in_c)) {
		printf("ERROR: failed to extract image dimensions of input tensor.\n");
		goto defer;
	}

	printf("INFO: input tensor image dimensions: %dx%d, with %d channels\n", in_w, in_h, in_c);

	// Output:

	unsigned int n_tensors_out = TfLiteInterpreterGetOutputTensorCount(interpreter);

	if (n_tensors_out != 1) {
		printf("ERROR: expected only 1 output tensor, got %d\n", n_tensors_out);
		goto defer;
	}

	const TfLiteTensor* output = TfLiteInterpreterGetOutputTensor(interpreter, 0);

	printf("INFO: Output tensor:\n");
	print_tensor_info(output);

	int out_w, out_h, out_c;

	if (!tensor_image_dims(output, &out_w, &out_h, &out_c)) {
		printf("ERROR: failed to extract image dimensions of output tensor.\n");
		goto defer;
	}

	printf("INFO: output tensor image dimensions: %dx%d, with %d channels\n", out_w, out_h, out_c);

	/* Load the input. */

	int w, h;
	image = stbi_load(arg_input, &w, &h, NULL, in_c);

	if (image == NULL) {
		printf("ERROR: failed to open image '%s'\n", arg_input);
		goto defer;
	}

	if (in_w != w || in_h != h) {
		printf("ERROR: input image %s does not match dimension of input tensor: expected %dx%d, got %dx%d\n", arg_input, in_w, in_h, w, h);
		goto defer;
	}

	printf("INFO: loaded input image '%s'\n", arg_input);

	// If the dimension and channels match, the byte size of the input and the tensor should be identical.
	size_t image_size = w * h * in_c;
	size_t tensor_size = TfLiteTensorByteSize(input);
	assert(tensor_size == image_size);

	// Write the input data into the tensor.
	memcpy(TfLiteTensorData(input), image, tensor_size);
	// Free the input, now that we've copied it over.
	stbi_image_free(image);
	image = NULL;

	/* Execute the model. */

	printf("INFO: Invoking interpreter...\n");

	struct timespec start, end;
	clock_gettime(CLOCK_MONOTONIC, &start); // Start timing

	if (TfLiteInterpreterInvoke(interpreter) != 0) {
		printf("ERROR: Model execution failed\n");
		goto defer;
	}

	clock_gettime(CLOCK_MONOTONIC, &end); // End timing

	// Calculate elapsed time in milliseconds
	double elapsed_time = (end.tv_sec - start.tv_sec) * 1000.0 + 
						  (end.tv_nsec - start.tv_nsec) / 1000000.0;

	printf("INFO: Model execution time: %.2f ms\n", elapsed_time);

	/* Save the output. */

	int res = stbi_write_png(arg_output, out_w, out_h, out_c, TfLiteTensorData(output), out_w * out_c);

	if (res == 0) {
		printf("ERROR: Failed to write output to '%s'\n", arg_output);
		goto defer;
	}

	printf("INFO: wrote output image to '%s'\n", arg_output);

	/* Free our resources. */

	result = 0;
defer:
	if (delegate != NULL)    TfLiteExternalDelegateDelete(delegate);
	if (model != NULL)       TfLiteModelDelete(model);
	if (interpreter != NULL) TfLiteInterpreterDelete(interpreter);
	if (image != NULL)       stbi_image_free(image);

	return result;
}
