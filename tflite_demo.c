#include <stdio.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <tensorflow/lite/version.h>
#include <tensorflow/lite/c/c_api.h>
#include <tensorflow/lite/c/c_api_experimental.h>

#include "TFLiteDelegate/QnnTFLiteDelegate.h"

void print_tensor_info(const TfLiteTensor* tensor) {
  	size_t tensor_size = TfLiteTensorByteSize(tensor);

	printf("INFO: Size: %lu bytes\n", tensor_size);

	int num_dims = TfLiteTensorNumDims(tensor);

	printf("INFO: Dimension: ");

	for (int i = 0; i < num_dims; i++) printf("%d ", TfLiteTensorDim(tensor, i));

	printf("\n");

	switch (TfLiteTensorType(tensor)) {
		case kTfLiteFloat16: printf("INFO: Type: f16\n"); break;
		case kTfLiteFloat32: printf("INFO: Type: f32\n"); break;
		case kTfLiteUInt8:   printf("INFO: Type: u8 \n"); break;
		case kTfLiteUInt32:  printf("INFO: Type: u32\n"); break;
		case kTfLiteInt8:    printf("INFO: Type: i8 \n"); break;
		case kTfLiteInt32:   printf("INFO: Type: i32\n"); break;
		default:             printf("INFO: Type: ???\n"); break;
	}
}

int main(int argc, char** argv) {
	int result = 1;

	printf("INFO: TFLite version: %d.%d.%d\n", TF_MAJOR_VERSION, TF_MINOR_VERSION, TF_PATCH_VERSION);

	if (argc < 4) {
		printf("Usage: %s <model.tflite> <input.jpg> <output.png>\n", argv[0]);
		return 1;
	}

	TfLiteModel* model = NULL;
	TfLiteDelegate* delegate = NULL;
	TfLiteInterpreter* interpreter = NULL;

	if ((model = TfLiteModelCreateFromFile(argv[1])) == NULL) {
		printf("ERROR: Failed to load model file '%s'\n", argv[1]);
		goto defer;
	}

	printf("INFO: Loaded model file '%s'\n", argv[1]);

	TfLiteQnnDelegateOptions delegateOpts = TfLiteQnnDelegateOptionsDefault();
	delegateOpts.backend_type = kHtpBackend;

	if ((delegate = TfLiteQnnDelegateCreate(&delegateOpts)) == NULL) {
		printf("ERROR: Failed to create delegate\n");
		goto defer;
	}

	TfLiteInterpreterOptions* interpreterOpts = TfLiteInterpreterOptionsCreate();

	if ((interpreter = TfLiteInterpreterCreate(model, interpreterOpts)) == NULL) {
		printf("ERROR: Failed to create interpreter\n");
		TfLiteInterpreterOptionsDelete(interpreterOpts);
		goto defer;
	}

	TfLiteInterpreterOptionsAddDelegate(interpreterOpts, delegate);
	TfLiteInterpreterOptionsDelete(interpreterOpts);

	if (TfLiteInterpreterModifyGraphWithDelegate(interpreter, delegate) != kTfLiteOk) {
		printf("ERROR: Failed to modify graph with delegate\n");
		goto defer;
	}

	if (TfLiteInterpreterAllocateTensors(interpreter) != kTfLiteOk) {
		printf("ERROR: Failed to allocate tensors\n");
		goto defer;
	}

	unsigned int n_tensors_in = TfLiteInterpreterGetInputTensorCount(interpreter);

	if (n_tensors_in != 1) {
		printf("Only one input tensor is supported");
		goto defer;
	}

	int x, y;
	unsigned char* image = stbi_load(argv[2], &x, &y, NULL, 3);
	size_t image_size = x * y * 3;

	if (image == NULL) {
		printf("ERROR: Failed to open image '%s'\n", argv[2]);
		goto defer;
	}

	for (unsigned int idx = 0; idx < n_tensors_in; ++idx) {
		TfLiteTensor* tensor = TfLiteInterpreterGetInputTensor(interpreter, idx);

		printf("INFO: Input tensor %d:\n", idx);
		print_tensor_info(tensor);

		size_t tensor_size = TfLiteTensorByteSize(tensor);

		if (tensor_size != image_size) {
			printf("ERROR: tensor and input data size does not match");
			stbi_image_free(image);
			goto defer;
		}

		memcpy(TfLiteTensorData(tensor), image, tensor_size);
	}

	stbi_image_free(image);

	if (TfLiteInterpreterInvoke(interpreter) != 0) {
		printf("Model execution failed\n");
		goto defer;
	}

	unsigned int n_tensors_out = TfLiteInterpreterGetOutputTensorCount(interpreter);

	for (unsigned int idx = 0; idx < n_tensors_out; ++idx) {
		const TfLiteTensor* tensor = TfLiteInterpreterGetOutputTensor(interpreter, idx);

		printf("INFO: Output tensor %d:\n", idx);
		print_tensor_info(tensor);

		size_t tensor_size = TfLiteTensorByteSize(tensor);

		int dim = sqrt(tensor_size / 3);

		int res = stbi_write_png(argv[3], dim, dim, 3, TfLiteTensorData(tensor), dim * 3);

		if (res == 0) {
			printf("Failed to write to '%s'\n", argv[3]);
			goto defer;
		}
	}

	result = 0;
defer:
	if (delegate != NULL)    TfLiteQnnDelegateDelete(delegate);
	if (model != NULL)       TfLiteModelDelete(model);
	if (interpreter != NULL) TfLiteInterpreterDelete(interpreter);

	return result;
}
