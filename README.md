# TFLite + libQnn

This demo calls TFLite using the libQnn delegate, loading it as an external delegate. It defaults to using the HTP backend of the QNN delegate.

It only supports TFLite models which perform image transformations; some examples of such models can be found at https://aihub.qualcomm.com/models. When it invokes the model, this demo loads input images and writes output images to and from the filesystem.

## Building

```shell
$ cmake -B build
$ cmake --build build
```

## Usage

```shell
$ build/tflite_demo <model.tflite> <input.png> <output.png>
```

Supported input formats: JPEG, PNG, TGA, BMP, PSD, GIF, HDR, PIC, and PNM.

Supported output formats: PNG.

## License

This project is licensed under the MIT License.
