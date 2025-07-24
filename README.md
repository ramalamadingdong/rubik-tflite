# TFLite + libQnn

This demo calls TFLite using the libQnn delegate, loading it as an external delegate. It defaults to using the HTP backend of the QNN delegate.

It only supports TFLite models which perform image transformations; some examples of such models can be found at https://aihub.qualcomm.com/models. When it invokes the model, this demo loads input images and writes output images to and from the filesystem.

## Installing IM SDK 

```shell
sudo apt-get update
sudo add-apt-repository ppa:carmel-team/noble-release -login
sudo apt install gstreamer1.0-qcom-sample-apps qcom-sensors-test-apps tensorflow-qcom
```

## Building
### C:
```shell
cd c
cmake -B build
cmake --build build
```
### Python:
```shell
cd python
pip install -r requirments.txt
```

## Usage

###  C:
```shell
cd c
build/tflite_demo <model.tflite> <input.png> <output.png>
```
### Python:
```shell
cd python
python3 tflite_demo.py <model.tflite> <input.png> <output.png>
```

Supported input formats: JPEG, PNG, TGA, BMP, PSD, GIF, HDR, PIC, and PNM.

Supported output formats: PNG.

## Example

Using a model like https://aihub.qualcomm.com/models/real_esrgan_x4plus and an input image like this one:

`examples/input.png`

![](examples/input.png)

The executable can be invoked with these inputs...

```
build/tflite_demo real_esrgan_x4plus.tflite examples/input.png examples/output.png
```

...and will produce an output like this:

`examples/output.png`

![](examples/output.png)

## License

This project is licensed under the MIT License.

The example image can be found at https://commons.wikimedia.org/wiki/File:PNG_transparency_demonstration_1.png.
