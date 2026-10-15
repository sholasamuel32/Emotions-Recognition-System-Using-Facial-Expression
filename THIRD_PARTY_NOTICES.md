# Third-party model and detector notices

## FER facial-expression model

MoodSense bundles an ONNX conversion of the quantized TensorFlow Lite
facial-expression model from the open-source `fer` project, version 25.10.3.
The conversion makes local inference portable across macOS, Windows, and Linux
through ONNX Runtime while preserving the original seven-output CNN.

- Project: `https://github.com/justinshenk/fer`
- Copyright: Copyright (c) 2018 Justin Shenk
- Licence: MIT License

The `fer` project credits methods and package structure derived from Iván de
Paz Centeno's MTCNN implementation and Octavio Arriaga's facial-expression
classification project. The bundled CNN recognizes seven facial-expression
labels: angry, disgust, fear, happy, sad, surprise, and neutral.

## OpenCV Haar cascade

The frontal-face Haar cascade is distributed with OpenCV and used to locate a
face before expression classification.

## MIT licence text

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## FER-2013 evaluation data

The executed facial-model notebook evaluates the model on the 3,589-image
FER-2013 `PrivateTest` split. The dataset itself is not bundled in the source
archive; the notebook can download it on demand from the repository documented
in the README. Cite:

Goodfellow, I. J., et al. (2013). Challenges in representation learning: A
report on three machine learning contests. *Neural Information Processing*,
117–124. `https://doi.org/10.1007/978-3-642-42051-1_16`
