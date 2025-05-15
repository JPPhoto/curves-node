# curves-node

Provides an InvokeAI node for applying tone mapping curves and dithering effects to images.

This node allows for fine-tuned tonal remapping using interpolation methods such as Catmull-Rom, linear, or
cubic splines. In addition, several dithering techniques—including blue noise, Floyd–Steinberg, and ordered
dithering—can be applied to enhance image stylization or improve the appearance of remapped gradients.

The node accepts a comma-separated string of input-output mappings to define the tone curve, like
`0:0,64:48,128:128,200:240,255:255`.

## Supported interpolation methods:

* **Catmull-Rom spline**
* **Linear**
* **Cubic spline (SciPy)**

## Supported dithering methods:

* **None**
* **Blue Noise (shuffled Bayer approximation)**
* **Floyd–Steinberg (very slow)**
* **Ordered (Bayer matrix)**

## Note:

This node supports all common image modes (RGB, RGBA, L, LA). Alpha channels are preserved and recomposited after
interpolation and dithering.
