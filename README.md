# 2D-DIC-Dataset-Generation-using-Interpolation
The algorithm in this resiponstory generate the speckle image pair using interpolation and the given displacement field as label. 

使用插值方法，通过给定的位移场，以及一帧给定的图像，生成以给定位移场为位移标签的变形前后的图像对。

## Work Flow 方法流程
Acoording to the relationship of the gray value:

按照变形前后灰度关系:

$I_d(u+U(u,v), v+V(u,v))=I_r(u, v)$ 

Using the given image as the deformed image $I_d$ , and the displacement field $U, V$ as the displacement label, the reference image $I_r$ can be generated using interpolation (Bicibic, B-spline or bilinear, provided in this responsitory), accelaterated by pytorch.

使用给定图像作为变形后图像 $I_d$ ，使用给定位移场$U, V$作为位移标签，参考图像可以通过上式使用插值方法生成（这里提供了双三次、B-样条、双线性插值方法），插值方法通过pytorch使用GPU加速。

## Requirement 运行环境
- python
- pytorch (cpu or cuda version)

## Use 使用

&ensp; |_```displacement_generation.py```  is used to generate random displacement.

&ensp; |_```generate_dataset.py```         has functions to generate the dataset.
  
&ensp; |_```interpolation.py```            performs interpolation.




