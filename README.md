# Magnetic-Substrate-Interface-Forward-and-Inverse
Recovering 3D Basin Basement Relief Using High-precision Magnetic Data Through Random Forest Regression Algorithm.
Here is the directory structure of the folder:
-------------Recovering 3D Basin Basement Relief Using High-precision Magnetic Data Through Random Forest Regression Algorithm-----------
data:
  dataset.zip: Generated stochastic models and corresponding magnetic anomalies.
  field_example.tif:The corrected ones are interpolated by Kriging to obtain the magnetic anomaly raster data.
code:
  Randomized_interface.py：Stochastic generation of magnetic substrate interface models.
  Forward.py：Orthorectification of magnetic anomalies in magnetic substrate interface models by the magnetic interface finite unit method
  Random_forest.py：Perform potential field processing transformations, and train and test models
--------------------------------Requirement------------------------------
	python 3.7
		gdal 3.4.2
		sklearn 1.2.0
		geoist
		matplotlib 3.5.3
		scipy 1.7.3
		numpy 1.21.5
    		imblearn 0.10.1
--------------------------------Requirement------------------------------
----------------------COPYRIGHTS---------------------------------------------
These programs may be freely redistributed under the condition that the 
copyright notices are not removed, and no compensation is received.  Private, 
research, and institutional use is free. You may distribute modified versions 
of this code UNDER THE CONDITION THAT THIS CODE AND ANY MODIFICATIONS MADE TO 
IT IN THE SAME FILE REMAIN UNDER COPYRIGHT OF THE ORIGINAL AUTHOR, BOTH SOURCE 
AND OBJECT CODE ARE MADE FREELY AVAILABLE WITHOUT CHARGE, AND CLEAR NOTICE IS 
GIVEN OF THE MODIFICATIONS. Distribution of this code as part of a commercial 
system is permissible ONLY BY DIRECT ARRANGEMENT WITH THE AUTHOR. If you use 
the code and data in the Magnetic-Substrate-Interface-Forward-and-Inverse folder, 
and especially if you use it to accomplish real work, PLEASE SEND ME AN EMAIL. 

Copyright 2023, Xinjun Zhang
2023-7-21
-----------------------------------------------------------------------------
For possible errors, please contact:

     zhangxinjun@tyut.edu.cn
     Xinjun Zhang
     Taiyuan University of Technology, 
     Shanxi, China.
     2023-7-21
