STL and Transferability

1.) Add raw company data to /data/raw/ folder

2.) Run Preprocessing.py

3.) Now you are able to run other .py files in root folder



For SVCCA use files in /SVCCA folder:

1.) Create Base Models by running Pretrain_CNN_models.py

2.) Run Activation_Vector.py to create Activation Vectors for each layer of pretrained base models

3.) Now run SVCCA.py to get Correlation Coefficients of two base model layers