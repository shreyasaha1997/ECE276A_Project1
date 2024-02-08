All the data is preprocessed and stored in https://drive.google.com/file/d/1UqiGLL_FXXKng7uj4LWk4j_hr_7lNUDX/view?usp=sharing. Please download and unzip the file. The outputs will be stored inside the data/<dataset_number>/logs folder. 

These are the commands to run the code. Added for every dataset for convenience - 

```
python3 main.py --dataset 1 --epochs 600 --orientation_tracking --panaroma_creation
python3 main.py --dataset 2 --epochs 600 --orientation_tracking --panaroma_creation
python3 main.py --dataset 3 --epochs 600 --orientation_tracking 
python3 main.py --dataset 4 --epochs 600 --orientation_tracking 
python3 main.py --dataset 5 --epochs 600 --orientation_tracking 
python3 main.py --dataset 6 --epochs 600 --orientation_tracking 
python3 main.py --dataset 7 --epochs 600 --orientation_tracking 
python3 main.py --dataset 8 --epochs 600 --orientation_tracking --panaroma_creation
python3 main.py --dataset 9 --epochs 600 --orientation_tracking --panaroma_creation
python3 main.py --dataset 10 --epochs 600 --orientation_tracking --panaroma_creation
python3 main.py --dataset 11 --epochs 600 --orientation_tracking --panaroma_creation
```

As further explained in the project report, there are three versions of panorama inside the logs folder - 

- panorama1_1.png refers to the default values of theta and phi used. 
- panorama1_0.01.png refers to the case when the theta values fluctuate a lot, even though the roll and pitch is almost zero, and hence the theta fluctuations are minimised by 0.01.
- panorama0.01_1.png refers to the case where the phi fluctuations are minimised as yaw is significantly less.
