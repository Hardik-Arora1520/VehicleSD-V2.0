VehicleSD This is the project which identify the speed of the vehicle 

Technologies used :
- Python
- opencv
- dlib
- tkinter


##### Vehicle Detection
1. We are using Haarcascade classifier to identify vehicles.
  - Vehicle Tracking - ( assigning IDs to vehicles )
2. We have used corelation tracker from dlib library.
3. Speed Calculation
  - We are calculating the distance moved by the tracked vehicle in a second, in terms of pixels, so we need pixel per meter to calculate the distance travelled in meters.With distance travelled per second in meters, we will get the speed of the vehicle.
