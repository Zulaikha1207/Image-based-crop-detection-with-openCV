This repo explores classical computer vision tehcniques to explore how they can be applied on agricultural data. The main task of this project is to detect and count the number of crops in an image. 

Here is a general approach I followed to achieve the tasks using OpenCV and Python:

- Preprocess
Convert the input image to grayscale, apply Gaussian blur, and performs binary thresholding + Otsu thresholding. The blurring is done to quite the contrast in the image. Thresholding is done so that we can separate the regions of interest (crops) from the background. Thresholding also allows the subsequent contour detection and analysis to focus on the distinct regions representing the crops, making it easier to identify and process them accurately

- Morphological operations
The opening and closing morphological operations help to (1) remove small foreground regions and smooth out the boundaries of larger regions. (2) to close small gaps in the foreground regions and connect nearby regions that belong to the same object. These help highlight the ROIs.

- Contouring, detecting & counting crops
Next, contouring is perfomed to segment the crops from the image. The contours are filtered based on their area (this is done by trail and error). I also draw a enclosing circle on each contour and keep track of the center and radius of each circle. Finally, the code counts the number of detected crops and displays the count on the image

- To ensure this code is reproducible, I've attached a requirements.txt file which contains the packages and their respective versions
- The src/ folder contains the source code of this project, wherein crop_counting.py contains functions embedded in a class. The functions are
short and mostly focus on performing only one task. The number of arguemts to the functions are always less than 4. Overall, this helps with 
modularity and imrpves readability, gets rid of duplicate code as well as helps with reproducibility.
- The data/ folder contains the input image
- The results/ folder contains images from each step of the task
- .gitignore is added to ignore the virtual environent folder I created for this project

To run the bud_detection code:
''' python src/main.py --image=path/to/image'''