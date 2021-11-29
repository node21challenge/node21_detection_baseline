# Nodule Detection Algorithm

This codebase implements a baseline model, [Faster R-CNN](https://papers.nips.cc/paper/2015/hash/14bfa6bb14875e45bba028a21ed38046-Abstract.html), 
for the nodule detection track in [NODE21](https://node21.grand-challenge.org/). 
It contains all necessary files to build a docker image which can be submitted as an algorithm on the [grand-challenge](https://www.grand-challenge.org) platform.
Participants in the nodule detection track can use this codebase as a template to understand how to create their own algorithm for submission.

To serve this algorithm in a docker container compatible with the requirements of grand-challenge, 
we used [evalutils](https://github.com/comic/evalutils) which provides methods to wrap your algorithm in Docker containers. 
It automatically generates template scripts for your container files, and creates commands for building, testing, and exporting the algorithm container.
We adapted this template code for our algorithm by following the
[general tutorial on how to create a grand-challenge algorithm](https://grand-challenge.org/blogs/create-an-algorithm/). 

We also explain this template repository, and how to set up your docker container in [this video](https://www.youtube.com/watch?v=45BCMquFk70).
Before diving into the details of this template code we recommend readers have the pre-requisites installed and have cloned this repository as described below:

## Prerequisites
* [Docker](https://www.docker.com/get-started)
* [evalutils](https://github.com/comic/evalutils)
* [git lfs](https://git-lfs.github.com/) 

The code in this repository is based on docker and evalutils.  

**Windows Tip**: For participants using Windows, it is highly recommended to 
install [Windows Subsystem for Linux (WSL)](https://docs.microsoft.com/en-us/windows/wsl/install-win10) 
to work with Docker on a Linux environment within Windows. Please make sure to install **WSL 2** by following the instructions on the same page. 
The alternative is to work purely out of Ubuntu, or any other flavor of Linux.
Also, note that the basic version of WSL 2 does not come with GPU support. 
Please watch the [official tutorial](https://www.youtube.com/watch?v=PdxXlZJiuxA) 
by Microsoft on installing WSL 2 with GPU support.

Please clone the repository as follows:
```python
git clone git@github.com:node21challenge/node21_detection_baseline.git
```


## Table of Contents  
[An overview of the baseline algorithm](#algorithm)  
[Configuring the Docker File](#dockerfile)  
[Export your algorithm container](#export)   
[Submit your algorithm](#submit)  

<a name="algorithm"/>

## An overview of the baseline algorithm
The baseline nodule detection algorithm is a [Faster R-CNN](https://papers.nips.cc/paper/2015/hash/14bfa6bb14875e45bba028a21ed38046-Abstract.html) model, which was implemented using [pytorch](https://pytorch.org/) library. The main file executed by the docker container is [*process.py*](https://github.com/node21challenge/node21_detection_baseline/blob/main/process.py). 

### Input and Output Interfaces
The algorithm needs to perform nodule detection on a given chest X-ray image (CXR), predict a nodule bounding box where a nodule is suspected 
and return the bounding boxes with an associated likelihood for each one. 
The algorithm takes a CXR as input and outputs a nodules.json file.  All algorithms submitted to the nodule detection track must comply with these
input and output interfaces.
It reads the input :
* CXR at ``` "/input/<uuid>.mha"```
  
 and writes the output to
* nodules.json file at ``` "/output/nodules.json".```

The nodules.json file contains the predicted bounding box locations and associated nodule likelihoods (probabilities). 
This file is a dictionary and contains multiple 2D bounding boxes coordinates 
in [CIRRUS](https://comic.github.io/grand-challenge.org/components.html#grandchallenge.components.models.InterfaceKind.interface_type_annotation) 
compatible format. 
The coordinates are expected in milimiters when spacing information is available. 
We provide a [function](https://github.com/node21challenge/node21_detection_baseline/blob/main/process.py#L131) 
in [*process.py*](https://github.com/node21challenge/node21_detection_baseline/blob/main/process.py) 
which converts the predictions of the Faster R-CNN model (2D pixel coordinates) to this format. An example json file is as follows:
```python
{
    "type": "Multiple 2D bounding boxes",
    "boxes": [
        {
        "corners": [
            [ 92.66666412353516, 136.06668090820312, 0],
            [ 54.79999923706055, 136.06668090820312, 0],
            [ 54.79999923706055, 95.53333282470703, 0],
            [ 92.66666412353516, 95.53333282470703, 0]
        ]
        probability=0.6
        },
        {
        "corners": [
            [ 92.66666412353516, 136.06668090820312, 0],
            [ 54.79999923706055, 136.06668090820312, 0],
            [ 54.79999923706055, 95.53333282470703, 0],
            [ 92.66666412353516, 95.53333282470703, 0]
        ]}
    ],
    "version": { "major": 1, "minor": 0 }
}
```
The implementation of the algorithm inference in process.py is straightforward (and must be followed by participants creating their own algorithm): 
load the model in the [*__init__*](https://github.com/node21challenge/node21_detection_baseline/blob/main/process.py#L29) function of the class, 
and implement a function called [*predict*](https://github.com/node21challenge/node21_detection_baseline/blob/main/process.py#L171) 
to perform inference on a CXR image. 
The function [*predict*](https://github.com/node21challenge/node21_detection_baseline/blob/main/process.py#L171) is run by 
evalutils when the [process](https://github.com/node21challenge/node21_detection_baseline/blob/main/process.py#L222) function is called. 
Since we want to save the predictions produced by the *predict* function directly as a *nodules.json* file, 
we have overwritten the function [*process_case*](https://github.com/node21challenge/node21_detection_baseline/blob/main/process.py#L75) of evalutils.  
We recommend that you copy this implementation in your file as well.

### Operating on a 3D image (Stack of 2D CXR images)

For the sake of time efficiency in the evaluation process of [NODE21](https://node21.grand-challenge.org/), 
the submitted algorithms to [NODE21](https://node21.grand-challenge.org/) are expected to operate on a 3D image which consists of multiple CXR images 
stacked together. The algorithm should go through the slices (CXR images) one by one and process them individually, 
as shown in [*predict*](https://github.com/node21challenge/node21_detection_baseline/blob/main/process.py#L171). 
When outputting results, the third coordinate of the bounding box in nodules.json file is used to identify the CXR from the stack. 
If the algorithm processes the first CXR image in 3D volume, the z coordinate output should be 0, if it processes the third CXR image, it should be 2, etc. 

  
### Running the container in multiple phases:
A selection of NODE21 algorithms will be chosen, based on performance and diversity of methodology, for further experimentation and inclusion in a peer-reviewed
article.  The owners of these algorithms (maximum 3 per algorithm) will be co-authors on this publication.  
For this reason, we request that the container submissions to NODE21 detection track should implement training functionality as well as testing. 
This should be implemented in the [*train*](https://github.com/node21challenge/node21_detection_baseline/blob/main/process.py#L94) function 
which receives the input (containing images and metadata.csv) and output directory as arguments. The input directory is expected to look like this:
```
Input_dir/
├── metadata.csv
├── Images
│   ├── 1.mha
│   ├── 2.mha
│   └── 3.mha
```
The algorithm should train a model by reading the images and associated label file (metadata.csv) from the input directory and it should save the model 
file to the output folder. The model file (*model_retrained*) should be saved to the output folder **frequently** since the containers will be executed in 
training mode with a pre-defined time-limit, and training could be stopped before the defined stopping condition is reached.

The algorithms should have the possibility of running in four different phases depending on the pretrained model in test or train phase:
1. ```no arguments``` given (test phase): Load the 'model' file, and test the model on a given image. This is the default mode.
2. ```--train``` phase: Train the model from *scratch* given the folder with training images and metadata.csv. Save the model frequently as model_retrained.
3. ```--retrain``` phase: Load the 'model' file, and retrain the model given the folder with training images and metadata.csv. Save the model frequently as model_retrained.
4. ```--retest``` phase: Load 'model_retrain' which was created during the training phase, and test it on a given image.
  
This may look complicated, but it is not, no worries! Once the training function is implemented, implementing these phases is just a few lines of code
(see __init__ function).

The algorithms submitted to NODE21 detection track will be run in default mode (test phase) by grand-challenge. 
All other phases will be used for further collaborative experiments for the peer-reviewed paper.   
  
📌 NOTE: in case the selected solutions cannot be run in the training phase (or --retrain and --retest phases), the participants will be contacted 
***one time only*** to fix their docker image. 
If the solution is not fixed on time or the participants are not responsive, we will have to exclude their algorithm 
and they will not be eligible for co-authorship in the overview paper.

💡 To test this container locally without a docker container, you should the **execute_in_docker** flag to 
False - this sets all paths to relative paths. You should set it back to **True** when you want to switch back to the docker container setting.

  
<a name="dockerfile"/>

### Configure the Docker file
We recommend that you use our [dockerfile](https://github.com/node21challenge/node21_detection_baseline/blob/main/Dockerfile) as a template, 
and update it according to your algorithm requirements. There are three main components you need to define in your docker file in order to 
wrap your algorithm in a docker container:
1. Choose the right base image (official base image from the library you need (tensorflow, pytorch etc.) recommended)
```python
FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime
```
📌 NOTE: You should use a base image that is compatible with CUDA 11.x since that is what will be used on the grand-challenge system.

2. Copy all the files you need to run your model : model weights, *requirements.txt*, all the python files you need etc.
```python
COPY --chown=algorithm:algorithm requirements.txt /opt/algorithm/
COPY --chown=algorithm:algorithm entrypoint.sh /opt/algorithm/
COPY --chown=algorithm:algorithm model /opt/algorithm/
COPY --chown=algorithm:algorithm resnet50-19c8e357.pth  /home/algorithm/.cache/torch/hub/checkpoints/resnet50-19c8e357.pth
COPY --chown=algorithm:algorithm training_utils /opt/algorithm/training_utils
```

3. Install all the dependencies, defined in *reqirements.txt*, in your dockerfile.
```python
RUN python -m pip install --user -rrequirements.txt
```
Ensure that all of the dependencies with their versions are specified in requirements.txt:
```
evalutils==0.2.4
scikit-learn==0.20.2
scipy==1.2.1
--find-links https://download.pytorch.org/whl/torch_stable.html 
torchvision==0.10.0+cu111 
torchaudio==0.9.0
scikit-image==0.17.2
```

<a name="export"/>

### Build, test and export your container
1. Switch to the correct algorithm folder at algorithms/noduledetection. To test if all dependencies are met, you can run the file build.bat (Windows) / build.sh (Linux) to build the docker container. 
Please note that the next step (testing the container) also runs a build, so this step is not necessary if you are certain that everything is set up correctly.

    *build.sh*/*build.bat* files will run the following command to build the docker for you:
    ```python 
    docker build -t noduledetector .
    ```

2. To test the docker container to see if it works as expected, *test.sh*/*test.bat* will run the container on images provided in  ```test/``` folder, 
and it will check the results (*nodules.json* produced by your algorithm) against ```test/expected_output.json```. 
Please update your ```test/expected_output.json``` according to your algorithm result when it is run on the test data. 
   ```python
   . ./test.sh
   ```
    If the test runs successfully you will see the message **Tests successfully passed...** at the end of the output.

    Once you validated that the algorithm works as expected, you might want to simply run the algorithm on the test folder 
    and check the nodules.json file for yourself.  If you are on a native Linux system you will need to create a results folder that the 
    docker container can write to as follows (WSL users can skip this step).  (Note that $SCRIPTPATH was created in the previous test script)
    ```python
   mkdir $SCRIPTPATH/results
   chmod 777 $SCRIPTPATH/results
   ```
   To write the output of the algorithm to the results folder use the following command (note that $SCRIPTPATH was created in the previous test script): 
   ```python
   docker run --rm --memory=11g -v $SCRIPTPATH/test:/input/ -v $SCRIPTPATH/results:/output/ noduledetector
   ```
   
3. If you would like to run the algorithm in training mode (or any other modes), please make sure your training folder (which is mapped to /input) 
   has *'metadata.csv'* and  ```images/``` folder as described above.  If you are on a native Linux system make sure that
   your output folder has 777 permissions as mentioned in the previous step.  You can use the following command to start training -(you may also need to add
   the flag *--shm-size 8G* (for example) to specify shared memory that the container can use:
   ```python
   docker run --rm --gpus all --memory=11g -v path_to_your_training_folder/:/input/ -v path_to_your_output_folder/:/output/ noduledetector --train
   ```

4. Run *export.sh*/*export.bat* to save the docker image which runs the following command:
   ```python
    docker save noduledetector | gzip -c > noduledetector.tar.gz
   ```
    
 ### Submit your algorithm  
Please make sure all steps described above work as expected before proceeding.  Ensure also that you have an account on grand-challenge.org and that you are a [verified user](https://grand-challenge.org/documentation/account-verification/) there.
 
 Once you test that your docker container runs as expected, you are ready to submit! Let us walk you through the steps you need to follow to upload and submit your algorithm to [NODE21](https://node21.grand-challenge.org/) detection track:

1. In order to submit your algorithm, you first have to create an algorithm entry for your docker container [here](https://grand-challenge.org/algorithms/create/).
   * Please choose a title and description for your algorithm and enter the modalities and structure information as in the example below.
      ![alt text](images/create_alg_det1.png)

   * Scrolling down the page, you will see that you need to enter further information:
   * Select a logo to represent your algorithm (preferably square image)
   * For the interfaces of the algorithm, please select *Generic Medical Image (Image)* as Inputs, and *Nodules (Multiple 2D Bounding Boxes)* as Outputs.
   * Do not forget to pick the workstation *Viewer CIRRUS Core (Public)*.  
   
   ![alt text](images/create_alg_det2.png)
  
   * At the bottom of the page, indicate that you would like your Docker image to use GPU and how much memory it needs
   ![alt text](images/container_img_config.PNG)
   * Don't forget to click the Save button at the bottom of the page to create your algorithm.  You will then see your algorithm page as shown below.
   ![alt text](images/create_alg_det3.png)
   
2. After creating your algorithm, you need to attach a docker container to it so it becomes functional.  There are 2 ways to do this:
   * OPTION 1: If you already built the tar.gz file as described above then you can upload this file directly
   * OPTION 2: You can provide your repository details and the file will be built by Grand-Challenge from your repository code. Please bear in mind that the root of the github repository must contain the dockerfile, the licence, the gitattributes in order to build the image for you. Further, you must have admin rights to the repository so that you can give permission for Grand Challenge to install an app there.
   
   For either option, first choose "Containers" on the left menu of your screen as shown below, then follow the detailed steps for your option below.
   ![alt text](images/create_alg_det4.png)
   
    **OPTION 1:** If you would like to upload your docker container directly, click on "upload a Container" button, and upload your tar.gz file. 
    You can also later overwrite your container by uploading a new one (ie. you should not create a whole new algorithm if you make some code changes):
    Please note that it can take a while until the container becomes active. (Status will change from "Ready: False" to "Active")  Check back later or refresh the URL after some time.

    **OPTION 2:** If you would like to submit your repository and let grand-challenge build the docker image for you, click on "Link Github Repo".
    Your repo will be listed in the dropdown box only if it has the Grand-Challenge app already installed.  
    Usually this is not the case to begin with, so you should click on "link a new Github Repo".  
    This will guide you through the installation of the Grand-challenge app in your repository.
    ![alt text](images/add_repo_det.png)
    
    After the installation of the app in your repository is complete you should be automatically returned to the Grand Challenge page shown above, 
    where you will find your repository now in the dropdown list.
    (In case you are not automatically returned there you can [find your algorithm](https://grand-challenge.org/algorithms/) and click "Link Github Repo" again)
    Select your repository from the dropdown list and click "Save"
    ![alt text](images/add_repo_det3.png)

    Finally, you need to tag your repository, this will trigger Grand-Challenge to start building the docker container.
    Please note that it can take a little while for Grand-Challenge to start building your container, and some further time to complete the build.  Check back later or refresh the URL after some time.
      
3. **Both OPTION1 and OPTION2 can continue from here:** Once your algorithm is ready to run you will see the status "Active" beside it as shown below
    ![alt text](images/build_complete_det.png)
    Once it becomes active, we suggest that you try out the algorithm to verify everything works as expected. For this, please click on *Try-out Algorithm* tab, and upload a *Generic Medical Image*. You could upload the image provided here in the test folder since it is a 3D image (CXRs are stacked together) which is the expected format.
  ![alt text](images/algorithm_tryout.PNG)
   OPTIONAL: You could look at the results of your algorithm: click on the *Results*, and *Open Result in Viewer* to visualize the results. You would be directed to CIRRUS viewer, and the results will be visualized with the predicted bounding boxes on chest x-ray images as below. You could move to the next and previous slice (slice is a chest x-ray in this case) by clicking on the up and down arrow in the keyboard.
     ![alt text](images/algorithm_results.PNG)

4. You have created and tested your Grand-Challenge Algorithm!!  The last step is to submit your algorithm to the NODE21 challenge to have it evaluated and get your spot on the leaderboard!
   Go to the [NODE21 submission page](https://node21.grand-challenge.org/evaluation/challenge/submissions/create/), and click on the track where you want to participate
   ("Detection Track Experimental Test Set" OR "Detection Track Final Test Set").  
   **Only one submission to the Final Test Set is allowed so please use the Experimental Test Set for your first testing efforts!**
   Choose your algorithm name from the dropdown list and click on "Save"
5. Allow some time for the evaluation to be processed.  It may not start immediately depending on what hardware resources are available at the time. 
   Your result will appear on the [leaderboard page](https://node21.grand-challenge.org/evaluation/challenge/leaderboard/) (make sure to select the track where you submitted).
   If you do not see your result after 24 hours please contact us via email or the forum and provide your algorithm name and date/time you submitted in Central European Time.
   

### Re-submitting if you make improvements
If you already have an algorithm on Grand-Challenge you do NOT need to make a new one whenever you have made some improvements or changes to your code. 
To re-submit, first just update your existing algorithm as follows:

If you used **OPTION 1** in previous steps and uploaded a tar.gz file then you can now upload a new tar.gz file which will overwrite the old one.  Find your algorithm at [https://grand-challenge.org/algorithms/](https://grand-challenge.org/algorithms/).  Click on "Containers", and "upload a container" as before.  Allow some time for the algorithm to become "Active" as previously. 

If you used **OPTION 2** in previous steps and allowed Grand-Challenge to build from your repository then a new build will start automatically whenever you tag your repository.  So just tag the repository and allow some time for the new build to start and complete, it will then show the "Active" label.

**OPTION 1 AND OPTION 2**:  Whichever option you use, once your algorithm is updated you need to resubmit to NODE21, this does not happen automatically!  Visit the [NODE21 submission page](https://node21.grand-challenge.org/evaluation/challenge/submissions/create/) and proceed to submit to your chosen track as before. 


