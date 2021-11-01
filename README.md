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

Before diving into the details of this template code we recommend readers have the pre-requisites installed and have cloned this repository as described on the 
[main README page](https://github.com/DIAGNijmegen/node21), and that they have gone through 
the [general tutorial on how to create a grand-challenge algorithm](https://grand-challenge.org/blogs/create-an-algorithm/). 

The details of how to build and submit the baseline NODE21 nodule detection algorithm using our template code are described below.

## Table of Contents  
[An overview of the baseline algorithm](#algorithm)  
[Configuring the Docker File](#dockerfile)  
[Export your algorithm container](#export)   
[Submit your algorithm](#submit)  

<a name="algorithm"/>

## An overview of the baseline algorithm
The baseline nodule detection algorithm is a [Faster R-CNN](https://papers.nips.cc/paper/2015/hash/14bfa6bb14875e45bba028a21ed38046-Abstract.html) model, which was implemented using [pytorch](https://pytorch.org/) library. The main file executed by the docker container is [*process.py*](https://github.com/DIAGNijmegen/node21/blob/main/algorithms/noduledetection/process.py). 

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
We provide a [function](https://github.com/DIAGNijmegen/node21/blob/main/algorithms/noduledetection/process.py#L121) 
in [*process.py*](https://github.com/DIAGNijmegen/node21/blob/main/algorithms/noduledetection/process.py) 
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
load the model in the [*__init__*](https://github.com/DIAGNijmegen/node21/blob/main/algorithms/noduledetection/process.py#L29) function of the class, 
and implement a function called [*predict*](https://github.com/DIAGNijmegen/node21/blob/main/algorithms/noduledetection/process.py#L166) 
to perform inference on a CXR image. 
The function [*predict*](https://github.com/DIAGNijmegen/node21/blob/main/algorithms/noduledetection/process.py#L166) is run by 
evalutils when the [process](https://github.com/DIAGNijmegen/node21/blob/main/algorithms/noduledetection/process.py#L217) function is called. 
Since we want to save the predictions produced by the *predict* function directly as a *nodules.json* file, 
we have overwritten the function [*process_case*](https://github.com/DIAGNijmegen/node21/blob/main/algorithms/noduledetection/process.py#L71) of evalutils.  
We recommend that you copy this implementation in your file as well.

### Operating on a 3D image (Stack of 2D CXR images)

For the sake of time efficiency in the evaluation process of [NODE21](https://node21.grand-challenge.org/), 
the submitted algorithms to [NODE21](https://node21.grand-challenge.org/) are expected to operate on a 3D image which consists of multiple CXR images 
stacked together. The algorithm should go through the slices (CXR images) one by one and process them individually, 
as shown in [*predict*](https://github.com/DIAGNijmegen/node21/blob/main/algorithms/noduledetection/process.py#L181). 
When outputting results, the third coordinate of the bounding box in nodules.json file is used to identify the CXR from the stack. 
If the algorithm processes the first CXR image in 3D volume, the z coordinate output should be 0, if it processes the third CXR image, it should be 2, etc. 

  
### Running the container in multiple phases:
A selection of NODE21 algorithms will be chosen, based on performance and diversity of methodology, for further experimentation and inclusion in a peer-reviewed
article.  The owners of these algorithms (maximum 3 per algorithm) will be co-authors on this publication.  
For this reason, we request that the container submissions to NODE21 detection track should implement training functionality as well as testing. 
This should be implemented in the [*train*](https://github.com/DIAGNijmegen/node21/blob/main/algorithms/noduledetection/process.py#L90) function 
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
We recommend that you use our [dockerfile](https://github.com/DIAGNijmegen/node21/blob/main/algorithms/noduledetection/Dockerfile) as a template, 
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
    
 <a name="submit"/>
 
 ### Submit your algorithm
 You could submit your algorithm in two different ways: by uploading your docker container (your .tar.gz file), or by submitting your github repository. 
 
 Once you test that your docker container runs as expected, you are ready to submit! Let us walk you through the steps you need to follow to upload and submit your algorithm to [NODE21](https://node21.grand-challenge.org/) detection track:

1. In order to submit your algorithm, you first have to create an algorithm entry for your docker container [here](https://grand-challenge.org/algorithms/create/).
   * Please choose a title for your algorithm and add a (squared image) logo. Enter the modalities and structure information as in the example below.
      ![alt text](https://github.com/DIAGNijmegen/node21/blob/main/images/algorithm_description.PNG)

   * Scrolling down the page, you will see that you need to enter further information:
   * Enter the URL of your GitHub repository which must be public, contain all your code and an [Apache 2.0 license](https://www.apache.org/licenses/LICENSE-2.0). When entering the repo name in algorithm-creation do not enter a full URL, only the part that comes after github.com/. For example if your github url is https://github.com/ecemlago/node21_detection_baseline/, please enter the field as *ecemlago/node21_detection_baseline*.
   * For the interfaces of the algorithm, please select *Generic Medical Image (Image)* as Inputs, and *Nodules (Multiple 2D Bounding Boxes)* as Outputs.
   * Do not forget to pick the workstation *Viewer CIRRUS Core (Public)*.  
   
   ![alt text](https://github.com/ecemlago/node21_detection_baseline/blob/master/images/alg_interfaces.PNG)
  
   * At the bottom of the page, indicate that you would like your Docker image to use GPU and how much memory it needs
   ![alt text](https://github.com/DIAGNijmegen/node21/blob/main/images/container_img_config.PNG)
   
2. After saving it, you can either upload your docker container (.tar.gaz) or you can let grand-challenge build your algorithm container from your github repository. 
    
    OPTION 1: If you would like to upload your docker container directly, please click on "upload a Container" button, and upload your container. You can also later overwrite your container by uploading a new one. That means that when you make changes to your algorithm, you could overwrite your container and submit the updated version of your algorithm to node21:
    ![alt text](https://github.com/ecemlago/node21_detection_baseline/blob/master/images/algorithm_uploadcontainer.PNG)
    
    OPTION 2: If you would like to submit your repository and let grand-challenge build the docker image for you, please click on "Link github repo" and select your repository to give repository access to grand-challenge to build your algorithm. Once this is done, you should tag the repo to kick off the build process. Please bear in mind that, the root of the github repository must contain the dockerfile, the licence, the gitattributes in order to build the image for you. Further, you must have admin rights to the repository so that you can give permission for GC to install an app there. 
    ![alt text](https://github.com/ecemlago/node21_detection_baseline/blob/master/images/container_image.PNG)

3. OPTIONAL: Please note that it can take a while (several minutes) until the container becomes active. After it uploads successfully you will see the details of the Algorithm with "Ready: False"
   You can check back at any time on the "Containers" page and see if the status has changed to "Active".
  Once it becomes active, we suggest that you try out the algorithm to verify everything works as expected. For this, please click on *Try-out Algorithm* tab, and upload a *Generic Medical Image*. You could upload the image provided here in the test folder since it is a 3D image (CXRs are stacked together) which is the expected format.
  ![alt text](https://github.com/DIAGNijmegen/node21/blob/main/images/algorithm_tryout.PNG)
4. OPTIONAL: You could look at the results of your algorithm: click on the *Results*, and *Open Result in Viewer* to visualize the results. You would be directed to CIRRUS viewer, and the results will be visualized with the predicted bounding boxes on chest x-ray images as below. You could move to the next and previous slice (slice is a chest x-ray in this case) by clicking on the up and down arrow in the keyboard.
    ![alt text](https://github.com/DIAGNijmegen/node21/blob/main/images/algorithm_results.PNG)

5. Go to the [NODE21](https://node21.grand-challenge.org/evaluation/challenge/submissions/create/) submission page, and submit your solution to the detection track by choosing your algorithm.
   ![alt text](https://github.com/DIAGNijmegen/node21/blob/main/images/node21_submission.PNG)
    



