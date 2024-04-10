# Installation Instructions

Welcome to the installation guide for Intramotev! This guide will walk you through the process of downloading the repository, setting up the environment, and running the test file.

## Step 1: Download the Repository

First, you'll need to download the Intramotev repository using Git. If you don't have Git installed, you can download it from [here](https://git-scm.com/).

Open a terminal or command prompt and run the following command to clone the repository:

```bash
git clone https://github.com/aryandhr/Intramotev.git
```

This command will create a local copy of the repository on your machine.

## Step 2: Prep Environment and install packages

First we want to create an environment to enable users to keep all files, packages, and dependencies in one place. 

### Build Environment

To create an environment, enter the repository folder and execute the following commands to create and enter the repository:

```bash
python3 -m venv rails_and_sails
source rails_and_sails/bin/activate
```

Note that you can rename the environment whatever you please.

### Install packages

 You have now created and entered your virtual environment. Next run the following commands to install all the required pacakges and dependencies:

```bash
pip install ultralytics
pip install scikit-learn 
```

If you don't have pip installed, then you need to install pip using whatever tools your computer requires.

## Step 3: Run the Test File

Once the environment is set up, you can run the test file to verify everything is set up correctly. Make sure to get your environment running or install all the packages as required above.

Then, run the test file using Python:

```bash
python run_image.py
```

This will execute the test script and display the results. The results will be stored in the output folder. You can view the csv file of all detected objects and the processed image.

That's it! You've successfully installed Intramotev and run the test file. If you encounter any issues during the installation process, feel free to reach out for assistance.
