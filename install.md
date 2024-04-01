# Installation Instructions

Welcome to the installation guide for Intramotev! This guide will walk you through the process of downloading the repository, setting up the environment, and running the test file.

## Step 1: Download the Repository

First, you'll need to download the Intramotev repository using Git. If you don't have Git installed, you can download it from [here](https://git-scm.com/).

Open a terminal or command prompt and run the following command to clone the repository:

```bash
git clone https://github.com/aryandhr/Intramotev.git
```

This command will create a local copy of the repository on your machine.

## Step 2: Create Environment

There are two ways that you can prep your device for running our code. The first is a conda environment. This will create a virtual environment with all the necessary packages on your computer. This is recommended if you have anaconda installed on your computer. If you don't, you will need to follow the second option, manually installing required pacakges.

### Conda Environment

Next, we'll create a virtual environment using Conda. If you don't have Conda installed, you can download it as part of the Anaconda distribution from their website.

Navigate to the directory where you cloned the repository and run the following command to create a Conda environment based on the provided environment.yml file:

```bash
conda env create --prefix ./envs --file Intramotev_env.yml
```

Once the environment is set up, you can activate the environment by running the following command:

```bash
conda activate ./envs
```

This will have you enter the environment!

### Manually install packages

The second way we can get this running is by manually creating our own environment and installing all required packages and dependencies. Depending on your device, there may be additional requirements, so be aware! To create an environment, enter the repo folder and execute the following commands to create and enter the repository:

```bash
python3 -m venv rails_and_sails
source rails_and_sails/bin/activate
```

Note that you can rename the environment whatever you please. You have now created and entered your virtual environment. Next run the following commands to install all the required pacakges and dependencies:

```bash
pip install ultralytics
```

If you don't have pip installed, then you need to install pip using whatever tools your computer requires.

## Step 3: Run the Test File

Once the environment is set up, you can run the test file to verify everything is set up correctly. Make sure to get your environment running or install all the packages as required above.

Then, run the test file using Python:

```bash
python test.py
```

This will execute the test script and display the results.

That's it! You've successfully installed Intramotev and run the test file. If you encounter any issues during the installation process, feel free to reach out for assistance.
