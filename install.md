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

Next, we'll create a virtual environment using Conda. If you don't have Conda installed, you can download it as part of the Anaconda distribution from their website.

Navigate to the directory where you cloned the repository and run the following command to create a Conda environment based on the provided environment.yml file:

```bash
conda env create --prefix ./envs --file Intramotev_env.yml
```

## Step 3: Run the Test File

Once the environment is set up, you can run the test file to verify everything is working correctly.

Activate the environment by running the following command:

```bash
conda activate ./Intramotev_env.yml
```

Then, run the test file using Python:

```bash
python test.py
```

This will execute the test script and display the results.

That's it! You've successfully installed Intramotev and run the test file. If you encounter any issues during the installation process, feel free to reach out for assistance.