Performing Steps – Assignment 6 Set A (MLflow Experiment Tracking)

Create project folder and enter it

mkdir mlflow_assignment6
cd mlflow_assignment6


Create and activate virtual environment

python -m venv venv
venv\Scripts\activate


Install required packages

pip install mlflow scikit-learn pandas numpy matplotlib


Start MLflow tracking server (keep running)

python -m mlflow server --host 127.0.0.1 --port 8080


Open browser → http://127.0.0.1:8080

Open new terminal (same folder, activate venv again)

venv\Scripts\activate


Create script file

notepad mlflow_tracking.py


→ Paste the given MLflow tracking code
→ Save and close.

Run the experiment

python mlflow_tracking.py


View results in browser
Visit → http://127.0.0.1:8080

Check:

Experiment: Apples_Experiment

Run name: apple_sales_prediction_model

Tabs: Parameters, Metrics, Artifacts

(Optional) Stop server when done → press Ctrl + C in the MLflow server window.

✅ Done — Set A complete.


Set B

open : 2. Install Extra Package for Tuning
pip install hyperopt

pip install setuptools

pip install -U wheel hyperopt


set C

Performing Steps – Set C (Model Deployment)

Activate virtual environment

cd C:\Users\ABHISHEK SONTAKKE\mlflow_assignment6
venv\Scripts\activate


Start MLflow tracking server (in a separate terminal and keep running)

python -m mlflow server --host 127.0.0.1 --port 8080


Open browser → http://127.0.0.1:8080

Verify that your model is registered

Sidebar → Models

Check model name wine-quality-predictor, Stage = Staging

Serve the model as a REST API

mlflow models serve -m "models:/wine-quality-predictor/Staging" -p 1234


or (specify version directly)

mlflow models serve -m "models:/wine-quality-predictor/1" -p 1234


Keep this window open.
Server URL → http://127.0.0.1:1234

Create input file for prediction

notepad input.json


Paste:

{
  "columns": ["alcohol","malic_acid","ash","alcalinity_of_ash","magnesium",
              "total_phenols","flavanoids","nonflavanoid_phenols",
              "proanthocyanins","color_intensity","hue",
              "od280/od315_of_diluted_wines","proline"],
  "data": [[13.2,1.78,2.14,11.2,100,2.65,2.76,0.26,1.28,10.45,1.22,2.31,1030]]
}


Send prediction request

curl -X POST -H "Content-Type: application/json" --data @input.json http://127.0.0.1:1234/invocations


✅ Expected output:

[1]


Stop the server after testing
Press Ctrl + C in the serving terminal.




========================
Ass 3

nano inventory "1st and 3rd take "

nano ansible .cfg "path copy and paste "
pwd "for path"

ssh-keygen -t ed25519 -c "ansible"

ansible "ADD then tab"

ls -l -/.ssh/

cat -/.ssh/ansible.pub   "copy all"

nano inventory ip change

nano -/.ssh/authorized_keys "paste all"

in other terminal :
ansible all -m ping 

ansible all -m builtin.apt -a "name=apache2 state=presented" --become


ls

cat 'ansible commands Assignment - III.txt'


change only "d ** ubuntu"

cd playbooks
ls
cd ..
ansible-playbook/java_playbook.yml
ansible-playbook playboos/deploy_website.yml

cd playbooks
code .
"update playbook"

"  https://www.tooplate.com/zip-templates/2103_central.zip  "


ASS3 End
=====




lab only-->
🧩 Assignment 1 – Job Build & Test Execution using Jenkins
Set A

Create a Jenkins Freestyle job to:

Fetch code from GitHub.

Build using Maven.

Archive artifacts.

Use post-build actions for success and failure cases.

Add Jenkins email notification.

Configure Git SCM, build triggers, and environment variables.

Set B

Create a Jenkins freestyle job that extends the basic Docker workflow with:

Git triggers for new code pushes.

Build Docker image using build args (BUILD_ID, GIT_COMMIT).

Multi-stage Docker build and tag image dynamically.

Push to Docker Hub and configure email alerts for build status.

Create two linked Jenkins jobs:

Job A (Build): Pull code, build Maven project, archive artifact.

Job B (Deploy): Triggered post-build, deploy artifact to target directory.

Set C

Design and document a fully automated CI/CD freestyle job using Jenkins that leverages on-demand EC2 instances as dynamic build agents.

⚙️ Assignment 2 – Cross-Environment Pipeline Synthesis with Groovy DSL
Set A

Write a Declarative Pipeline to:

Checkout from Git.

Build with Maven.

Run tests, SonarQube, and quality gate.

Archive artifacts.

Add environment variables and post actions for results.

Set B

Implement multi-branch pipeline.

Add shared library integration.

Define parameters for environment-specific builds (dev, stage, prod).

Set C

Demonstrate Jenkins Shared Library use (vars, resources, src).

Show pipeline using multiple Docker agents and containers.

🧰 Assignment 3 – Deploying Configurations with Ansible Playbooks
Set A

Launch two VMs (Debian + RedHat).

Create ansible.cfg and inventory files.

Run these ad-hoc commands:

Install apache2/httpd.

Demonstrate copy, file, service, apt, yum, user, setup modules.

Set B

Launch 3 VMs (2 Debian + 1 RedHat).

Create playbooks to:

Update apt, install Java, show version.

Install and start Nginx.

Add multiple users.

Set C

Create playbook to:

Install Nginx.

Download project from tooplate.com
.

Deploy to /var/www/html.

Make accessible via IP.

Deploy and configure Time Service using handlers and conditionals.

🧮 Assignment 4 – Ansible Variable Management, Templating, and Roles
Set A

Launch 2 VMs (Debian family).

Create ansible.cfg and inventory.

Create host_vars and group_vars.

Demonstrate variable usage with user creation and debug.

Set B

Create template directory with:

index.html.j2 showing hostname & IP.

nginx.conf.j2 with configuration.

Write playbooks to install Nginx and copy configs.

Set C

Create apache role (for Debian) serving hostname/IP.

Create httpd role (for RedHat) serving hostname/IP.

☸️ Assignment 5 – Kubernetes (K8s) Deployment Practical
Set A

Deploy PostgreSQL + Adminer on a K8s cluster (1 master, 1 worker).

Set B

Deploy two-tier app (PostgreSQL + Go API).

Set C

Deploy three-tier app:

PostgreSQL database.

Go API.

Frontend consuming API.

🤖 Assignment 6 – Introduction to MLOps with MLflow
Set A

Launch MLflow tracking server & client.

View experiment metadata.

Display default experiment name & stage.

Create “apples” experiment with tags.

Generate synthetic dataset for apple sales.

Train and log model using MLflow Tracking.

Set B

Launch MLflow tracking server for hyperparameter tuning.

Optimize Wine Quality Prediction model with parameters:

Learning Rate

Momentum

Steps:

Prepare Data

Define Model

Set Up & Run Optimization

Analyze Results

Register Best Model

Set C

Deploy model locally via REST API.

Build and test Docker container for deployment.
----->end


s

| **Type**        | **Protocol** | **Port Range** | **Source** |
| --------------- | ------------ | -------------- | ---------- |
| SSH             | TCP          | 22             | 0.0.0.0/0  |
| Custom TCP Rule | TCP          | 30000          | 0.0.0.0/0  |
| Custom TCP Rule | TCP          | 30009          | 0.0.0.0/0  |
| Custom TCP Rule | TCP          | 27017          | 0.0.0.0/0  |
| Custom TCP Rule | TCP          | 30007          | 0.0.0.0/0  |
| Custom TCP Rule | TCP          | 8080           | 0.0.0.0/0  |
| Custom TCP Rule | TCP          | 6443           | 0.0.0.0/0  |
| PostgreSQL      | TCP          | 5432           | 0.0.0.0/0  |
| Custom TCP Rule | TCP          | 8081           | 0.0.0.0/0  |
| Custom TCP Rule | TCP          | 30008          | 0.0.0.0/0  |

