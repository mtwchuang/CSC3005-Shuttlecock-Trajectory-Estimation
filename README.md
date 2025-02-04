# Shuttlecock Trajectory Estimation
[Insert Project Description]

## Installation
### Python Virtual Environment and Dependencies
 
Working from the Linux command line on your IDE, dependencies will be handled through virtualenv. You can check if it is installed on pip by entering this on the CLI
```
pip install virtualenv
```
Once installed, create your virtual environment "dependencies" in your local directory
```
virtualenv dependencies
```
Activate the virtual environment
```
./dependencies/Scripts/activate
```
If you encounter an error saying ...\dependencies\Scripts\activate.ps1 cannot be loaded because running scripts is disabled on system, enter this to enable running of scripts on the system. 
```
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Download the dependencies
```
pip install -r requirements.txt
```
To deactivate the virtual environment
```
deactivate
```

## Order of Jupyter Scripts
### Preprocessing

### 