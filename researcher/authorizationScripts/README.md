# Mlflow authorization  
This folder contains scripts related to authorization and authentication processes. These scripts are essential for managing access control and security of your account with the cloud running instance of mlflow.

## Usage

1. Please provide the contact information required [here](https://docs.google.com/forms/d/e/1FAIpQLSdJRO8ZcNn1gNe3yRuWtP2j2ujQbCgA0NEdjMhlyHuInfDr_Q/viewform?usp=header).
2. The administrator will proceed with the creation of an account within the mlflow server.
3. You will receive an email once the creation of the profile is completed.
4. You will then be able to access it using your desired username and the password: "password". It is strongly recommended to update your password using the script `updatePassword.sh`:
```bash
 bash updatePassword.sh <username> password <newPassword>
```
5. Set the environment variables `MLFLOW_TRACKING_USERNAME` and `MLFLOW_TRACKING_PASSWORD`
6. You're ready to go!

### Optional
By default, other users can't see your experiments. However, you can set specific access to experiments that you created using the script `setPermission.sh`:
```bash
 bash setPermission.sh <experiment-id> <target-username> <permission>
```
The available permissions are:
| Permission | Can read | Can update | Can delete | Can manage |
|-----------------|-------------|-----|-----|-----|
| READ | yes | no | no | no |
| EDIT | yes | yes | no | no |
| MANAGE | yes | yes | yes | yes |
| NO_PERMISSION | No  | yes | no | no |

More info availba at the official [webpage](https://mlflow.org/docs/latest/auth).