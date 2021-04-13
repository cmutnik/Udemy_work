# Nvidia Certification #

----
## Class Notes ##

There is a [brief PDF on activation functions](./Notes/activation_functions.pdf), but most notes can be seen [here.](./Notes/)

----
## Docker Notes ##

in the [`docker-compose.yml`](./docker-compose.yml) file, you can specify `environment` for env vars stored in [`.env`](./.env):  
```yml
version: "3.7"
services:
  tf-jupyter:
    environment:
      - JUPYTER_TOKEN=${JUPYTER_TOKEN}
```
Here, `$JUPYTER_TOKEN` is stored in [`.env`](./.env) as 
```
JUPYTER_TOKEN=jupyter
```
and causes the user to be prompted for the set password.  If removed from the compose file, you wont be prompted.  You can use any filename youd like, if you pass in `docker-compose --env-file` or specify directly in the `yml` file:  
```yml
web:
  env_file:
    - web-variables.env
```