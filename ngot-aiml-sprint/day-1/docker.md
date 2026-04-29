# Essential Docker Commands — Learn These by Heart

## Build & Run

`docker build -t eta:v1 .`  
Build image from Dockerfile in current dir

`docker run -d -p 8000:8000 --name api eta:v1`  
Run container (detached, port mapped)

`docker-compose up --build`  
Start ALL services defined in docker-compose.yml

## Inspect & Debug

`docker ps`  
List running containers

`docker logs api -f`  
Stream live logs from container

`docker exec -it api bash`  
Open shell INSIDE running container

## Clean Up

`docker stop api`  
Stop a running container

`docker rm api`  
Remove stopped container

`docker rmi eta:v1`  
Delete an image
