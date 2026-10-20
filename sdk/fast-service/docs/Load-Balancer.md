# Load Balancer

## Nginx

1. pull the image
```shell
docker pull nginx
```

2. set up the config file nginx.conf with the following content
```
http {
    upstream backend {
        server <YOUR_SERVER_IP>:7001;
        server <YOUR_SERVER_IP>:7002;
        server <YOUR_SERVER_IP>:7003;
    }

    server {
        listen 7000;

        location / {
            proxy_pass http://backend;
        }
    }
}

events {}
```

3. run a container
```shell
docker run -d --name nginx-load-balancer -p 7000:7000 -v ./nginx.conf:/etc/nginx/nginx.conf:ro nginx
```

