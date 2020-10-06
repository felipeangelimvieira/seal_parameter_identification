#jupyter lab --ip=0.0.0.0 --port=8080 --allow-root


docker run -v %cd%:/var/code/ -p 8080:8080 jax jupyter lab --ip=0.0.0.0 --port=8080 --allow-root