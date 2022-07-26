# mobinetv2 serving (flask) using docker
# usage
```
use the following POST API endpoint to classify an image:
/predict

Request body example:
{ 
    "url": "https://s27870.pcdn.co/assets/flower-729512_1920-1024x678.jpg.optimal.jpg" 
}

Response body example:
{
    "ant": 0.005777660757303238,
    "bee": 0.007221573032438755,
    "daisy": 0.9742687940597534,
    "fly": 0.001883076736703515,
    "ladybug": 0.001868181861937046
}
```
```
use the following GET API endpoint to model metadata:
/metadata
```

# docker commands
```
docker build --build-arg VERSION=mobinet_v2 -t pytorch_serve:mobinetv2 .

docker run -p 5000:5000 -d --name pytorch_serve_mobinetv2 pytorch_serve:mobinetv2
```

# docker image in registry
[docker.io/martinoywa/pytorch_serve_mobinetv2](docker.io/martinoywa/pytorch_serve_mobinetv2)

# model source
[https://pytorch.org/hub/pytorch_vision_mobilenet_v2/](https://pytorch.org/hub/pytorch_vision_mobilenet_v2/)