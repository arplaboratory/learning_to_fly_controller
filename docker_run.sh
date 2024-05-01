set -e
docker run -it --rm -v $(pwd):/workdir -w /workdir rltools/crazyflie-controller-builder