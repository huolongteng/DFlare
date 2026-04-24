# Running DFlare on Windows with Docker

This repository depends on an older TensorFlow and Keras stack. Running it directly on Windows is possible, but it is much more fragile than running the original Linux-oriented stack inside Docker.

The files added for this workflow are:

- `Dockerfile`
- `.dockerignore`
- `docker/entrypoint.sh`
- `docker/requirements.tflite.txt`

These files target the main TFLite workflow described in the paper and repository:

```powershell
python gen_wrapper_tflite.py --dataset mnist --arch lenet1 --maxit 10000000 --seed 0 --num 500 --cps_type quan --output_dir ./results
```

## 1. What this Docker image covers

The container is designed for the current repository entrypoint:

- `gen_wrapper_tflite.py`
- `test_gen_main.py`
- `model_loader/load_tflite.py`

It includes the old TensorFlow 2.2 / Keras 2.4 environment and `pyflann`, which is the most Windows-sensitive dependency in this project.

It does not try to unify every environment mentioned in the paper. The original README explicitly notes that some model families need separate environments, especially DeepSpeech.

## 2. Files you must have before running

Make sure these inputs are present in the repository before you build or run the container:

- `seed_inputs.p`
- `diffchaser_models/lenet1.h5`
- `diffchaser_models/lenet1-quan.lite`
- `diffchaser_models/lenet5.h5`
- `diffchaser_models/lenet5-quan.lite`
- `diffchaser_models/resnet.h5`
- `diffchaser_models/resnet-quan.lite`

If you want to use models downloaded separately from the paper authors, place them in the same paths expected by the code.

## 3. Windows host setup

Recommended setup:

1. Install Docker Desktop.
2. Enable the WSL 2 backend in Docker Desktop.
3. Open PowerShell in the repository root.

Build the image:

```powershell
docker build -t dflare:tflite .
```

Run the example command from the README:

```powershell
docker run --rm -it `
  --name dflare-run `
  -v "${PWD}\results:/workspace/DFlare/results" `
  dflare:tflite `
  python gen_wrapper_tflite.py --dataset mnist --arch lenet1 --maxit 10000000 --seed 0 --num 500 --cps_type quan --output_dir ./results
```

Run CIFAR / ResNet:

```powershell
docker run --rm -it `
  -v "${PWD}\results:/workspace/DFlare/results" `
  dflare:tflite `
  python gen_wrapper_tflite.py --dataset cifar --arch resnet --maxit 10000000 --seed 0 --num 500 --cps_type quan --output_dir ./results
```

Open an interactive shell inside the container:

```powershell
docker run --rm -it `
  -v "${PWD}\results:/workspace/DFlare/results" `
  dflare:tflite `
  bash
```

If you run the image without a command, the container stays alive by default. This is intentional so the same image can also be used as a custom Runpod Pod image.

## 4. Remote Linux server setup

If you prefer to run on a remote Ubuntu machine, the steps are usually simpler than trying to make the old stack work natively on Windows.

On the remote machine:

```bash
sudo apt-get update
sudo apt-get install -y docker.io git
sudo systemctl enable --now docker
sudo usermod -aG docker "$USER"
```

Then reconnect your shell and run:

```bash
git clone <your-repo-url> DFlare
cd DFlare
docker build -t dflare:tflite .
mkdir -p results
docker run --rm -it \
  -v "$(pwd)/results:/workspace/DFlare/results" \
  dflare:tflite \
  python gen_wrapper_tflite.py --dataset mnist --arch lenet1 --maxit 10000000 --seed 0 --num 500 --cps_type quan --output_dir ./results
```

If the remote machine is only for running experiments, this is the most stable path.

## 4.5 Build on a MacBook, then run on Runpod

Yes, this is a good workflow, but there is one important rule:

- build the image as `linux/amd64`, not as Apple Silicon `arm64`

Runpod’s documentation specifically recommends `--platform linux/amd64` for Mac builds targeting Runpod.

Suggested flow on macOS:

1. Install Docker Desktop.
2. Sign in to Docker Hub or another registry.
3. From the repository root, build and push an `amd64` image.

If you are on Apple Silicon, use `buildx`:

```bash
docker buildx create --use --name runpod-builder
docker buildx inspect --bootstrap
docker buildx build \
  --platform linux/amd64 \
  -t YOUR_DOCKER_USERNAME/dflare:tflite-v1 \
  --push \
  .
```

If you want to test locally before pushing:

```bash
docker buildx build \
  --platform linux/amd64 \
  -t dflare:tflite-v1 \
  --load \
  .

docker run --rm -it \
  --platform linux/amd64 \
  -v "$(pwd)/results:/workspace/DFlare/results" \
  dflare:tflite-v1 \
  python gen_wrapper_tflite.py --dataset mnist --arch lenet1 --maxit 10000000 --seed 0 --num 500 --cps_type quan --output_dir ./results
```

Then in Runpod:

1. Push the image to Docker Hub, GHCR, or another supported registry.
2. In Runpod, create a custom Pod template.
3. Put your image URL in the Container Image field, for example `docker.io/YOUR_DOCKER_USERNAME/dflare:tflite-v1`.
4. Set container disk to something safe like 15 GB or more.
5. Deploy the Pod.
6. Open the web terminal or SSH into the Pod.
7. Run the experiment command manually inside the Pod:

```bash
cd /workspace/DFlare
python gen_wrapper_tflite.py --dataset mnist --arch lenet1 --maxit 10000000 --seed 0 --num 500 --cps_type quan --output_dir ./results
```

Important notes:

- Do not upload a local Docker tarball unless you only want backup/archive purposes. For Runpod deployment, the normal path is to push the image to a container registry and let Runpod pull it.
- Avoid using the `latest` tag for Runpod. Use version tags like `tflite-v1`, `tflite-v2`, and so on.
- If you later bake very large models into the image, image push/pull time will grow a lot. At that point, it may be better to keep models on a mounted volume or download them at startup.

## 5. GPU option

The provided `Dockerfile` uses a CPU TensorFlow base image because that is the safest default for reproduction.

If you specifically need GPU support:

1. Change the first line of `Dockerfile` to `FROM tensorflow/tensorflow:2.2.0-gpu`.
2. Install the NVIDIA Container Toolkit on the Linux host.
3. Run the container with `--gpus all`.

Example:

```bash
docker run --rm -it --gpus all \
  -v "$(pwd)/results:/workspace/DFlare/results" \
  dflare:tflite \
  python gen_wrapper_tflite.py --dataset mnist --arch lenet1 --maxit 10000000 --seed 0 --num 500 --cps_type quan --output_dir ./results
```

## 6. Common issues

`docker: command not found`

Install Docker Desktop on Windows, or Docker Engine on the remote Linux host.

`Illegal instruction`

Older TensorFlow Docker images expect CPU support for AVX instructions. If this happens, the host CPU is too old for this image.

`File not found` for models

Check that the expected `.h5` and `.lite` files exist under `diffchaser_models/`.

Results are not visible on the host

Make sure you mounted the `results` directory with `-v`.

## 7. Why this route is recommended

For this repository, Docker is safer than a native Windows install because:

- the code targets a Linux-first research environment
- TensorFlow 2.2 and Keras 2.4 are old enough to be painful on modern Windows setups
- `pyflann` is much more predictable inside Linux containers than on Windows

If you want, the next step can be a second container profile for the non-TFLite or PyTorch-related paths in `model_loader/`.
