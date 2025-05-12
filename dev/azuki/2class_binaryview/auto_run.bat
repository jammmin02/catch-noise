echo off
SET project_name=noise-analyzer

echo [1] Docker Building Image...
docker build -t %project_name% .

echo [2] VSCode Open
start .

echo [3] Docker Running a container
docker run -it --rm -v %cd%:/app --name %project_name% %project_name%