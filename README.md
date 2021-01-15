System requirement:

Graphic card: Nvidia graphic card with memory over 8 GB

OS: Windows 10 

Softwares: Python3.7, CUDA 10.2

Usage:

python3 VISoR_Reconstruction/scripts/generate_reconstruction.py --dataset [dataset_file] --parameters [parameters_file] --output_path [output_path] [generated_file_path]

python3 VISoR_Reconstruction/scripts/run.py [generated_file_path]

Example:

python3 VISoR_Reconstruction/scripts/generate_reconstruction.py --dataset D:\workspace\data\sample-data\Data.visor --parameters D:\workspace\Software\visor-reconstruction\VISoR_Reconstruction\preset\mouse_fast.json --output_path D:\workspace\data\sample-reconstruction D:\workspace\data\sample-reconstruction\input.json

python3 VISoR_Reconstruction/scripts/run.py D:\workspace\data\sample-reconstruction\input.json
