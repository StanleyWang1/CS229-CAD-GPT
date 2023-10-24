[1] SETUP the conda environment:
- make sure you're in the right directory (GitHub Repo)
- Run in terminal:
    conda env create -f environment.yml
- Run in terminal:
    conda activate CADGPT
- If using VSCode, make sure you are selecting the proper python executable matching the venv   
    You might get "cadquery not found" if this is not done...
- If you are getting a "OCP not found", run:
    conda install -c conda-forge -c cadquery ocp
    (this takes forever...)
- Need this for STEP import
    conda install -c conda-forge pythonocc-core=7.7.2
- Might also need (for Stanley's Apple Silicon Mac):
    pip install multimethod
    pip install typish
    pip install ezdxf
    pip install nptyping
    conda install -c conda-forge nlopt
    pip install casadi

[2] Deactivate/Uninstall Environment
- To deactivate this environment:
    conda deactivate
- To delete the environment:
    conda remove --name CADGPT --all
