# PhotonicNas




## CIFAR_Scripts
All final scripts of the experiments performed on CIFAR-10 Dataset

## GTSDB_Scripts
All final scripts of the experiments performed on GTSDB Dataset

## Deprecated
Code used during development, not useful anymore for the purpose of the experiment. Left in a separate folder for reference 

## NNi Installation
For the correct installation of the NNI framework, a version of python higher than
3.7 must be installed. Errors occur in case of an installation higher than 3.12, since
some libraries rely on earlier versions and some core functions become deprecated
when installing newer versions of python.
In this regard, the version used for the experiment is the following:
Python version: 3.11.9
Most packages are installed along with the command:
pip install --latest nni
Another option is to directly build the environment from the source code:
git clone https://github.com/microsoft/nni.git
cd nni
pip install --upgrade setuptools pip wheel
NNi provides the option to install all the dependency through the following
command, even though during the project it was not able to grab some of the core
libraries:
pip install nni[all]

A requirement.txt is present in the root folder.
| Package                      | Version        |
| ---------------------------- | -------------- |
| absl-py                      | 2.1.0          |
| aiohttp                      | 3.9.5          |
| aiosignal                    | 1.3.1          |
| anyio                        | 4.2.0          |
| argon2-cffi                  | 21.3.0         |
| argon2-cffi-bindings         | 21.2.0         |
| arrow                        | 1.3.0          |
| astor                        | 0.8.1          |
| asttokens                    | 2.0.5          |
| astunparse                   | 1.6.3          |
| async-lru                    | 2.0.4          |
| attrs                        | 23.1.0         |
| Babel                        | 2.11.0         |
| beautifulsoup4               | 4.12.2         |
| bleach                       | 4.1.0          |
| Brotli                       | 1.0.9          |
| cachetools                   | 5.5.0          |
| certifi                      | 2024.2.2       |
| cffi                         | 1.16.0         |
| chardet                      | 5.2.0          |
| charset-normalizer           | 2.0.4          |
| cloudpickle                  | 3.0.0          |
| colorama                     | 0.4.6          |
| comm                         | 0.2.1          |
| contourpy                    | 1.2.1          |
| cycler                       | 0.12.1         |
| debugpy                      | 1.6.7          |
| decorator                    | 5.1.1          |
| defusedxml                   | 0.7.1          |
| executing                    | 0.8.3          |
| fastjsonschema               | 2.16.2         |
| filelock                     | 3.11.0         |
| flatbuffers                  | 24.3.25        |
| fonttools                    | 4.53.1         |
| fqdn                         | 1.5.1          |
| frozenlist                   | 1.4.1          |
| fsspec                       | 2024.5.0       |
| gast                         | 0.6.0          |
| gitdb                        | 4.0.11         |
| GitPython                    | 3.1.43         |
| google-auth                  | 2.36.0         |
| google-auth-oauthlib         | 1.2.1          |
| google-pasta                 | 0.2.0          |
| graphviz                     | 0.20.3         |
| grpcio                       | 1.64.0         |
| h5py                         | 3.12.1         |
| idna                         | 3.7            |
| intel-openmp                 | 2021.4.0       |
| ipykernel                    | 6.28.0         |
| ipython                      | 8.20.0         |
| isoduration                  | 20.11.0        |
| jedi                         | 0.18.1         |
| Jinja2                       | 3.1.3          |
| joblib                       | 1.4.2          |
| json-tricks                  | 3.17.3         |
| json5                        | 0.9.6          |
| jsonpointer                  | 3.0.0          |
| jsonschema                   | 4.19.2         |
| jsonschema-specifications    | 2023.7.1       |
| jupyter_client               | 8.6.0          |
| jupyter_core                 | 5.5.0          |
| jupyter-events               | 0.8.0          |
| jupyter-lsp                  | 2.2.0          |
| jupyter_server               | 2.10.0         |
| jupyter-server-mathjax       | 0.2.6          |
| jupyter_server_terminals     | 0.4.4          |
| jupyterlab                   | 4.0.11         |
| jupyterlab-pygments          | 0.1.2          |
| jupyterlab_server            | 2.25.1         |
| keras                        | 3.7.0          |
| kiwisolver                   | 1.4.5          |
| larq                         | 0.13.3         |
| libclang                     | 18.1.1         |
| lightning                    | 2.2.5          |
| lightning-utilities          | 0.11.2         |
| Markdown                     | 3.6            |
| markdown-it-py               | 3.0.0          |
| MarkupSafe                   | 2.1.3          |
| matplotlib                   | 3.9.1          |
| matplotlib-inline            | 0.1.6          |
| mdurl                        | 0.1.2          |
| mistune                      | 2.0.4          |
| mkl                          | 2021.4.0       |
| ml-dtypes                    | 0.4.1          |
| mpmath                       | 1.3.0          |
| multidict                    | 6.0.5          |
| namex                        | 0.0.8          |
| nbclient                     | 0.8.0          |
| nbconvert                    | 7.10.0         |
| nbdime                       | 4.0.1          |
| nbformat                     | 5.9.2          |
| nbstripout                   | 0.7.1          |
| nest-asyncio                 | 1.6.0          |
| networkx                     | 3.3            |
| nni                          | 3.0            |
| notebook                     | 7.0.8          |
| notebook_shim                | 0.2.3          |
| numpy                        | 1.26.4         |
| nvidia-ml-py                 | 12.550.52      |
| oauthlib                     | 3.2.2          |
| opencv-python                | 4.10.0.84      |
| opt_einsum                   | 3.4.0          |
| optree                       | 0.13.1         |
| overrides                    | 7.4.0          |
| packaging                    | 23.2           |
| pandas                       | 2.2.2          |
| pandocfilters                | 1.5.0          |
| parso                        | 0.8.3          |
| pillow                       | 10.3.0         |
| pip                          | 24.3.1         |
| platformdirs                 | 3.10.0         |
| prettytable                  | 3.10.0         |
| prometheus-client            | 0.14.1         |
| prompt-toolkit               | 3.0.43         |
| protobuf                     | 4.25.5         |
| psutil                       | 5.9.0          |
| pure-eval                    | 0.2.2          |
| pyasn1                       | 0.6.1          |
| pyasn1_modules               | 0.4.1          |
| pycparser                    | 2.21           |
| Pygments                     | 2.15.1         |
| pyparsing                    | 3.1.2          |
| PySocks                      | 1.7.1          |
| python-dateutil              | 2.9.0.post0    |
| python-json-logger           | 2.0.7          |
| PythonWebHDFS                | 0.2.3          |
| pytorch-lightning            | 2.2.5          |
| pytorch-quantization         | 2.1.2          |
| pytz                         | 2024.1         |
| pywin32                      | 305.1          |
| pywinpty                     | 2.0.10         |
| PyYAML                       | 6.0.1          |
| pyzmq                        | 25.1.2         |
| referencing                  | 0.30.2         |
| requests                     | 2.31.0         |
| requests-oauthlib            | 2.0.0          |
| responses                    | 0.25.0         |
| rfc3339-validator            | 0.1.4          |
| rfc3986-validator            | 0.1.1          |
| rich                         | 13.9.4         |
| rpds-py                      | 0.10.6         |
| rsa                          | 4.9            |
| schema                       | 0.7.7          |
| scikit-learn                 | 1.5.0          |
| scipy                        | 1.13.1         |
| Send2Trash                   | 1.8.2          |
| setuptools                   | 75.6.0         |
| simplejson                   | 3.19.2         |
| six                          | 1.16.0         |
| smmap                        | 5.0.1          |
| sniffio                      | 1.3.0          |
| soupsieve                    | 2.5            |
| sphinx_glpi_theme            | 0.6            |
| stack-data                   | 0.2.0          |
| sympy                        | 1.12           |
| tbb                          | 2021.12.0      |
| tensorboard                  | 2.18.0         |
| tensorboard-data-server      | 0.7.2          |
| tensorboardX                 | 2.6.2.2        |
| tensorflow-estimator         | 2.15.0         |
| tensorflow-intel             | 2.15.0         |
| tensorflow-io-gcs-filesystem | 0.31.0         |
| termcolor                    | 2.5.0          |
| terminado                    | 0.17.1         |
| terminaltables               | 3.1.10         |
| threadpoolctl                | 3.5.0          |
| tinycss2                     | 1.2.1          |
| torch                        | 2.3.0+cu121    |
| torchaudio                   | 2.3.0+cu121    |
| torchmetrics                 | 1.4.0.post0    |
| torchvision                  | 0.18.0+cu121   |
| tornado                      | 6.3.3          |
| tqdm                         | 4.66.4         |
| traitlets                    | 5.7.1          |
| typeguard                    | 4.1.2          |
| types-python-dateutil        | 2.9.0.20240316 |
| typing_extensions            | 4.11.0         |
| tzdata                       | 2024.1         |
| uri-template                 | 1.3.0          |
| urllib3                      | 2.2.1          |
| wcwidth                      | 0.2.13         |
| webcolors                    | 24.6.0         |
| webencodings                 | 0.5.1          |
| websocket-client             | 1.8.0          |
| websockets                   | 12.0           |
| Werkzeug                     | 3.0.3          |
| wheel                        | 0.45.1         |
| win-inet-pton                | 1.1.0          |
| wrapt                        | 1.14.1         |
| yarl                         | 1.9.4          |


## Tensorboard
A .bat file is present in the utility folder to open logs in a browser window
