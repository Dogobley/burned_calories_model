FROM python:3.12.2

RUN pip install asttokens==2.4.1 \
blinker==1.9.0 \
certifi==2024.8.30 \
charset-normalizer==3.4.0 \
click==8.1.7 \
colorama==0.4.6 \
comm==0.2.2 \
contourpy==1.3.0 \
cycler==0.12.1 \
debugpy==1.8.8 \
decorator==5.1.1 \
executing==2.1.0 \
Flask==3.0.3 \
fonttools==4.54.1 \
idna==3.10 \
ipykernel==6.29.5 \
ipython==8.29.0 \
itsdangerous==2.2.0 \
jedi==0.19.1 \
Jinja2==3.1.4 \
joblib==1.4.2 \
jupyter_client==8.6.3 \
jupyter_core==5.7.2 \
kiwisolver==1.4.7 \
MarkupSafe==3.0.2 \
matplotlib==3.9.2 \
matplotlib-inline==0.1.7 \
nest-asyncio==1.6.0 \
numpy==2.1.3 \
packaging==24.2 \
pandas==2.2.3 \
parso==0.8.4 \
pillow==11.0.0 \
platformdirs==4.3.6 \
prompt_toolkit==3.0.48 \
psutil==6.1.0 \
pure_eval==0.2.3 \
Pygments==2.18.0 \
pyparsing==3.2.0 \
python-dateutil==2.9.0.post0 \
pytz==2024.2 \
pyzmq==26.2.0 \
requests==2.32.3 \
scikit-learn==1.5.2 \
scipy==1.14.1 \
seaborn==0.13.2 \
six==1.16.0 \
stack-data==0.6.3 \
threadpoolctl==3.5.0 \
tornado==6.4.1 \
tqdm==4.67.0 \
traitlets==5.14.3 \
tzdata==2024.2 \
urllib3==2.2.3 \
waitress==3.0.1 \
wcwidth==0.2.13 \
Werkzeug==3.1.3 \
xgboost==2.1.2

WORKDIR /app
COPY ["model.bin", "backend.py", "./"]

EXPOSE 9696

ENTRYPOINT [ "waitress-serve", "--listen=0.0.0.0:9696", "backend:app" ]