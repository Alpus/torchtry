FROM pytorch/pytorch:0.4_cuda9_cudnn7
LABEL maintainer="alpus@prisma-ai.com"

COPY requirements.txt requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

EXPOSE 8888

WORKDIR /workdir
ENTRYPOINT ["jupyter", "notebook", "--allow-root", "--ip=0.0.0.0", "--no-browser", "--NotebookApp.iopub_data_rate_limit=10000000", "--NotebookApp.iopub_msg_rate_limit=10000000"]
