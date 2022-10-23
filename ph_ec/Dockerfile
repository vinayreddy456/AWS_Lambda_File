FROM public.ecr.aws/lambda/python:3.8

# Copy function code
COPY app.py ${LAMBDA_TASK_ROOT}

# Install the function's dependencies using file requirements.txt
# from your project folder.

COPY requirements.txt  .
COPY creds.json .
COPY ec_xg.dat .
COPY ph_xg.dat .
COPY x_scaler.bin .
COPY y_EC_scaler.bin .
COPY y_PH_scaler.bin .
RUN  pip3 install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
CMD [ "app.handler" ]
