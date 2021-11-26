# ----------------------------------
#          INSTALL & TEST
# ----------------------------------
install_requirements:
	@pip install -r requirements.txt

check_code:
	@flake8 scripts/* crypto_prediction/*.py

black:
	@black scripts/* crypto_prediction/*.py

test:
	@coverage run -m pytest tests/*.py
	@coverage report -m --omit="${VIRTUAL_ENV}/lib/python*"

clean:
	@rm -f */version.txt
	@rm -f .coverage
	@rm -fr */__pycache__ */*.pyc __pycache__
	@rm -fr build dist
	@rm -fr crypto_prediction-*.dist-info
	@rm -fr crypto_prediction.egg-info

install:
	@pip install . -U

all: clean install test black check_code

count_lines:
	@find ./ -name '*.py' -exec  wc -l {} \; | sort -n| awk \
        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./scripts -name '*-*' -exec  wc -l {} \; | sort -n| awk \
		        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./tests -name '*.py' -exec  wc -l {} \; | sort -n| awk \
        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''


# -----------------------------------
#            SERVER
# -----------------------------------

run_webserver_locally:
	uvicorn api.api:app --reload

# -----------------------------------
#            	  GCP
# -----------------------------------

### GCP Storage - - - - - - - - - - - - - - - - - - - - - -

BUCKET_NAME= 'crypto_prediction'


##### Training  - - - - - - - - - - - - - - - - - - - - - -

# will store the packages uploaded to GCP for the training
BUCKET_TRAINING_FOLDER = 'trainings'

##### Model - - - - - - - - - - - - - - - - - - - - - - - -

# not required here

### GCP AI Platform - - - - - - - - - - - - - - - - - - - -

##### Machine configuration - - - - - - - - - - - - - - - -

REGION=europe-west1

PYTHON_VERSION=3.7
FRAMEWORK=scikit-learn
RUNTIME_VERSION=1.15

##### Package params  - - - - - - - - - - - - - - - - - - -

PACKAGE_NAME=crypto_prediction
FILENAME=model.py

##### Job - - - - - - - - - - - - - - - - - - - - - - - - -

JOB_NAME=taxi_fare_training_pipeline_$(shell date +'%Y%m%d_%H%M%S')


run_locally:
	@python -m ${PACKAGE_NAME}.${FILENAME}

gcp_submit_training:
	gcloud ai-platform jobs submit training ${JOB_NAME} \
		--job-dir gs://${BUCKET_NAME}/${BUCKET_TRAINING_FOLDER} \
		--package-path ${PACKAGE_NAME} \
		--module-name ${PACKAGE_NAME}.${FILENAME} \
		--python-version=${PYTHON_VERSION} \
		--runtime-version=${RUNTIME_VERSION} \
		--region ${REGION} \
		--stream-logs


# -----------------------------------
#              DOCKER
# -----------------------------------

# base: https://kitt.lewagon.com/camps/735/challenges?path=07-Data-Engineering%2F04-Predict-in-production%2F03-GCR-cloud-run

GCLOUD_PROJECT_ID = crypto-prediction-333213
DOCKER_IMAGE_NAME = crypto_v1

# (re)build image it with the correct tag
docker_1build:
	docker build -t eu.gcr.io/$(GCLOUD_PROJECT_ID)/$(DOCKER_IMAGE_NAME) .

# test if it serves the api locally but from docker
docker_2test:
	docker run -e PORT=8000 -p 8000:8000 eu.gcr.io/$(GCLOUD_PROJECT_ID)/$(DOCKER_IMAGE_NAME)

# push the image to the 'google container registry'
docker_3push:
	docker push eu.gcr.io/$(GCLOUD_PROJECT_ID)/$(DOCKER_IMAGE_NAME)

docker_4deploy:
	gcloud run deploy --image eu.gcr.io/$(GCLOUD_PROJECT_ID)/$(DOCKER_IMAGE_NAME) --platform managed --region europe-west1
