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

ftest:
	@Write me

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

# ----------------------------------
#      UPLOAD PACKAGE TO PYPI
# ----------------------------------
PYPI_USERNAME=<AUTHOR>
build:
	@python setup.py sdist bdist_wheel

pypi_test:
	@twine upload -r testpypi dist/* -u $(PYPI_USERNAME)

pypi:
	@twine upload dist/* -u $(PYPI_USERNAME)

# -----------------------------------
#              DOCKER
# -----------------------------------

# base: https://kitt.lewagon.com/camps/735/challenges?path=07-Data-Engineering%2F04-Predict-in-production%2F03-GCR-cloud-run

GCLOUD_PROJECT_ID = lewagon-735
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


# -----------------------------------
#            EXTRA STUFF
# -----------------------------------

run_locally:
	uvicorn api.api:app --reload
