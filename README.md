# Video Analytics: UI

This is a proof of concept project focusing on activity recognition from video footage. The application lets user to select a video from the local storage and sends it to a model in the backend through REST API. The application also receives the feedback from the model and shows on the interface.

• The application is developed in Django.
• REST Api is used to communicate with the model.


## Usage

 1. Using Python 3.8, run `python -m venv env` to create a virtual environment
 2. Run `pip install -r requirements.txt` to install dependencies
 3. Run `cd app/` to change to `app/`
 3. Run `python manage.py runserver` to start development server
 4. Navigate to [http://127.0.0.1:8000](http://127.0.0.1:8000) to test
