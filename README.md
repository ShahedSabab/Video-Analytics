# Video Analytics: UI

This is a proof of concept project focusing on activity recognition from video footage. The application lets user to select a video from the local storage and sends it to a model in the backend through REST API. The application also receives the feedback from the model and shows on the interface.

• The application is developed in Django. <br>
• It on docker container. <br>
• REST Api is used to communicate with the model. <br>
• The system triggers an email to the specified address if the model detects any unatuhorized activities.<br>


## How to run:
Just use the following commands from the cmd/terminal: 
>docker-compose -f docker-compose-deploy.yml up --build

Goto the following address from any browser:
>127.0.0.1:8080
