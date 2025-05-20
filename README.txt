Saludos Doctora Alexandra Gonzalez

Muchas gracias por ampliar la fecha para la presentación de este trabajo :)

Por favor, para ejecutar el mlflow, ingrese por consola a la carpeta donde esta esté documento y ejecute:
mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 9090

El proyecto contiene un notebook (jupiter-lab) llamado penguins.ipynb el cual entrena un modelo en mlflow para la predicción
de especies de pinguinos; se tomo un dataset muy conocido en internet (penguins.csv), con datos como:
epecie, en que isla habitan, tamaño del pico, profundidad del pico, peso y sexo de pinguinos.

Tambien encontrará el archivo app.py el cual lanza la aplicación mlflow y presentará una interfaz (index.htm) en la cual
se deben ingresar datos y el sistema predice que especie de pinguino es.

El proyecto está subido en github en la siguiente dirección: https://github.com/georgevill/tareamlflow.git

Que tenga un buen día!

