# Deep Neural Network Test
## V1.0 Beta 3
### Made by [abgache](https://abgache.pro/)
Activation Function : _Sigmoïd_

## Saving system
**Filetype** : _".nna"_
**Format ** : json encoded with base64
### Json Format : 
list = [(4, 2, 5, 3), (0.5, 0, 1, ...), ((0.45, 1, 0.78, ...), (0.97, 0, 0.2)), (1, 0.12, ...)]
        ^  ^  ^  ^        ^^^                       ^^^^^^^^^^^^                    ^^^
        e  h  m  l         wa                            wh                          wo
e = input neurons / h = How many hidden layers / m = How many neurons in earch hidden layer / l = output neurons / 
wa = weight between the input layer and the first hidden layer (on commence par le premier neuronne d'entrée avec le 1er caché, puis le 2nd caché, ...) /
wh = weight between all the hidden layers (1 groupe entre la 1er et 2nd couche, 2 grp entre la 2nd et 3eme, ... (mm systeme que wa)) / 
wo = weight between the last hidden layer and the ouput layer (mm systeme que wa)
## Défaults :
