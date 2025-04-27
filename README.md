# Deep Neural Network Test
## V1.0 Beta 3
### Made by [abgache](https://abgache.pro/)
Foncion d'activation : _Sigmoïd_
fonction de perte: _MSE_
## Système de sauvegarde
**Filetype** : _".nna"_
**Format ** : json encodé en base64
### Json Format : 
dnn: list = [(4, 2, 5, 3), (0.5, 0, 1, ...), ((0.45, 1, 0.78, ...), (0.97, 0, 0.2)), (1, 0.12, ...)]                                                                                                                     
La première partie contient les paramètres du réseau, dans le cas présent, 4 est le nombre de neuronnes d'entrée, 2 le nombre de couches cachées, 5 la densité en neuronnes de chaque couche caché et 3 le nombre de neuronnes de sorties.
La seconde contient, en ordre, le poid entre la couche d'entrée et la première couche caché. 
La troisième est une liste de listes des poids entre toutes les couches cachées et finalement, la quatrième et dernière partie contient le poid entre la dernière couche caché et la couche de sortie.
## Defaults :
 - Aucune variations de quantitées de neuronnes dans les couches cachées  n'est possible.
 - Aucun changement de Fonction d'activation ou de Fonction de perte n'est possible.
