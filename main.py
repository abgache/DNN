# V1.0 Beta 3
# Made by abgache - https://abgache.pro/
# Neural Networks test
# Activation Function : Sigmoïd

# save : list = [(4, 2, 5, 3), (0.5, 0, 1, ...), ((0.45, 1, 0.78, ...), (0.97, 0, 0.2)), (1, 0.12, ...)]
#                 ^  ^  ^  ^        ^^^                       ^^^^^^^^^^^^                    ^^^
#                 e  h  m  l         wa                            wh                          wo
# e = input neurons / h = How many hidden layers / m = How many neurons in earch hidden layer / l = output neurons / 
# wa = weight between the input layer and the first hidden layer (on commence par le premier neuronne d'entrée avec le 1er caché, puis le 2nd caché, ...) /
# wh = weight between all the hidden layers (1 groupe entre la 1er et 2nd couche, 2 grp entre la 2nd et 3eme, ... (mm systeme que wa)) / 
# wo = weight between the last hidden layer and the ouput layer (mm systeme que wa) 

from math import *
from random import *
import base64, json

e = 2.71828182845904523536028747135266249775724709369995

#========== Functions ==========
def sigmoid(n: float) -> float:
    global e
    return 1/(1+(e**(-n)))
def loss(result: float, w_result: float) -> float:
    return (result - w_result)**2
def rn(max_value: int = 100000) -> float:
    if max_value < 1:
        raise ValueError("The maximum value need to be superior to 1.")
    return (random.randint(1, max_value)) / max_value
def check_network(network: list): # True means that the network is working and False means that it does not work
    # basic network informations
    input_neurons = network[0][0]
    ouput_neurons = network[0][3]
    hidden_layers = network[0][1]
    hidden_layers_density = network[0][2]

    # Check that there is more than 1 input or ouput neuron    
    if not input_neurons >= 1:
        return (False, "Pas assez de neuronnes d'entrée")
    if not ouput_neurons >= 1:
        return (False, "Pas assez de neuronnes de sortie")
    # Check if these is more or less weight than possible between the input layer and the 1st hidden layer
    if (input_neurons * hidden_layers_density) != len(network[1]):
        return (False, "Il y a plus ou moin de poids que possible entre la couche d'entrée et la première couche caché")
    # Check if these is more or less weight than possible between the output layer and the last hidden layer
    if (ouput_neurons * hidden_layers_density) != len(network[3]):
        return (False, "Il y a plus ou moin de poids que possible entre la dernière couche caché et la couche de sortie")
    # Check if these is more or less weight than possible between all the hidden layers
    if (hidden_layers - 1) != len(network[2]):
        return (False, "Il y a plus ou moin de poids que possible entre les couches cachées")
    return (True, "Working fine !")
def predict(network: list, input_value: tuple):
    # Network check to avoid future errors
    network_state = check_network(network)
    if not network_state[0]:
        raise ValueError(network_state[1])
    if network[0][0] != len(input_value):
        raise ValueError("Il y a plus ou moins de valeurs d'entrées que de neuronnes d'entrées.")
    flh = () # First hidden layer
    # Calculate the first hidden layer
    for i in range(len(network[1])):
        flh += (sigmoid(input_value[i]*(network[1][i])),)
    # Calculation of hidden layers
    hl = ()
    tmp = ()
    for i in range(len(network[2])):
        for e in range(len(network[2][i])):
            tmp += sigmoid(flh[e] * network[2][i][e],)
        hl += (tmp,)
        tmp = ()
    # Calculation of the result
    tmp = hl[len(hl)-1]
    result = ()
    for i in range(len(network[3])):
        result += sigmoid(tmp[i]*network[3][i],)
    return result
def train(network: list, train_data: list, learning_rate: float = 0.1, epochs: int = 1000) -> list:
    """
    Trains the neural network by adjusting the weights.
    
    :param network: The neural network to train
    :param train_data: The training data, where each element contains a tuple
                        (input_values, expected_output)
    :param learning_rate: The learning rate for gradient descent
    :param epochs: The number of training iterations
    :return: The neural network after training
    """
    # Training Data Format
    # The training data is a crucial part of the neural network training process.
    # It must be organized as a list of tuples, where each tuple represents a training sample,
    # containing two tuples: one for the input data and one for the expected output.

    # Format:
    # train_data = [
    #     # Each item in the list represents a training sample
    #     # Each sample consists of two tuples:
    #     # 1. Input data (a tuple of values for each input neuron)
    #     # 2. Expected results (a tuple of values for each output neuron)
    #     ((inputs), (expected_outputs))
    # ]

    # Example:
    # train_data = [
    #     ((0, 1, 1), (0.44, 0.4, 0.9)),  # Inputs and expected results for the first sample
    #     ((0, 0), (0.89, 0.2)),           # Inputs and expected results for the second sample
    #     # Additional samples can be added in the same manner
    # ]

    # Notes:
    # 1. The number of input values in each tuple must match the number of neurons in the input layer of the network.
    # 2. The number of expected output values in each tuple must match the number of neurons in the output layer of the network.
    # 3. The training process can be performed on a training dataset containing multiple samples.
    # 4. The values in the tuples are usually real numbers (floats), representing the activation or intensity of the signal at each neuron.

    # Network check to avoid future errors
    network_state = check_network(network)
    if not network_state[0]:
        raise ValueError(network_state[1])

    # Loop over the number of epochs
    for epoch in range(epochs):
        # Boucle sur chaque élément du jeu d'entraînement
        for data in train_data:
            input_values, expected_output = data

            # Calculer les sorties du réseau
            output = predict(network, input_values)

            # Calculer l'erreur de perte (MSE)
            errors = [loss(expected_output[i], output[i]) for i in range(len(expected_output))]

            # Calculer les gradients pour chaque poids
            # D'abord, calculer les erreurs pour la dernière couche
            deltas_output = []
            for i in range(len(output)):
                delta = (expected_output[i] - output[i]) * output[i] * (1 - output[i])  # Gradient de la sigmoïde
                deltas_output.append(delta)

            # Calculer les erreurs pour les couches cachées
            deltas_hidden = []
            for i in range(len(network[2])-1, -1, -1):
                delta_layer = []
                for j in range(len(network[2][i])):
                    delta = sum([deltas_output[k] * network[3][k] for k in range(len(deltas_output))]) * network[2][i][j] * (1 - network[2][i][j])
                    delta_layer.append(delta)
                deltas_hidden.insert(0, delta_layer)

            # Mettre à jour les poids du réseau (descente de gradient)
            # Poids entre la dernière couche cachée et la sortie
            for i in range(len(network[3])):
                for j in range(len(network[2][-1])):
                    network[3][i] += learning_rate * deltas_output[i] * network[2][-1][j]

            # Poids entre les couches cachées
            for layer_idx in range(len(network[2])-1, 0, -1):
                for i in range(len(network[2][layer_idx])):
                    for j in range(len(network[2][layer_idx-1])):
                        network[2][layer_idx][i] += learning_rate * deltas_hidden[layer_idx][i] * network[2][layer_idx-1][j]

            # Poids entre la couche d'entrée et la première couche cachée
            for i in range(len(network[1])):
                network[1][i] += learning_rate * deltas_hidden[0][i] * input_values[i]

        # Affichage de la perte pour suivre l'entraînement
        if epoch % 100 == 0:
            total_loss = sum(errors) / len(errors)
            print(f"Epoch {epoch}/{epochs} - Loss: {total_loss}")

    return network
def new_network(inp_n: int, out_n: int, hid_n: int, m_hd_n: int, style = "random") -> list: 
    # inp_n: input neurons out_n: output hid_n: hidden layers m_hd_n number of hidden layers
    result = [(inp_n, m_hd_n, hid_n, out_n)]
    # Style = How the default weights / biais will be : r/random = random, f_1 = Full of 1's & f_0 = Full of 0's 
    if style == "random" or style == "r":
        # weight between the input layer and the first hidden layer
        tmp = () # temp variable reset
        for i in range(result[0][0]*result[0][2]):
            tmp += (rn(),)
        result += tmp
        tmp = ()
        # weight between the hidden layers
        tmp1 = () # 2nd temp variable reset
        for i in range(m_hd_n - 1): # Pour le nombre de couches cachées
            for e in range(hid_n ** 2): # Pour chaque duo de couches cachées
                tmp += (rn(),)
            tmp1 += (tmp,)
            tmp = ()
        result += tmp1
        tmp1 = ()
        # weight between the last hidden layer and the output layer
        tmp = () # temp variable reset
        for i in range(result[0][3]*result[0][2]):
            tmp += (rn(),)
        result += tmp
        tmp = ()
    elif style == "f1" or style == "f_1" or style == "full_1":
        # weight between the input layer and the first hidden layer
        tmp = () # temp variable reset
        for i in range(result[0][0]*result[0][2]):
            tmp += (1,)
        result += tmp
        tmp = ()
        # weight between the hidden layers
        tmp1 = () # 2nd temp variable reset
        for i in range(m_hd_n - 1): # Pour le nombre de couches cachées
            for i in range(hid_n ** 2): # Pour chaque duo de couches cachées
                tmp += (1,)
            tmp1 += (tmp,)
            tmp = ()
        result += tmp1
        tmp1 = ()
        # weight between the last hidden layer and the output layer
        tmp = () # temp variable reset
        for i in range(result[0][3]*result[0][2]):
            tmp += (1,)
        result += tmp
        tmp = ()
    elif style == "f0" or style == "f_0" or style == "full_0":
                # weight between the input layer and the first hidden layer
        tmp = () # temp variable reset
        for i in range(result[0][0]*result[0][2]):
            tmp += (0,)
        result += tmp
        tmp = ()
        # weight between the hidden layers
        tmp1 = () # 2nd temp variable reset
        for i in range(m_hd_n - 1): # Pour le nombre de couches cachées
            for i in range(hid_n ** 2): # Pour chaque duo de couches cachées
                tmp += (0,)
            tmp1 += (tmp,)
            tmp = ()
        result += tmp1
        tmp1 = ()
        # weight between the last hidden layer and the output layer
        tmp = () # temp variable reset
        for i in range(result[0][3]*result[0][2]):
            tmp += (0,)
        result += tmp
        tmp = ()
    else:
        raise ValueError(f"Invalid style : \"{style}\" is not a valid choise, you have to choose between : r/f1/f0 - Please read the documentation for more details.")
    return result
def save(data_list, filename): # ChatGPT
    json_data = json.dumps(data_list)
    byte_data = json_data.encode('utf-8')
    encoded_data = base64.b64encode(byte_data)
    with open(filename, 'wb') as file:
        file.write(encoded_data)
def load(filename): # ChatGPT
    with open(filename, 'rb') as file:
        encoded_data = file.read()
    byte_data = base64.b64decode(encoded_data)
    json_data = byte_data.decode('utf-8')
    data_list = json.loads(json_data)
    return data_list
def credits():
    print("V1.0 Beta 1")
    print("Made by abgache - https://abgache.pro/")
    print("Neural Networks test")
    print("Activation Function : Sigmoïd\n")
#==============================

if __name__ == "__main__":
    credits()
    a = input("1 - Create a new DNN\n2 - Load a DNN")
    if a == "1":
        nn = new_network(input("How many input neurons >>> "), input("How many neurons in a single hidden layer >>> "), input("How many hidden layers >>> "), input("How many ouput neurons >>> "))
        save(nn, (input("DNN name >>> ")+".nna"))
        a = input("1 - Train the DNN\n2 - Use the DNN")
        if a == "1":
            train(nn)
        elif a == "2":
            predict(nn, ())
