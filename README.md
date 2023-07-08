# Stock-Market-Prediction
A Course Project to predict the Stock price using LSTM

In the financial markets, stock indexes are essential because they offer useful information on the general performance of the industries or the entire market. Stock market indices are used by analysts and researchers to analyze the market thoroughly and identify patterns. They employ indices to analyze past performance, monitor market cycles, and assess the effects of economic events on particular industries. There are two types of investment strategies, passive and active. A passive investment approach, commonly referred to as passive management or index investing tries to replicate the performance of a certain market index or benchmark. Active investment is an investment strategy that involves actively picking and managing individual stocks, bonds, or other assets with the goal of outperforming the overall market or a specific benchmark.

What are LSTMs?
LSTMs, short for Long Short-Term Memory networks, are a type of recurrent neural network (RNN) architecture that is particularly effective in modelling sequential data and capturing long-term dependencies. The key feature of LSTMs is their ability to selectively remember or forget information over long time intervals, which makes them well-suited for tasks involving long-term dependencies. LSTMs achieve this through a memory cell, which is the core component of the network. The memory cell maintains an internal state and three gating mechanisms that regulate the flow of information: the input gate, the forget gate, and the output gate.
The input gate determines how much new information is added to the memory cell, while the forget gate controls the extent to which old information is retained. These gates are adaptive and learn to selectively update the memory cell based on the input and the previous state. The output gate then determines how much of the memory cell's content is used to produce the output of the LSTM. 
In practice, LSTMs are implemented as layers in a neural network, with each LSTM unit processing one step of the input sequence. The network can be trained using backpropagation through time, which extends the backpropagation algorithm to recurrent connections. This enables the LSTM to learn and adapt its internal parameters to effectively model the input sequence and make predictions or classifications based on it.

In this project, we have used bi-directional LSTMs to predict the Closing price of the next two days.
What are Bi-Directional LSTMs?
Bi-directional LSTMs (BiLSTMs) are an extension of the standard LSTM architecture that take into account both past and future information when making predictions. While regular LSTMs process sequences in a forward manner, from the beginning to the end, BiLSTMs process the sequence in both directions simultaneously. 
In a Bidirectional LSTM, the input sequence is divided into two parts: the forward sequence, which goes from the beginning to the end, and the backward sequence, which goes from the end to the beginning. Each part is processed independently by separate LSTM layers. The outputs from the two layers at each time step are then combined, usually by concatenation or addition, to form the final representation for that time step.
The advantage of BiLSTMs is that they can capture context from both the past and the future of each time step in the input sequence. For example, in natural language processing tasks like sentiment analysis, understanding the context both before and after a particular word can help better determine its sentiment. Similarly, in speech recognition tasks, knowing the context before and after a certain audio segment can improve the accuracy of phoneme or word recognition.
During training, the BiLSTM is usually fed with the entire sequence at once, and the backward pass through the network is performed by backpropagating the gradients from the final output to the initial time step. This process allows the BiLSTM to update its parameters based on the information from both directions.
While BiLSTMs are powerful for many sequence-to-sequence tasks, they come at the cost of increased computational complexity due to the bidirectional nature. This means that training a BiLSTM model can be more computationally intensive compared to a traditional unidirectional LSTM. However, the improved performance in tasks that benefit from bidirectional context often justifies this computational overhead.
The Libraries used to implement the same are :
1. Numpy : For Array Operations
2. Pandas : For Data manipulation and analysis
3. Sklearn : Imported sklearn.metrices.r2_score to calculate the R2 Score to know the model's accuracy
4. Keras : a) Imported the keras.models.Sequential to create deep learning models where an instance of the Sequential class is created and                model layers are created and added to it
           b) Imported keras.layers.LSTM to implement the LSTM model with the following arguments :
              i)   unit : Positive integer, dimensionality of the output space
              ii)  activation : Activation function to use. Here, we used ReLU (Rectified Linear Unit) as the activation function.
                                Default function is tanh.
           c) Imported keras.layers.Dense to create a deeply connected neural network layer with the argument as unit , which is a positive               integer, dimensionality of the output space 
           d) Imported keras.layers.Bidirectional with the arguments as :
              i)   layer : Here, we gave the LSTM layer as the argument.
              ii)  input_shape
5. Tensorflow : Imported tensorflow.keras.utils and set random_seed(1) to save the set of random values generated due to the Stochastic                    nature of the optimisation techniques.
Instruction : 1. Imported the dataset and the actual values.
              2. Split the sequence in an input/output form using the split_sequence define in the code.
              3. Applied various LSTM architectures like : Vanilla LSTM , Stacked LSTM, Bidirectional LSTM
              4. Among all the LSTMs, Bidirectional LSTM gave the most accurate result.
   
   
