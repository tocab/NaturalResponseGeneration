# NaturalResponseGeneration
Learning to generate responses for questions with deep learning methods

This project does not contain the data files. For learning to generate answers for questions,
the _Yahoo! Answers Comprehensive Questions and Answers version 1.0 (multi part)_
was used. The text data was pre processed and prepared to fit into the neural network.

## Usage

To build the project, you can use the Dockerfile. It will also download the pre-trained
weights of the different models to make predictions.

When you have built the image, run it with the command
```
docker run -p 8080:8080 image-name/image-id
```

A little python bottle server will be started. Navigate to http://0.0.0.0:8080/infer
to start generating answers for your own question. There are two input fields like on
Yahoo! Questions. The subject field can be a short comprehension of the question and the
content field could be a longer text. It is also possible to leave one of the fields
empty.

## Todo
* Change web framework to flask
* Deploying a web version of this project

# Sources:

BAHDANAU , Dzmitry ; CHO , Kyunghyun ; BENGIO , Yoshua: Neural Machine Translation by Jointly Learning to Align and Translate, 2014

SUTSKEVER , Ilya ; VINYALS , Oriol ; LE , Quoc V.: Sequence to Sequence Learning with Neural Networks, 2014

GOODFELLOW , Ian J. ; POUGET-ABADIE , Jean ; MIRZA , Mehdi ; XU , Bing
; WARDE-FARLEY , David ; OZAIR , Sherjil ; COURVILLE , Aaron ; BENGIO ,
Yoshua: Generative Adversarial Networks, 2014

YU, Lantao ; ZHANG, Weinan ; WANG , Jun ; YU, Yong: SeqGAN: Sequence Generative Adversarial Nets with Policy Gradient, 2016

XU, Zhen ; LIU, Bingquan ; WANG, Baoxun ; SUN, Chengjie ; WANG, Xiao-long ; WANG, Zhuoran ; QI, Chao: Neural Response Generation via GAN with an Approximate Embedding Layer, 2017