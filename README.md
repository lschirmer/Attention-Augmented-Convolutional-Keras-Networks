# Attention-Augmented-Convolutional-Networks-Keras

Implementation of Attention Augmented Convolutional Networks using Keras

For more details see the original paper: https://arxiv.org/pdf/1904.09925.pdf

Inline-style: 
![](/image_attn.png "Attention image")

#Example:
```python 
ip = Input(shape=(386, 386, 64))
augmented_conv1 = AttentionAugmentation2D(32, 3, 8, 8, 8, relative=True)(ip)

model = Model(ip, augmented_conv1)
model.summary()

x = K.zeros([1000, 386, 386, 64])
y = model(x)
print("Attention Augmented Conv out shape : ", y.shape)
```
