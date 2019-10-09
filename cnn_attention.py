import tensorflow as tf
import keras

from keras.layers import Layer
from keras.layers import Conv2D, SeparableConv2D
from keras.layers import concatenate

from keras import initializers
from keras import backend as K
from keras.layers import Input
from keras.models import Model


class AttentionAugmentation2D(keras.models.Model):
    def __init__(self,Fout, k, dk, dv, Nh, relative=False):
        super(AttentionAugmentation2D,self).__init__()
        self.output_filters = Fout
        self.kernel_size = k
        self.depth_k = dk
        self.depth_v = dv
        self.num_heads = Nh
        self.relative = relative

        self.conv_out = SeparableConv2D(filters = Fout - dv,kernel_size = k,padding="same")
        self.qkv = Conv2D(filters = 2*dk + dv,kernel_size= 1)
        self.attn_out_conv = Conv2D(filters = dv,kernel_size = 1)

    def call(self,inputs):
        out = self.conv_out(inputs)
        shape = K.int_shape(inputs)
        if None in shape:
            shape = [-1 if type(v)==type(None) else v for v in shape]
        Batch,H,W,_ = shape

        flat_q, flat_k, flat_v = self.compute_flat_qkv(inputs)
        dkh = self.depth_k//self.num_heads
        logits = tf.matmul(flat_q, flat_k, transpose_b=True)
        if self.relative:
            h_rel_logits, w_rel_logits = self.relative_logits(self.q, H, W, self.num_heads,dkh)
            logits += h_rel_logits
            logits += w_rel_logits
        
        weights = K.softmax(logits,axis=-1)
        attn_out = tf.matmul(weights,flat_v)
        attn_out = K.reshape(attn_out, [Batch, self.num_heads, H, W, self.depth_v // self.num_heads])

        attn_out = self.combine_heads_2d(attn_out)
        attn_out = self.attn_out_conv(attn_out)
        output =  concatenate([out,attn_out],axis=3)
        return output

    def combine_heads_2d(self,x):
        # [batch, num_heads, height, width, depth_v // num_heads]
        transposed = K.permute_dimensions(x, [0, 2, 3, 1, 4])
        # [batch, height, width, num_heads, depth_v // num_heads]
        shape = K.int_shape(transposed)
        if None in shape:
            shape = [-1 if type(v)==type(None) else v for v in shape]
        batch, h , w, a , b = shape 
        ret_shape = [batch, h ,w, a*b]
        # [batch, height, width, depth_v]
        return K.reshape(transposed, ret_shape)

    def rel_to_abs(self,x):
        shape = K.shape(x)
        shape = [shape[i] for i in range(3)]
        B, Nh, L = shape
        col_pad = K.zeros((B, Nh, L, 1))
        x = K.concatenate([x, col_pad], axis=3)
        flat_x = K.reshape(x, [B, Nh, L * 2 * L])
        flat_pad = K.zeros((B, Nh, L-1))
        flat_x_padded = K.concatenate([flat_x, flat_pad], axis=2)
        final_x = K.reshape(flat_x_padded, [B, Nh, L+1, 2*L-1])
        final_x = final_x[:, :, :L, L-1:]
        return final_x

    def relative_logits_1d(self, q, rel_k, H, W, Nh, transpose_mask):
        rel_logits = tf.einsum('bhxyd,md->bhxym', q, rel_k)
        rel_logits = K.reshape(rel_logits, [-1, Nh*H, W, 2*W-1])
        rel_logits = self.rel_to_abs(rel_logits)
        rel_logits = K.reshape(rel_logits, [-1, Nh, H, W, W])
        rel_logits = K.expand_dims(rel_logits, axis=3)
        rel_logits = K.tile(rel_logits, [1, 1, 1, H, 1, 1])
        rel_logits = K.permute_dimensions(rel_logits, transpose_mask)
        rel_logits = K.reshape(rel_logits, [-1, Nh, H*W, H*W])
        return rel_logits


    def relative_logits(self,q,H, W, Nh,dkh):
        key_rel_w  = K.random_normal(shape = (int(2 * W - 1),dkh))
        key_rel_h  = K.random_normal(shape = (int(2 * H - 1),dkh))

        rel_logits_w = self.relative_logits_1d(q,key_rel_w, H, W, Nh, [0, 1, 2, 4, 3, 5])

        rel_logits_h = self.relative_logits_1d(K.permute_dimensions(q, [0, 1, 3, 2, 4]),key_rel_h, W, H, Nh, [0, 1, 4, 2, 5, 3])

        return rel_logits_h , rel_logits_w

    def split_heads_2d(self,q,Nh):
        batch, height,width,channels = K.int_shape(q)
        ret_shape = [-1,height,width,Nh,channels//Nh]
        split = K.reshape(q,ret_shape)
        transpose_axes = (0, 3, 1, 2, 4)
        split = K.permute_dimensions(split, transpose_axes)
        return split


    def compute_flat_qkv(self,inputs):

        qkv = self.qkv(inputs)
        B,H,W,_ = K.int_shape(inputs)
        q,k,v = tf.split(qkv,[self.depth_k,self.depth_k,self.depth_v],axis=3)

        dkh = self.depth_k // self.num_heads
        dvh = self.depth_v // self.num_heads
        q *= dkh ** -0.5

        self.q = self.split_heads_2d(q, self.num_heads)
        self.k = self.split_heads_2d(k, self.num_heads)
        self.v = self.split_heads_2d(v, self.num_heads)

        flat_q = K.reshape(q, [-1, self.num_heads, H * W, dkh])
        flat_k = K.reshape(k, [-1, self.num_heads, H * W, dkh])
        flat_v = K.reshape(v, [-1, self.num_heads, H * W, dvh])

        return flat_q, flat_k, flat_v
    
    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[-1] = self.output_filters
        return tuple(output_shape)

if __name__ == '__main__':

    #v = dk/out
    #k = dv/out

    #paper = v=k=0.25 Nh= 8

    ip = Input(shape=(386, 386, 64))
    augmented_conv1 = AttentionAugmentation2D(64, 3, 16, 16, 8, relative=True)(ip)

    model = Model(ip, augmented_conv1)
    model.summary()

    x = K.zeros([1000, 386, 386, 64])
    y = model(x)
    print("Attention Augmented Conv out shape : ", y.shape)
