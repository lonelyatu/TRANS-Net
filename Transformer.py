# -*- coding: utf-8 -*-

import tensorflow as tf

def batch_norm(input_,scope='BN',bn_train=True):

    return tf.contrib.layers.batch_norm(input_,scale=True,epsilon=1e-8,
                                        is_training=bn_train,scope=scope)

def LayerNorm(inputs, name):
    return tf.contrib.layers.layer_norm(inputs, scope=name)

def lrelu(inputs):
    return tf.nn.relu(inputs)

def GeLu(inputs):
    return 0.5 * inputs * (1 + tf.nn.tanh(inputs * 0.7978845608 * (1 + 0.044715 * inputs * inputs)))

def conv2d(input, filters, kernel_size, name, strides = (1,1), paddings = 'same',dilation_rate=(1, 1)):
    return tf.layers.conv2d(input, filters=filters, kernel_size=kernel_size, strides=strides,padding=paddings, 
                            dilation_rate=(1, 1),kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                            bias_initializer=tf.constant_initializer(0.01),
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4),
                            name=name)

def conv2d_transpose(input,filters,kernel_size,name,paddings='same',strides=[2,2]):
    return tf.layers.conv2d_transpose(input,filters,kernel_size,strides=strides,padding=paddings,
                            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4),
                            bias_initializer=tf.constant_initializer(0.01),name=name)

def get_split(InputTemp, num, split_dim, concat_dim):

    OutTemp = tf.split(InputTemp, num, axis=split_dim)
    splitTensor = OutTemp[0]
    for i in range(len(OutTemp) - 1):
        splitTensor = tf.concat([splitTensor, OutTemp[i+1]], 1)
    return splitTensor
        
def extract_patches(InputTemp, NumTemp):
    batch, rows, cols, channels = InputTemp.shape
    InputTempExpandDim = tf.reshape(InputTemp, [batch, 1, rows, cols, channels])
    
    TempWeight = get_split(InputTempExpandDim, num=NumTemp, split_dim=2, concat_dim=0)
    TempHeight = get_split(TempWeight, num=NumTemp, split_dim=3, concat_dim=0)

    return TempHeight

def rebin_patches(InputTemp, NumTemp):

    TempWeight = get_split(InputTemp, num=NumTemp, split_dim=0, concat_dim=3)
    TempHeight = get_split(TempWeight, num=NumTemp, split_dim=0, concat_dim=2)

    return TempHeight

def EmbededPatches(inputs, patch_size, dim=1024, emb_dropout=0.1):
    batch, rows, cols, channels = inputs.shape
    num_patches = rows // patch_size
    splitTensor = extract_patches(inputs, num_patches)
    _, num, _, _, _ = splitTensor.shape
    split_patches = tf.reshape(splitTensor, [batch, num, -1]) # [batch, num, patch_dim]
    _, _, patch_dim = split_patches.shape     
    
    weights = tf.get_variable("weights", [patch_dim, dim], tf.float32, tf.random_normal_initializer(stddev=0.01))
    bias = tf.get_variable("bias2", [dim], initializer=tf.constant_initializer(0.0))
    embeded_patches = tf.matmul(split_patches, weights) + bias # [batch, num, dim]

######### V1    
#    cls_tokens = tf.get_variable("cls_tokens", [1, 1, dim], tf.float32, tf.random_normal_initializer(stddev=0.01))
#    cls_tokens_repeat = tf.repeat(cls_tokens, repeats=batch, axis=0)
  
#    embeded_patches_cat = tf.concat([embeded_patches, cls_tokens_repeat], 1)

#    pos_embedding = tf.get_variable("pos_embedding", [1, num + 1, dim], tf.float32, tf.random_normal_initializer(stddev=0.01))
    
#    embeded_patches_pos = embeded_patches_cat + pos_embedding
############

########### V2
    pos_embedding = tf.get_variable("pos_embedding", [1, num, dim], tf.float32, tf.random_normal_initializer(stddev=0.01))
    embeded_patches_pos = embeded_patches + pos_embedding
##############
    
    embeded_patches_pos_drop = tf.nn.dropout(embeded_patches_pos, 1-emb_dropout)
    
    return embeded_patches_pos_drop # V2  [batch, num, dim] # V1 [batch, num+1, dim]

def Attention(inputs, dim, heads=8, dim_head=64, dropout=0., name ='attn'):
    with tf.variable_scope(name):
        b, n, _, h = *inputs.shape, heads
        inner_dim = dim_head * heads
        
        weights1 = tf.get_variable(name + 'weight1', [dim, inner_dim * 3], tf.float32, tf.random_normal_initializer(stddev=0.01))
        qkv = tf.matmul(inputs, weights1)
        q, k, v = tf.split(qkv, 3, axis=-1)
        q = tf.transpose(tf.reshape(q, [b, n, h, -1]), [0, 2, 1, 3])
        k = tf.transpose(tf.reshape(k, [b, n, h, -1]), [0, 2, 1, 3])
        v = tf.transpose(tf.reshape(v, [b, n, h, -1]), [0, 2, 1, 3]) # [b, h, n, d]

        scale = dim_head ** -0.5
        dots = tf.einsum('b h i d, b h j d -> b h i j', q, k) * scale
        
        attn = tf.nn.softmax(dots, axis=-1)
        out = tf.einsum('b h i j, b h j d -> b h i d', attn, v)
        out_reshape = tf.reshape(tf.transpose(out, [0, 2, 1, 3]), [b, n, -1])
        
        if not (heads == 1 and dim_head == dim):
            weights2 = tf.get_variable(name + 'weight2', [inner_dim, dim], tf.float32, tf.random_normal_initializer(stddev=0.01))
            bias2 = tf.get_variable("bias2", [dim], initializer=tf.constant_initializer(0.0))
            out_final = tf.matmul(out_reshape, weights2) + bias2
            out_final_drop = tf.nn.dropout(out_final, 1-dropout)
            return out_final_drop

        else:
            return out_reshape    

def FeedForWard(inputs, dim, hidden_dim, dropout, name='feed'):
    with tf.variable_scope(name):
        weights1 = tf.get_variable("weights1", [dim, hidden_dim], tf.float32, tf.random_normal_initializer(stddev=0.01))
        bias1 = tf.get_variable("bias1", [hidden_dim], initializer=tf.constant_initializer(0.0)) 
        linear1 = tf.matmul(inputs, weights1) + bias1 
        gelu1 = GeLu(linear1)
        drop1 = tf.nn.dropout(gelu1, 1-dropout)
        
        weights2 = tf.get_variable("weights2", [hidden_dim, dim], tf.float32, tf.random_normal_initializer(stddev=0.01))
        bias2 = tf.get_variable("bias2", [dim], initializer=tf.constant_initializer(0.0)) 
        linear2 = tf.matmul(drop1, weights2) + bias2
        drop2 = tf.nn.dropout(linear2, 1-dropout)
        return drop2

def TransformerEncoder(inputs, dim, depth, heads, dim_head, mlp_dim, dropout):
    feed = inputs
    with tf.variable_scope('TransformerEncoder'):
        for d in range(depth):
            attn = Attention(LayerNorm(feed, 'LN_Attn'+str(d)), dim, heads, dim_head, dropout, 'attn'+str(d)) + feed
            feed = FeedForWard(LayerNorm(attn, 'LN_Feed'+str(d)), dim, mlp_dim, dropout, 'feed'+str(d)) + attn
    return feed
    
def TransformerMoudle(inputs, dropout_rate =1.0, train_sign=True, name='transformer'):
    with tf.variable_scope(name):
        # inputs == [batch, rows, cols, channels]
        batch, rows, cols, channels = inputs.shape
        patch_size = 16 # patch size
        dim = 1024 # latent dimension
        emb_dropout = 0.5 * dropout_rate # dropout for self-attention
        depth = 3 # num of transformer
        heads = 8 # num of head
        dim_head = 64 # head dimension
        mlp_dim = 512 # mlp dimension
        dropout = 0.5 * dropout_rate# dropout for MLP
        embeded_patches = EmbededPatches(inputs, patch_size, dim, emb_dropout)
        transformer = TransformerEncoder(embeded_patches, dim, depth, heads, dim_head, mlp_dim, dropout) # V2 [batch, num, dim]
        decoder = tf.reshape(transformer, [batch, rows // patch_size, cols // patch_size, dim])
        down1 = conv2d(inputs, channels, [4,4], 'down1', [4,4], 'valid')
        
        conv1 = conv2d(decoder, channels, [1, 1], 'conv1')
        
        upsamp1 = conv2d_transpose(conv1, channels, [5,5], 'up1', strides=[patch_size // 4, patch_size // 4])
        
        concat1 = tf.concat([down1, upsamp1], 3)
        
        out1_conv7 = conv2d(concat1, channels, [3,3], 'conv7')
        out1_bn7 = batch_norm(out1_conv7, 'bn7', bn_train=train_sign)
        out1_lrelu7 = lrelu(out1_bn7)
        out1_conv8 = conv2d(out1_lrelu7, channels, [3,3], 'conv8')
        out1_bn8 = batch_norm(out1_conv8, 'bn8', bn_train=train_sign)
        out1_lrelu8 = lrelu(out1_bn8)
        
        upsamp2 = conv2d_transpose(out1_lrelu8, channels, [5,5], 'up2', strides=[patch_size //4, patch_size // 4])
        concat2 = tf.concat([inputs, upsamp2],3)
        
        out1_conv9 = conv2d(concat2, channels, [3,3], 'conv9')
        out1_bn9 = batch_norm(out1_conv9, 'bn9', bn_train=train_sign)
        out1_lrelu9 = lrelu(out1_bn9)
        out1_conv10 = conv2d(out1_lrelu9, channels, [3,3], 'conv10')
        out1_bn10 = batch_norm(out1_conv10, 'bn10', bn_train=train_sign)
        out1_lrelu10 = lrelu(out1_bn10)
        
        out1_conv11 = conv2d(out1_lrelu10, channels, [3,3], 'conv11')
            
        return out1_conv11

    
    