import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

def cos_sim_tf(a, b=None):
    if b is None:
        normalize_a = tf.nn.l2_normalize(a, 1)
        return tf.matmul(normalize_a, normalize_a, transpose_b=True)
    else:
        normalize_a = tf.nn.l2_normalize(a, 1)        
        normalize_b = tf.nn.l2_normalize(b, 1)
        return tf.matmul(normalize_a, normalize_b, transpose_b=True)
    
def cos_sim_np(a):
    a = a / np.linalg.norm(a, axis=1, keepdims=True)        
    res = a @ a.T
    return res.astype(np.float32)