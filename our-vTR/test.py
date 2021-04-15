import tensorflow as tf
import torch

def in_tf():
    alpha = 1000
    beta = 1 / alpha
    bkg = tf.constant([0.25, 0.25, 0.25, 0.25])
    bkg_tf = tf.cast(bkg, tf.float32)

    def calculate(x):
        alpha_x = tf.math.scalar_mul(alpha, x)
        ax_reduced = tf.math.reduce_max(alpha_x, axis=1)
        axr_expanded = tf.expand_dims(ax_reduced, axis=1)
        ax_sub_axre = tf.subtract(alpha_x, axr_expanded)
        softmaxed = tf.math.reduce_sum(tf.math.exp(ax_sub_axre), axis=1)
        sm_log_expanded = tf.expand_dims(tf.math.log(softmaxed), axis=1)
        axsaxre_sub_smle = tf.subtract(ax_sub_axre, sm_log_expanded)

        bkg_streched = tf.tile(bkg_tf, [ tf.shape(x)[0] ])
        bkg_stacked = tf.reshape(bkg_streched, [ tf.shape(x)[0], tf.shape(bkg_tf)[0] ])
        bkgs_log = tf.math.log(bkg_stacked)

        return tf.math.scalar_mul(beta, tf.subtract(axsaxre_sub_smle, bkgs_log))

    filt_list = tf.map_fn(calculate, x_tf)
    transf = tf.transpose(filt_list, [1, 2, 0])