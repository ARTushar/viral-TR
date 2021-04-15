import tensorflow as tf
import torch


def calculate_tf(x):
    alpha = 1000
    beta = 1 / alpha
    bkg = tf.constant([0.25, 0.25, 0.25, 0.25])
    bkg_tf = tf.cast(bkg, tf.float32)

    print("x:", x)
    alpha_x = tf.math.scalar_mul(alpha, x)
    print("alpha_x:", alpha_x)
    ax_reduced = tf.math.reduce_max(alpha_x, axis=1)
    print("ax_reduced:", ax_reduced)
    axr_expanded = tf.expand_dims(ax_reduced, axis=1)
    print("axr_expanded:", axr_expanded)
    ax_sub_axre = tf.subtract(alpha_x, axr_expanded)
    print("ax_sub_axre:", ax_sub_axre)
    softmaxed = tf.math.reduce_sum(tf.math.exp(ax_sub_axre), axis=1)
    print("softmaxed:", softmaxed)
    sm_log_expanded = tf.expand_dims(tf.math.log(softmaxed), axis=1)
    print("sm_log_expanded:", sm_log_expanded)
    axsaxre_sub_smle = tf.subtract(ax_sub_axre, sm_log_expanded)
    print("axsaxre_sub_smle:", axsaxre_sub_smle)

    bkg_streched = tf.tile(bkg_tf, [ tf.shape(x)[0] ])
    print("bkg_streched:", bkg_streched)
    bkg_stacked = tf.reshape(bkg_streched, [ tf.shape(x)[0], tf.shape(bkg_tf)[0] ])
    print("bkg_stacked:", bkg_stacked)
    bkgs_log = tf.math.log(bkg_stacked)
    print("bkgs_log:", bkgs_log)

    compared = tf.subtract(axsaxre_sub_smle, bkgs_log)
    print("compared:", compared)
    scaled = tf.math.scalar_mul(beta, compared)
    print("scaled:", scaled)

    return scaled


def in_tf():
    x_tf = tf.random.poisson((12, 4, 1), 0.1)
    print("x_tf:", x_tf.shape)
    x_tf = tf.transpose(x_tf, [2, 0, 1])
    print("after transpose: x_tf:", x_tf.shape)
    filt_list = tf.map_fn(calculate_tf, x_tf)
    print("filt_list:", filt_list.shape)
    transf = tf.transpose(filt_list, [1, 2, 0])
    print("transf:", transf.shape)


def calculate(x):
    alpha = 1000
    beta = 1 / alpha
    distribution = torch.tensor([0.25, 0.25, 0.25, 0.25])
    distr_log = torch.log(distribution.repeat(x.shape[0], 1))
    print("distr_log:", distr_log)

    alpha_x = alpha * x
    print("alpha_x:", alpha_x)
    ax_max, _ = torch.max(alpha_x, dim=1, keepdim=True)
    print("ax_max:", ax_max)
    ax_sub_axmx = alpha_x - ax_max
    print("ax_sub_axmx:", ax_sub_axmx)
    exp_sum = torch.sum(torch.exp(ax_sub_axmx), dim=1, keepdim=True)
    print("exp_sum:", exp_sum)
    es_log = torch.log(exp_sum)
    print("es_log:", es_log)
    biased = ax_sub_axmx - es_log
    print("biased:", biased)

    compared = biased - distr_log
    print("compared:", compared)
    scaled = beta * compared
    print("scaled:", scaled)

    return scaled

# in_tf()
# calculate(torch.randn((12, 4)))

w = torch.ones(512, 4, 12)

def change(x):
    print("-------- x.shape --------", x.shape)
    distribution = torch.tensor([[0.25, 0.25, 0.25, 0.25]]).T
    print("distribution:", distribution.shape)
    distr_log = torch.log(distribution.repeat(1, 12))
    print("distr_log:", distr_log.shape)

    alpha = 1000
    alpha_x = alpha * x
    ax_max, _ = torch.max(alpha_x, dim=0, keepdim=True)
    ax_sub_axmx = alpha_x - ax_max
    exp_sum = torch.sum(torch.exp(ax_sub_axmx), dim=0, keepdim=True)
    es_log = torch.log(exp_sum)
    return (ax_sub_axmx - es_log - distr_log) / alpha

cw = torch.stack([change(z) for z in w])

print(w.shape)
print(cw.shape)