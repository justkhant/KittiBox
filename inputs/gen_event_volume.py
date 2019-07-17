import h5py
import tensorflow as tf
import numpy as np

def calc_floor_ceil_delta(x):
    x_fl = tf.floor(x)
    x_ce = tf.ceil(x)
    x_ce = tf.where(tf.greater(x_ce - x_fl, 0.1),
                    x_ce,
                    x_ce+1)
    dx_ce = x - x_fl
    dx_fl = x_ce - x
    return [[x_fl, dx_fl], [x_ce, dx_ce]]

def single_trilinear_step(x_, dx, y_, dy, t_, dt, p):
    events = tf.stack((y_, x_, t_), axis=-1)
    updates = tf.multiply(tf.multiply(tf.multiply(dx, dy), dt), p)
    return events, updates

# Generates an event volume (x-y-t) where the timestamps are interpolated to the neighboring bins.
def gen_interpolated_event_volume(events, volume_size, do_interp=True):
    event_shape = tf.shape(events)
    B = event_shape[0]
    [x, y, t, p] = tf.unstack(events, 4, axis=-1)

    if not do_interp:
        x = tf.round(x)
        y = tf.round(y)
        t = tf.round(t)

    x_shape = tf.shape(x)
    n_batches = x_shape[0]
    n_events = x_shape[1]

    xs = calc_floor_ceil_delta(x)
    ys = calc_floor_ceil_delta(y)
    ts = calc_floor_ceil_delta(t)

    inds = []
    updates = []

    for i in range(2):
        for j in range(2):
            for k in range(2):
                ind, update = single_trilinear_step(xs[i][0], xs[i][1],
                                                    ys[j][0], ys[j][1],
                                                    ts[k][0], ts[k][1],
                                                    1.)
                inds.append(ind)
                updates.append(update)
                 
    inds_stacked = tf.concat(inds, axis=1)
    inds_stacked = tf.cast(tf.reshape(inds_stacked, [-1, 3]), tf.int32)

    batch_inds = tf.expand_dims(tf.range(0, n_batches), axis=1)
    batch_inds = tf.tile(batch_inds, [1, n_events * 8])
    batch_inds = tf.reshape(batch_inds, [-1, 1])
    inds_stacked = tf.concat([batch_inds, inds_stacked],
                             axis=-1)

    updates_stacked = tf.concat(updates, axis=-1)
    updates_stacked = tf.multiply(updates_stacked, tf.tile(p, [1, 8]))
    updates_stacked = tf.reshape(updates_stacked, [-1])

    event_volume = tf.scatter_nd(tf.cast(inds_stacked, tf.int32),
                                 updates_stacked,
                                 tf.cast(volume_size, tf.int32))

    summed_volume = tf.reduce_sum(tf.abs(event_volume), axis=-1, keepdims=True)

    sum_event_img_vec = tf.reshape(summed_volume, [B, -1])
    nonzero_mask = tf.cast(tf.greater(sum_event_img_vec, 0), tf.float32)
    num_nonzero = tf.cast(tf.count_nonzero(sum_event_img_vec, axis=-1, keepdims=True), tf.float32)
    mean = tf.reduce_sum(tf.abs(sum_event_img_vec), axis=-1, keepdims=True) / num_nonzero
    var = tf.reduce_sum((tf.square(tf.abs(sum_event_img_vec)-mean)*nonzero_mask),
                        axis=-1,
                        keepdims=True)
    var /= num_nonzero

    mean = tf.expand_dims(tf.expand_dims(mean, axis=-1), axis=-1)
    var = tf.expand_dims(tf.expand_dims(var, axis=-1), axis=-1)

    #pos_event_values = tf.gather(sum_event_img_vec, tf.where(tf.greater(sum_event_img_vec, 0)))
    #mean, var = tf.nn.moments(pos_event_values,
    #                       axes=[0])

    bins = volume_size[-1]
   
    summed_volume_tiled = tf.tile(summed_volume, [1, 1, 1, bins])
    
    #event_volume = tf.sign(event_volume) * tf.clip_by_value(tf.abs(event_volume),
    #                                                        0,
    #                                                        mean+2*tf.sqrt(var))
    event_volume = tf.where(tf.less_equal(tf.abs(summed_volume_tiled - mean), 3 * tf.sqrt(var)),
                            event_volume,
                            tf.zeros(tf.shape(event_volume)))
    """
    event_volume = tf.where(tf.less_equal(tf.abs(summed_volume_tiled - mean), 3.*tf.sqrt(var)),
                            event_volume,
                            tf.clip_by_value(tf.abs(event_volume),
                                             mean-3.*tf.sqrt(var),
                                             mean+3.*tf.sqrt(var)))
    """
    #event_volume = tf.clip_by_value(event_volume,
    #                                mean-2.*tf.sqrt(var),
    #                                mean+2.*tf.sqrt(var))

    thresholded_event_img = tf.reduce_sum(tf.abs(event_volume), axis=-1)
    
    yx_inds = tf.reshape(tf.stack([y, x], axis=-1), [-1, 2])
    batch_inds = tf.expand_dims(tf.range(0, n_batches), axis=1)
    batch_inds = tf.tile(batch_inds, [1, n_events])
    batch_inds = tf.cast(tf.reshape(batch_inds, [-1, 1]), tf.float32)
    
    inds_stacked = tf.concat([batch_inds, yx_inds],
                             axis=-1)
    final_vals = tf.gather_nd(thresholded_event_img, tf.cast(inds_stacked, tf.int32))
    valid_vals = tf.reshape(tf.cast(tf.greater(final_vals, 0), tf.float32), tf.shape(p))
    
    #event_volume /= tf.reduce_max(event_volume)
    
    return event_volume, valid_vals
