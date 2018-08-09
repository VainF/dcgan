import tensorflow as tf 
import argparse, glob
import numpy as np 
import os
import cv2

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', required=True, help='path to dataset')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
    parser.add_argument('--image_size', type=int, default=64, help='the height/width of input image to network')
    parser.add_argument('--nz', type=int, default=200, help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=500, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate, default=0.0002')
    parser.add_argument('--val_num', type=int, default=80, help='number of generated image during validation')
    return parser
    
# get dataset loader
def get_lfw_dataset(root, img_size):
    def _parse_function(filename):
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_jpeg(image_string)
        image_resized = tf.image.resize_images(image_decoded, [img_size, img_size], method=tf.image.ResizeMethod.BILINEAR)
        return image_resized

    filenames = glob.glob(root+'/**/*.jpg')
    dataset_size = len(filenames)
    print('%d images loaded from %s'%(dataset_size, root))
    filenames = tf.constant(filenames)

    dataset = tf.data.Dataset.from_tensor_slices(filenames)
    dataset = dataset.map(_parse_function)
    return dataset, dataset_size

def print_shape(name, shape):
    log_info = "%s".ljust(20-len(name))
    print(log_info%name, shape)

def Discriminator(x, ndf, reuse=False):
    with tf.variable_scope('DCGAN_D', reuse=reuse) as sc:
        print('---- Discriminator ----')
        nc = 3
        n = tf.layers.conv2d(inputs=x, filters=ndf, kernel_size=[4,4], strides=[2,2], padding='SAME',use_bias=False, name='conv_0')
        n = tf.nn.leaky_relu(n)
        print_shape('D_0:',n.shape)
        # 128 x 128

        n = tf.layers.conv2d(inputs=n, filters=ndf*2, kernel_size=[4,4], strides=[2,2], padding='SAME',use_bias=False, name='conv_1')
        n = tf.layers.batch_normalization(inputs=n,epsilon=1e-5,momentum=0.1, training=True)
        n = tf.nn.leaky_relu(n)
        print_shape('D_1:',n.shape)
        # 64 x 64

        n = tf.layers.conv2d(inputs=n, filters=ndf*4, kernel_size=[4,4], strides=[2,2], padding='SAME',use_bias=False, name='conv_2')
        n = tf.layers.batch_normalization(inputs=n,epsilon=1e-5,momentum=0.1, training=True)
        n = tf.nn.leaky_relu(n)
        print_shape('D_2:',n.shape)
        # 32 x 32

        n = tf.layers.conv2d(inputs=n, filters=ndf*8, kernel_size=[4,4], strides=[2,2], padding='SAME',use_bias=False, name='conv_3')
        n = tf.layers.batch_normalization(inputs=n,epsilon=1e-5,momentum=0.1, training=True)
        n = tf.nn.leaky_relu(n)
        print_shape('D_3:',n.shape)
        # 4 x 4

        logits = tf.layers.conv2d(inputs=n, filters=1, kernel_size=[4,4], strides=[1,1], padding='VALID',use_bias=False, name='conv_4')
        print_shape('logits:',logits.shape)
        # 1 x 1 logits
    return logits


def Generator(z, ngf, reuse=False):
    with tf.variable_scope('DCGAN_G', reuse=reuse) as sc:
        print('---- Generator ----')
        nc = 3
        n = tf.layers.conv2d_transpose(inputs=z,filters=ngf*8, kernel_size=[4,4], strides=[1,1],padding='VALID',use_bias=False,name='deconv_0')
        n = tf.layers.batch_normalization(inputs=n, epsilon=1e-5,momentum=0.1, name='bn_0', training=True)
        n = tf.nn.relu(n)
        print_shape('G_0:',n.shape)
        # 4 x 4

        n = tf.layers.conv2d_transpose(inputs=n,filters=ngf*4, kernel_size=[4,4], strides=[2,2],padding='SAME',use_bias=False,name='deconv_1')
        n = tf.layers.batch_normalization(inputs=n,epsilon=1e-5,momentum=0.1, name='bn_1', training=True)
        n = tf.nn.relu(n)
        print_shape('G_1:',n.shape)
        # 8 x 8

        n = tf.layers.conv2d_transpose(inputs=n,filters=ngf*2, kernel_size=[4,4], strides=[2,2],padding='SAME',use_bias=False,name='deconv_2')
        n = tf.layers.batch_normalization(inputs=n,epsilon=1e-5,momentum=0.1, name='bn_2', training=True)
        n = tf.nn.relu(n)
        print_shape('G_2:',n.shape)
        # 16 x 16

        n = tf.layers.conv2d_transpose(inputs=n,filters=ngf, kernel_size=[4,4], strides=[2,2],padding='SAME',use_bias=False,name='deconv_3')
        n = tf.layers.batch_normalization(inputs=n,epsilon=1e-5,momentum=0.1, name='bn_3', training=True)
        n = tf.nn.relu(n)
        print_shape('G_3:',n.shape)
        # 32 x 32

        n = tf.layers.conv2d_transpose(inputs=n,filters=3, kernel_size=[4,4], strides=[2,2],padding='SAME',use_bias=False,name='deconv_4')
        x = tf.nn.tanh(n)
        print_shape('fake:', x.shape)
        # 64x64
    return x

def to_255(imgs):
    imgs = imgs*255
    imgs = np.where(imgs>255, 255, imgs)
    imgs = np.where(imgs<0, 0, imgs)
    return imgs

def main():
    args = get_parser().parse_args()
    print(args)

    Z = tf.placeholder(dtype=tf.float32, shape=[None, 1, 1, args.nz], name='Z') # latent code z
    X = tf.placeholder(dtype=tf.float32, shape=[None, args.image_size, args.image_size, 3], name='X') # real images from dataset
    try:
        os.mkdir('checkpoints')
    except: pass
    # Net
    fakes = Generator(Z, args.ngf)
    fake_logits = Discriminator(fakes, args.ndf)
    real_logits = Discriminator(X, args.ndf, reuse=True)
    
    # Dataset
    lfw_dataset, dataset_size = get_lfw_dataset(args.data_root, args.image_size)
    lfw_dataset = lfw_dataset.batch(args.batch_size)
    lfw_dataset = lfw_dataset.shuffle(args.batch_size*2)
    #lfw_dataset = lfw_dataset.prefetch(args.batch_size*2)
    lfw_iter = lfw_dataset.make_initializable_iterator()
    img_batch = lfw_iter.get_next()

    # Loss
    loss_D = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake_logits), logits=fake_logits) + 
                            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real_logits), logits=real_logits))
    loss_G = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(fake_logits), logits=fake_logits))
    # Saver
    var_G = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='DCGAN_G')
    var_D = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='DCGAN_D')
    saver_G = tf.train.Saver(var_G)
    saver_D = tf.train.Saver(var_D)
    #print(var_G, var_D)
        
    # Optimizer
    opt_G = tf.train.AdamOptimizer(args.lr).minimize(loss_G, var_list=var_G)
    opt_D = tf.train.AdamOptimizer(args.lr).minimize(loss_D, var_list=var_D)

    # GPU config
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True

    max_iter = dataset_size // args.batch_size
    # Random noises for validation
    val_noise = np.random.normal(size=(args.val_num, 1, 1, args.nz))
    try:
        os.mkdir('val')
    except: pass

    with tf.Session(config=tf_config) as sess:
        # Restore Model
        sess.run(tf.global_variables_initializer())
        try:
            saver_G.restore(sess, 'checkpoints/dcgan_g.ckpt')
            saver_D.restore(sess, 'checkpoints/dcgan_d.ckpt')
            print("Model Restored!")
        except: pass
        # Train Loop
        for ep in range(args.epochs):
            sess.run(lfw_iter.initializer)
            img_batch = lfw_iter.get_next()
            for itr in range(max_iter):
                X_batch = sess.run(img_batch)
                X_batch = X_batch/255
                #print(X_batch.dtype)
                if X_batch.shape[0]!=args.batch_size: continue
                Z_noise = np.random.normal(size=(args.batch_size, 1, 1, args.nz))
                _, d_loss = sess.run([opt_D,loss_D], feed_dict={X: X_batch, Z: Z_noise})
                _, g_loss = sess.run([opt_G,loss_G], feed_dict={X: X_batch, Z: Z_noise})
                print('Epoch %d/%d, Iter %d/%d: d_loss = %.9f, g_loss = %.9f'%(ep,args.epochs, itr,max_iter, d_loss, g_loss))
                
            # Checkpoints
            saver_G.save(sess, 'checkpoints/dcgan_g.ckpt')
            saver_D.save(sess, 'checkpoints/dcgan_d.ckpt')

            # Validation
            if ep%10==0:
                try:
                    os.mkdir('val/%d'%ep)
                except: pass
                g_imgs, = sess.run([fakes], feed_dict={Z: val_noise})
                g_imgs = to_255(g_imgs)
                cnt = 0
                # save images
                for img in g_imgs:
                    cv2.imwrite(os.path.join('val/%d'%ep, "ep_%d_img_%d.png"%(ep,cnt)) ,cv2.cvtColor(np.uint8(img), cv2.COLOR_RGB2BGR))
                    cnt+=1
        
if __name__=='__main__':
    main()